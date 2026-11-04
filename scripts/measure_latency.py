#!/usr/bin/env python3
"""Measure end-to-end latency: VAD segment "said" -> TTS first audio byte.

Mirrors the frontend pipeline (transcribe + KWS in parallel → evaluate →
chat → TTS first chunk) but from a single Python process, so numbers include
network + upstream processing time.

Wake turn uses assets/ref_voice.wav (its transcript contains 志玲姐姐).
Follow-up turn uses a CAM++ sample speaker audio; still inside the active
window, so it exercises the steady-state "ask → reply" path.
"""
from __future__ import annotations

import asyncio
import json
import struct
import sys
import time
import wave
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BACKEND = "http://127.0.0.1:8000"

REF_WAV = ROOT / "backend/assets/ref_voice.wav"
SPEAKER1_WAV = ROOT / (
    "voiceprint-service/models/models/"
    "iic--speech_campplus_sv_zh-cn_16k-common/snapshots/v1.0.0/examples/"
    "speaker1_b_cn_16k.wav"
)


def load_float32_16k_mono(path: Path) -> bytes:
    """Load any WAV and return raw little-endian float32 PCM @ 16kHz mono
    (the format the frontend Silero VAD hands to the backend)."""
    import numpy as np
    with wave.open(str(path)) as w:
        n_ch = w.getnchannels()
        sr = w.getframerate()
        sw = w.getsampwidth()
        assert sw in (2, 4), f"unsupported sample width {sw}"
        raw = w.readframes(w.getnframes())
    if sw == 2:
        dtype = np.int16
        scale = 32768.0
    else:
        dtype = np.int32
        scale = 2147483648.0
    arr = np.frombuffer(raw, dtype=dtype).astype(np.float32) / scale
    if n_ch > 1:
        arr = arr.reshape(-1, n_ch).mean(axis=1)
    if sr != 16000:
        # naive linear resample, good enough for latency measurement
        target_len = int(len(arr) * 16000 / sr)
        idx = np.linspace(0, len(arr) - 1, target_len)
        arr = np.interp(idx, np.arange(len(arr)), arr).astype(np.float32)
    return arr.tobytes()


def ms(t0: float) -> float:
    return (time.monotonic() - t0) * 1000


async def run_turn(
    client: httpx.AsyncClient,
    session_id: str,
    label: str,
    audio_pcm: bytes,
) -> dict | None:
    print(f"\n=== {label} ===")
    # The frontend creates the turn at VAD *speech-start*, i.e. before the
    # user finishes. We do it here (untimed) so the timed region below starts
    # at "VAD end" — the moment the complete segment is available.
    turn = (
        await client.post(
            f"{BACKEND}/api/session/{session_id}/turns",
            json={"source": "vad", "assistant_speaking": False},
        )
    ).json()
    turn_id = turn["turn_id"]

    t0 = time.monotonic()
    t_asr = time.monotonic()
    asr_task = asyncio.create_task(
        client.post(
            f"{BACKEND}/api/transcribe?session_id={session_id}",
            content=audio_pcm,
            headers={"Content-Type": "application/octet-stream"},
        )
    )
    kws_task = asyncio.create_task(
        client.post(
            f"{BACKEND}/api/kws/detect",
            content=audio_pcm,
            headers={"Content-Type": "application/octet-stream"},
        )
    )
    asr_resp, kws_resp = await asyncio.gather(asr_task, kws_task)
    asr_resp.raise_for_status()
    kws_resp.raise_for_status()
    asr = asr_resp.json()
    kws = kws_resp.json()
    print(f"[asr+kws] {ms(t_asr):6.0f} ms  text={asr.get('text','')!r}  kws.detected={kws.get('detected')}")

    t_eval = time.monotonic()
    files = {"audio": ("utterance.pcm", audio_pcm, "application/octet-stream")}
    data = {
        "text": asr.get("text", ""),
        "language": asr.get("language", ""),
        "kws_detected": "true" if kws.get("detected") else "false",
    }
    if kws.get("keyword"):
        data["kws_keyword"] = kws["keyword"]
    if kws.get("available") is not None:
        data["kws_available"] = "true" if kws["available"] else "false"
    eval_resp = await client.post(
        f"{BACKEND}/api/session/{session_id}/turns/{turn_id}/evaluate",
        files=files,
        data=data,
    )
    eval_resp.raise_for_status()
    ev = eval_resp.json()
    speaker = (ev.get("speaker") or {}).get("speaker_id") or (ev.get("speaker") or {}).get("id")
    print(
        f"[voice_gate+decision] {ms(t_eval):6.0f} ms  "
        f"should_reply={ev.get('should_reply')}  decision={ev.get('decision') or {}}  "
        f"reason={ev.get('reason')!r}"
    )

    if not ev.get("should_reply"):
        print(f"⛔ turn not eligible for chat; skipping. reason={ev.get('reason')!r}")
        return

    t_chat = time.monotonic()
    chat_resp = await client.post(
        f"{BACKEND}/api/chat",
        json={"session_id": session_id, "turn_id": turn_id},
    )
    chat_resp.raise_for_status()
    chat = chat_resp.json()
    reply = chat["reply"]
    print(f"[chat/LLM] {ms(t_chat):6.0f} ms  reply={reply[:60]!r}")

    t_tts = time.monotonic()
    headers_ms = None
    first_chunk_ms = None
    total_bytes = 0
    async with client.stream(
        "POST",
        f"{BACKEND}/api/tts",
        json={"text": reply, "instruction": chat.get("tts_instruction") or ""},
    ) as res:
        if res.status_code != 200:
            body = await res.aread()
            print(f"⛔ TTS failed: {res.status_code} {body[:200]!r}")
            return
        headers_ms = ms(t_tts)
        sample_rate = res.headers.get("x-sample-rate")
        async for chunk in res.aiter_raw():
            if first_chunk_ms is None:
                first_chunk_ms = ms(t_tts)
            total_bytes += len(chunk)

    print(
        f"[tts] headers={headers_ms:.0f} ms  first-chunk={first_chunk_ms:.0f} ms  "
        f"bytes={total_bytes}  sample_rate={sample_rate}"
    )

    e2e = ms(t0)
    print(f"==> END-TO-END  VAD-end → first TTS byte: {e2e:.0f} ms")
    return {
        "label": label,
        "asr_kws": ms(t_asr),
        "voice_gate": ms(t_eval),
        "chat": ms(t_chat),
        "tts_first_chunk": first_chunk_ms,
        "e2e": e2e,
    }


async def main() -> int:
    if not REF_WAV.exists():
        print(f"missing audio: {REF_WAV}", file=sys.stderr)
        return 1

    ref_audio = load_float32_16k_mono(REF_WAV)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        s = (await client.post(f"{BACKEND}/api/session/start")).json()
        session_id = s["session_id"]
        print(f"session={session_id}  (audio={REF_WAV.name}, "
              f"{len(ref_audio)/4/16000:.2f}s of float32@16k)")

        results = []
        # Turn 1 enrolls the speaker (resolve) and joins via the wake phrase in
        # the ref audio's transcript. Turns 2-3 reuse the same identity inside
        # the active window — the steady-state "ask -> first sound" path.
        for i in range(3):
            r = await run_turn(client, session_id, f"Turn {i+1} (ref_voice)", ref_audio)
            if r:
                results.append(r)

        if results:
            e2es = [r["e2e"] for r in results]
            print("\n--- summary (VAD-end -> first TTS byte) ---")
            for r in results:
                print(f"  {r['label']:22s}  {r['e2e']:7.0f} ms")
            if len(e2es) > 1:
                steady = sum(e2es[1:]) / (len(e2es) - 1)
                print(f"  steady-state avg (turn 2+): {steady:.0f} ms")
            else:
                print(f"  only 1 measured turn: {e2es[0]:.0f} ms")

        perf = (await client.get(f"{BACKEND}/api/session/{session_id}/performance")).json()
        print("\n--- backend performance events (per-stage) ---")
        for e in perf["events"]:
            print(
                f"  {e.get('stage'):18s}  {e.get('elapsed_ms'):8.1f} ms  "
                f"source={e.get('source'):8s}  turn={(e.get('turn_id') or '')[:8]:8s}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
