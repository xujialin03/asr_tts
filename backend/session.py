"""Per-session state for the voice interaction pipeline.

A Session keeps only application-local state:
  - the remote ASR streaming session id (start/chunk/finish), and
  - the wake/continuous-conversation activation window.

The dedicated Hermes agent owns the assistant conversation and its history.
"""

from __future__ import annotations

import asyncio
import secrets
import sys
import time
from dataclasses import dataclass, field

import httpx

import config

ASR_CHUNK_BYTES = 8000 * 4  # 500ms @ 16kHz float32 mono


def _pad_asr_chunk(body: bytes) -> bytes:
    if not body:
        return body
    # Remote ASR is happiest with fixed 500ms float32 chunks. Pad short final
    # chunks with silence instead of sending arbitrary short buffers.
    if len(body) < ASR_CHUNK_BYTES:
        return body + (b"\x00" * (ASR_CHUNK_BYTES - len(body)))
    # Keep float32 alignment if a caller ever sends odd byte counts.
    rem = len(body) % 4
    if rem:
        return body + (b"\x00" * (4 - rem))
    return body


@dataclass
class Session:
    id: str
    asr_session_id: str | None = None
    # Last transcript seen from a chunk; used as a fallback if `finish` hangs.
    last_asr: dict = field(default_factory=lambda: {"language": "", "text": ""})
    active_until: float = 0.0
    chat_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # ---- ASR lifecycle (relayed to the remote ASR service) ----
    async def asr_start(self, client: httpx.AsyncClient) -> str:
        r = await client.post(
            f"{config.ASR_BASE_URL}/api/start",
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        r.raise_for_status()
        self.asr_session_id = r.json()["session_id"]
        self.last_asr = {"language": "", "text": ""}
        return self.asr_session_id

    async def asr_chunk(self, client: httpx.AsyncClient, body: bytes) -> dict:
        # The remote ASR session ends after `finish`; transparently start a new
        # one so the same backend Session can be reused across conversations.
        if not self.asr_session_id:
            await self.asr_start(client)
        r = await self._post_chunk(client, body)
        if r.status_code == 400:
            # Remote session died mid-stream: restart once and retry.
            await self.asr_start(client)
            r = await self._post_chunk(client, body)
        r.raise_for_status()
        j = r.json()
        self.last_asr = {"language": j.get("language", ""), "text": j.get("text", "")}
        return j

    async def _post_chunk(self, client: httpx.AsyncClient, body: bytes):
        return await client.post(
            f"{config.ASR_BASE_URL}/api/chunk",
            params={"session_id": self.asr_session_id},
            content=_pad_asr_chunk(body),
            headers={"Content-Type": "application/octet-stream"},
            timeout=httpx.Timeout(15.0, connect=5.0),
        )

    async def asr_finish(self, client: httpx.AsyncClient) -> dict:
        # Idempotent: a second finish (e.g. button + VAD racing) returns the
        # last known transcript instead of erroring.
        if not self.asr_session_id:
            return dict(self.last_asr)
        try:
            r = await client.post(
                f"{config.ASR_BASE_URL}/api/finish",
                params={"session_id": self.asr_session_id},
                timeout=httpx.Timeout(5.0, connect=10.0),
            )
            r.raise_for_status()
            j = r.json()
        except (httpx.HTTPError, ValueError):
            # Remote `finish` hung or errored: fall back to the last chunk's
            # cumulative transcript so the turn still completes.
            j = dict(self.last_asr)
        # The remote session is now closed (or abandoned); drop it so the next
        # recording starts fresh.
        self.asr_session_id = None
        return j

    # ---- turn gate ----
    def activate(self, seconds: float | None = None) -> None:
        duration = config.ACTIVE_WINDOW_SECONDS if seconds is None else seconds
        duration = max(1.0, min(float(duration), 300.0))
        # Never shorten an already longer active window.
        self.active_until = max(self.active_until, time.time() + duration)

    def is_active(self) -> bool:
        return time.time() <= self.active_until

    def active_remaining(self) -> float:
        return max(0.0, self.active_until - time.time())

    def match_interrupt(self, text: str) -> tuple[bool, str]:
        """Return (matched, remaining_text_after_interrupt_phrase)."""
        normalized = (text or "").strip()
        for pattern in config.INTERRUPT_PATTERNS:
            match = pattern.search(normalized)
            if match:
                remainder = (normalized[: match.start()] + normalized[match.end() :]).strip(
                    " ，,。.!！?？、"
                )
                return True, remainder
        return False, normalized

    def should_reply(self, text: str, source: str = "vad") -> tuple[bool, str]:
        text = (text or "").strip()
        if not text:
            return False, "空识别结果"
        if source == "push_to_talk":
            self.activate()
            return True, "手动按住说话"
        if any(w and w in text for w in config.WAKE_WORDS):
            self.activate()
            return True, "命中唤醒词"
        if time.time() <= self.active_until:
            self.activate()
            return True, "连续对话窗口内"
        return False, "未命中唤醒词且不在连续对话窗口"

    # ---- Hermes turn ----
    async def _chat_hermes(self, user_text: str) -> str:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            config.HERMES_CLIENT_PATH,
            "ask",
            user_text,
            "--timeout",
            str(config.HERMES_TIMEOUT_SECONDS),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.HERMES_TIMEOUT_SECONDS + 10,
            )
        except BaseException:
            if process.returncode is None:
                process.kill()
                try:
                    await asyncio.wait_for(process.communicate(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
            raise

        if process.returncode != 0:
            detail = stderr.decode(errors="replace").strip()
            raise RuntimeError(detail or f"Hermes client exited with {process.returncode}")
        reply = stdout.decode(errors="replace").strip()
        if not reply:
            raise RuntimeError("Hermes returned an empty response")
        return reply

    async def chat(self, user_text: str) -> str:
        """Send one turn to the dedicated Hermes agent."""
        user_text = (user_text or "").strip()
        if not user_text:
            return ""

        async with self.chat_lock:
            reply = await self._chat_hermes(user_text)
            self.activate()
            return reply


class SessionManager:
    def __init__(self) -> None:
        self._store: dict[str, Session] = {}

    def create(self) -> Session:
        sid = secrets.token_hex(16)
        s = Session(id=sid)
        self._store[sid] = s
        return s

    def get(self, sid: str) -> Session | None:
        return self._store.get(sid)

    def get_or_create(self, sid: str | None) -> Session:
        if sid and sid in self._store:
            return self._store[sid]
        return self.create()

    def delete(self, sid: str) -> bool:
        return self._store.pop(sid, None) is not None


# One shared manager for the whole app.
sessions = SessionManager()
