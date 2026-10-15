"""Backend for the voice-chat pipeline.

Relays the remote Qwen3-ASR and Breeze-TTS services and sends assistant turns
to the dedicated Hermes agent. Local sessions keep only ASR and interaction
state; Hermes owns the assistant conversation history.

Frontend flow:
  POST /api/session/start   -> create local interaction session -> {session_id}
  POST /api/chunk           -> stream float32 PCM to ASR        -> {language, text}
  POST /api/finish          -> finish ASR                       -> {language, text}
  POST /api/chat            -> send text to Hermes              -> {reply, tts_instruction}
  POST /api/tts             -> synthesize speech (streaming PCM)
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import config
from session import ASR_CHUNK_BYTES, sessions

# Long read timeout: a chunk call blocks until the remote returns a transcript.
client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

app = FastAPI(title="Voice Chat Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await client.aclose()


# TTS sample rate is advertised by the remote service; fetch once at startup.
TTS_SAMPLE_RATE = 24000
# Reference audio (for voice cloning) loaded once into memory.
TTS_REF_BYTES: bytes | None = None
# Remote Breeze-TTS only supports one inference at a time.
TTS_LOCK = asyncio.Lock()
TTS_REQUEST_IDS = itertools.count(1)
logger = logging.getLogger("uvicorn.error")


@app.on_event("startup")
async def _startup() -> None:
    global TTS_SAMPLE_RATE, TTS_REF_BYTES
    try:
        r = await client.get(f"{config.TTS_BASE_URL}/health", timeout=10.0)
        TTS_SAMPLE_RATE = int(r.json().get("sample_rate", 24000))
    except Exception:
        pass
    if config.TTS_REF_AUDIO:
        try:
            with open(config.TTS_REF_AUDIO, "rb") as f:
                TTS_REF_BYTES = f.read()
        except OSError:
            TTS_REF_BYTES = None


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "asr": config.ASR_BASE_URL,
        "assistant": "hermes",
        "tts": config.TTS_BASE_URL,
        "kws": config.KWS_BASE_URL if config.KWS_ENABLED else "disabled",
    }


# ---- session lifecycle ----
@app.post("/api/session/start")
async def session_start() -> JSONResponse:
    # Only create the local app session here. Remote ASR sessions are started
    # lazily when audio is actually sent; otherwise VAD/chat-only usage leaks
    # idle remote ASR sessions and eventually makes the ASR service hang.
    s = sessions.create()
    return JSONResponse({"session_id": s.id, "asr_session_id": None})


def _get_session(session_id: str):
    s = sessions.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    return s


# ---- ASR relay ----
@app.post("/api/chunk")
async def chunk(request: Request, session_id: str) -> JSONResponse:
    s = _get_session(session_id)
    body = await request.body()
    try:
        j = await s.asr_chunk(client, body)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"ASR chunk failed: {e}")
    return JSONResponse(j)


@app.post("/api/finish")
async def finish(session_id: str) -> JSONResponse:
    s = _get_session(session_id)
    try:
        j = await s.asr_finish(client)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"ASR finish failed: {e}")
    return JSONResponse(j)


@app.post("/api/transcribe")
async def transcribe(request: Request, session_id: str) -> JSONResponse:
    """Transcribe a complete utterance (float32 PCM 16k mono) without finish().

    Silero VAD gives us a full speech segment. The remote ASR's /api/finish is
    currently prone to hanging, so this path sends the whole segment as chunks
    and returns the latest cumulative chunk result.
    """
    s = _get_session(session_id)
    body = await request.body()
    if not body:
        return JSONResponse({"language": "", "text": ""})
    try:
        await s.asr_start(client)
        # Remote ASR expects fixed 500ms little-endian float32 chunks.
        result = {"language": "", "text": ""}
        for off in range(0, len(body), ASR_CHUNK_BYTES):
            chunk = body[off : off + ASR_CHUNK_BYTES]
            if chunk:
                result = await s.asr_chunk(client, chunk)
        # Abandon the remote session instead of calling finish(), which can hang.
        s.asr_session_id = None
        return JSONResponse(result)
    except httpx.HTTPStatusError as e:
        s.asr_session_id = None
        detail = e.response.text[:500] if e.response is not None else str(e)
        raise HTTPException(
            status_code=502,
            detail=f"ASR transcribe failed: {e.response.status_code if e.response else ''} {detail}",
        )
    except httpx.HTTPError as e:
        s.asr_session_id = None
        raise HTTPException(status_code=502, detail=f"ASR transcribe failed: {type(e).__name__}: {e}")


@app.post("/api/voiceprint/resolve")
async def resolve_voiceprint(request: Request, session_id: str) -> JSONResponse:
    """Resolve VAD audio to a stable speaker ID, creating an identity if unknown."""
    s = _get_session(session_id)
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="audio body is empty")
    try:
        response = await client.post(
            f"{config.VOICEPRINT_BASE_URL}/v1/resolve",
            content=body,
            headers={"Content-Type": "application/octet-stream"},
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        raise HTTPException(
            status_code=502,
            detail=f"voiceprint resolve failed: {exc.response.status_code} {detail}",
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"voiceprint resolve failed: {type(exc).__name__}: {exc}",
        ) from exc
    s.speaker_id = result["speaker_id"]
    speaker = result.get("speaker") or {}
    s.speaker_display_name = speaker.get("display_name")
    s.speaker_preferred_name = speaker.get("preferred_name")
    s.speaker_identity_named = bool(
        s.speaker_preferred_name
        or (
            s.speaker_display_name
            and not s.speaker_display_name.startswith("访客 ")
        )
    )
    result["identity_named"] = s.speaker_identity_named
    result["audio_ref"] = s.store_voice_audio(body)
    return JSONResponse(result)


class VoiceprintUserUpdate(BaseModel):
    display_name: str | None = None
    preferred_name: str | None = None


async def _voiceprint_request(
    method: str,
    path: str,
    *,
    content: bytes | None = None,
    json_body: dict | None = None,
    params: dict | None = None,
) -> dict | list | None:
    try:
        response = await client.request(
            method,
            f"{config.VOICEPRINT_BASE_URL}{path}",
            content=content,
            json=json_body,
            params=params,
            headers={"Content-Type": "application/octet-stream"} if content is not None else None,
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        response.raise_for_status()
        return response.json() if response.content else None
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text[:500]) from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"voiceprint service failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.get("/api/voiceprint/users")
async def voiceprint_users() -> JSONResponse:
    return JSONResponse(await _voiceprint_request("GET", "/v1/users"))


@app.get("/api/voiceprint/users/{speaker_id}")
async def voiceprint_user(speaker_id: str) -> JSONResponse:
    return JSONResponse(await _voiceprint_request("GET", f"/v1/users/{speaker_id}"))


@app.patch("/api/voiceprint/users/{speaker_id}")
async def update_voiceprint_user(speaker_id: str, req: VoiceprintUserUpdate) -> JSONResponse:
    changes = req.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(status_code=400, detail="no fields to update")
    result = await _voiceprint_request(
        "PATCH", f"/v1/users/{speaker_id}", json_body=changes
    )
    sessions.update_speaker_profile(
        speaker_id,
        result.get("display_name"),
        result.get("preferred_name"),
    )
    return JSONResponse(result)


@app.delete(
    "/api/voiceprint/users/{speaker_id}",
    status_code=204,
    response_class=Response,
)
async def delete_voiceprint_user(speaker_id: str) -> Response:
    await _voiceprint_request("DELETE", f"/v1/users/{speaker_id}")
    return Response(status_code=204)


@app.get("/api/voiceprint/users/{speaker_id}/voiceprints")
async def voiceprint_templates(speaker_id: str) -> JSONResponse:
    return JSONResponse(
        await _voiceprint_request("GET", f"/v1/users/{speaker_id}/voiceprints")
    )


@app.post("/api/voiceprint/users/{speaker_id}/voiceprints/current", status_code=201)
async def add_current_voiceprint(
    speaker_id: str,
    audio_ref: str,
    label: str = "voice-command",
) -> JSONResponse:
    audio = sessions.find_voice_audio(speaker_id, audio_ref)
    if audio is None:
        raise HTTPException(status_code=404, detail="audio reference expired or does not belong to speaker")
    result = await _voiceprint_request(
        "POST",
        f"/v1/users/{speaker_id}/voiceprints",
        content=audio,
        params={"label": label},
    )
    return JSONResponse(result, status_code=201)


@app.delete("/api/voiceprint/users/{speaker_id}/voiceprints")
async def delete_voiceprint_templates(speaker_id: str) -> JSONResponse:
    return JSONResponse(
        await _voiceprint_request("DELETE", f"/v1/users/{speaker_id}/voiceprints")
    )


@app.delete(
    "/api/voiceprint/users/{speaker_id}/voiceprints/{voiceprint_id}",
    status_code=204,
    response_class=Response,
)
async def delete_voiceprint_template(speaker_id: str, voiceprint_id: str) -> Response:
    await _voiceprint_request(
        "DELETE", f"/v1/users/{speaker_id}/voiceprints/{voiceprint_id}"
    )
    return Response(status_code=204)


@app.post("/api/kws/detect")
async def kws_detect(request: Request) -> JSONResponse:
    """Proxy complete VAD audio to the optional, replaceable KWS service."""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="audio body is empty")
    if not config.KWS_ENABLED:
        return JSONResponse(
            {"available": False, "detected": False, "keyword": None, "engine": "disabled"}
        )
    try:
        response = await client.post(
            f"{config.KWS_BASE_URL}/v1/detect",
            content=body,
            headers={"Content-Type": "application/octet-stream"},
            timeout=httpx.Timeout(15.0, connect=2.0),
        )
        response.raise_for_status()
        result = response.json()
        result["available"] = True
        return JSONResponse(result)
    except (httpx.HTTPError, ValueError) as exc:
        if config.KWS_FAIL_OPEN:
            logger.warning("KWS unavailable, using exact text fallback: %s", exc)
            return JSONResponse(
                {
                    "available": False,
                    "detected": False,
                    "keyword": None,
                    "engine": "fallback",
                }
            )
        raise HTTPException(
            status_code=502,
            detail=f"KWS failed: {type(exc).__name__}: {exc}",
        ) from exc


# ---- Hermes chat ----
class DecideReq(BaseModel):
    session_id: str
    text: str
    source: str = "vad"
    assistant_speaking: bool = False
    kws_detected: bool = False
    kws_keyword: str | None = None
    kws_available: bool | None = None


@app.post("/api/decide")
async def decide(req: DecideReq) -> JSONResponse:
    s = _get_session(req.session_id)
    interrupt = False
    reply_text = req.text.strip()

    if req.assistant_speaking:
        interrupt, reply_text = s.match_interrupt(req.text)
        if not interrupt:
            should_reply, reason = False, "播放期间未命中打断词，视为环境干扰"
        elif reply_text:
            s.activate()
            should_reply, reason = True, "命中打断词并包含后续问题"
        else:
            should_reply, reason = False, "命中纯打断指令"
    else:
        should_reply, reason = s.should_reply(
            req.text,
            req.source,
            kws_detected=req.kws_detected,
            kws_keyword=req.kws_keyword,
        )

    return JSONResponse(
        {
            "should_reply": should_reply,
            "interrupt": interrupt,
            "reply_text": reply_text,
            "reason": reason,
            "speaker_id": s.speaker_id,
            "kws_detected": req.kws_detected,
            "kws_keyword": req.kws_keyword,
            "kws_available": req.kws_available,
            "active": s.is_active(),
            "active_remaining": round(s.active_remaining(), 1),
        }
    )


class TouchReq(BaseModel):
    session_id: str
    seconds: float | None = None
    force: bool = False


@app.post("/api/touch")
async def touch(req: TouchReq) -> JSONResponse:
    s = _get_session(req.session_id)
    # Speech-start may only extend an already active session. TTS playback is
    # allowed to force an extension because the assistant is already replying.
    was_active = s.is_active()
    if was_active or req.force:
        s.activate(req.seconds)
    return JSONResponse(
        {
            "active": s.is_active(),
            "active_remaining": round(s.active_remaining(), 1),
            "extended": was_active or req.force,
        }
    )


class ChatReq(BaseModel):
    session_id: str
    text: str


@app.post("/api/chat")
async def chat(req: ChatReq) -> JSONResponse:
    s = _get_session(req.session_id)
    try:
        reply = await s.chat(req.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Hermes call failed: {e}")
    return JSONResponse(
        {
            "reply": reply.text,
            "tts_instruction": reply.instruction,
            "speaker_id": s.speaker_id,
            "active": s.is_active(),
            "active_remaining": round(s.active_remaining(), 1),
        }
    )


# ---- TTS (streaming PCM) ----
class TtsReq(BaseModel):
    text: str
    instruction: str | None = Field(default=None, max_length=500)


@app.post("/api/tts")
async def tts(req: TtsReq) -> StreamingResponse:
    """Relay the remote TTS service's streaming PCM (16-bit mono @ sample_rate)."""
    request_id = next(TTS_REQUEST_IDS)
    started_at = time.monotonic()
    text = (req.text or "").strip()
    text_preview = text.replace("\n", " ")[:80]
    logger.info(
        "TTS request #%s received chars=%s text=%r locked=%s",
        request_id,
        len(text),
        text_preview,
        TTS_LOCK.locked(),
    )
    if not text:
        logger.warning("TTS request #%s rejected: empty text", request_id)
        raise HTTPException(status_code=400, detail="text required")

    form = {"text": text, "cfg_scale": str(config.TTS_CFG_SCALE)}
    instruction = (req.instruction or "").strip() or config.TTS_PROMPT
    if instruction:
        form["instruction"] = instruction

    # Attach the fixed reference audio + its transcript so the cloned voice
    # stays consistent across every request.
    files = None
    if TTS_REF_BYTES:
        form["ref_text"] = config.TTS_REF_TEXT
        files = {"ref_audio": ("ref_voice.wav", TTS_REF_BYTES, "audio/wav")}

    if TTS_LOCK.locked():
        logger.warning(
            "TTS request #%s rejected: previous request still holds lock",
            request_id,
        )
        raise HTTPException(status_code=409, detail="TTS 正在合成上一条，请稍后再试")

    await TTS_LOCK.acquire()
    logger.info("TTS request #%s acquired lock", request_id)
    stream_cm = client.stream(
        "POST",
        f"{config.TTS_BASE_URL}/v1/audio/speech",
        data=form,
        files=files,
        timeout=httpx.Timeout(120.0, connect=10.0),
    )
    upstream = None
    try:
        upstream = await stream_cm.__aenter__()
        logger.info(
            "TTS request #%s upstream status=%s content_type=%s",
            request_id,
            upstream.status_code,
            upstream.headers.get("content-type", ""),
        )
        if upstream.status_code != 200:
            detail = (await upstream.aread()).decode(errors="replace")
            logger.error(
                "TTS request #%s upstream failed status=%s detail=%s",
                request_id,
                upstream.status_code,
                detail[:500],
            )
            raise HTTPException(
                status_code=502,
                detail=f"TTS failed: {upstream.status_code} {detail}",
            )
    except BaseException as exc:
        if upstream is not None:
            await stream_cm.__aexit__(type(exc), exc, exc.__traceback__)
        TTS_LOCK.release()
        logger.exception(
            "TTS request #%s failed before streaming after %.2fs; lock released",
            request_id,
            time.monotonic() - started_at,
        )
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(
            status_code=502,
            detail=f"TTS upstream connection failed: {type(exc).__name__}: {exc}",
        ) from exc

    async def relay():
        total_bytes = 0
        try:
            async for chunk in upstream.aiter_raw():
                total_bytes += len(chunk)
                yield chunk
            logger.info(
                "TTS request #%s stream completed bytes=%s elapsed=%.2fs",
                request_id,
                total_bytes,
                time.monotonic() - started_at,
            )
        except BaseException:
            logger.exception(
                "TTS request #%s stream interrupted bytes=%s elapsed=%.2fs",
                request_id,
                total_bytes,
                time.monotonic() - started_at,
            )
            raise
        finally:
            await stream_cm.__aexit__(None, None, None)
            TTS_LOCK.release()
            logger.info("TTS request #%s lock released", request_id)

    return StreamingResponse(
        relay(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(TTS_SAMPLE_RATE),
            "X-TTS-Request-ID": str(request_id),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
