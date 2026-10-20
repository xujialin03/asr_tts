"""Backend for the voice-chat pipeline.

Relays the remote Qwen3-ASR and Breeze-TTS services and sends assistant turns
to the dedicated Hermes agent. Local sessions keep only ASR and interaction
state; Hermes owns the assistant conversation history.

Frontend flow:
  POST /api/session/start   -> create local interaction session -> {session_id}
  POST /api/chunk           -> stream float32 PCM to ASR        -> {language, text}
  POST /api/finish          -> finish ASR                       -> {language, text}
  POST /api/chat            -> consume an admitted turn in Hermes -> {reply, tts_instruction}
  POST /api/tts             -> synthesize speech (streaming PCM)
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time

import httpx
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import config
from session import ASR_CHUNK_BYTES, SpeakerIdentity, sessions

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
    asyncio.create_task(sessions.warmup_decisions())


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
    return JSONResponse({**s.snapshot(), "asr_session_id": None})


@app.get("/api/session/{session_id}")
async def session_state(session_id: str) -> JSONResponse:
    return JSONResponse(_get_session(session_id).snapshot())


@app.delete("/api/session/{session_id}", status_code=204, response_class=Response)
async def end_session(session_id: str) -> Response:
    if not sessions.delete(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return Response(status_code=204)


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


async def _resolve_speaker(body: bytes) -> dict:
    try:
        response = await client.post(
            f"{config.VOICEPRINT_BASE_URL}/v1/resolve",
            content=body,
            headers={"Content-Type": "application/octet-stream"},
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        response.raise_for_status()
        return response.json()
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


@app.post("/api/voiceprint/resolve")
async def resolve_voiceprint(request: Request, session_id: str) -> JSONResponse:
    """Resolve audio for diagnostics without affecting the conversation."""
    _get_session(session_id)
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="audio body is empty")
    return JSONResponse(await _resolve_speaker(body))


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
    sessions.remove_speaker(speaker_id)
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


# ---- multi-speaker conversation turns ----
class BeginTurnReq(BaseModel):
    source: str = "vad"
    assistant_speaking: bool = False


@app.post("/api/session/{session_id}/turns")
async def begin_turn(session_id: str, req: BeginTurnReq) -> JSONResponse:
    if req.source != "vad":
        raise HTTPException(status_code=422, detail="source must be vad")
    turn = _get_session(session_id).begin_turn(req.source, req.assistant_speaking)
    return JSONResponse(
        {
            "turn_id": turn.id,
            "sequence": turn.sequence,
            "started_while_active": turn.started_while_active,
            "expires_at": turn.expires_at,
        }
    )


@app.post("/api/session/{session_id}/turns/{turn_id}/evaluate")
async def evaluate_turn(
    session_id: str,
    turn_id: str,
    audio: UploadFile,
    text: str = Form(default=""),
    language: str = Form(default=""),
    kws_detected: bool = Form(default=False),
    kws_keyword: str | None = Form(default=None),
    kws_available: bool | None = Form(default=None),
) -> JSONResponse:
    session = _get_session(session_id)
    body = await audio.read()
    if not body:
        raise HTTPException(status_code=400, detail="audio body is empty")

    async with session.turn_lock:
        turn = session.get_turn(turn_id)
        if not turn:
            raise HTTPException(status_code=404, detail="turn not found or expired")
        if turn.evaluated:
            raise HTTPException(status_code=409, detail="turn already evaluated")

        wake_detected = kws_detected or any(
            word and word in text for word in config.WAKE_WORDS
        )
        needs_identity = bool(text) or turn.assistant_speaking
        if not needs_identity:
            turn = session.reject_turn(turn_id, "ASR 未产生有效文本")
        else:
            try:
                identity = SpeakerIdentity.from_result(await _resolve_speaker(body))
            except HTTPException as exc:
                turn = session.reject_turn(
                    turn_id,
                    f"声纹识别失败，本段未发送给助手：{exc.detail}",
                )
            else:
                try:
                    turn = session.evaluate_turn(
                        turn_id,
                        identity,
                        body,
                        text=text,
                        language=language,
                        kws_detected=kws_detected,
                        kws_keyword=kws_keyword,
                    )
                    turn = await session.decide_turn(turn)
                except (KeyError, ValueError) as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc

    result = turn.public()
    result.update(
        {
            "kws_detected": kws_detected,
            "kws_keyword": kws_keyword,
            "kws_available": kws_available,
            "session": session.snapshot(),
        }
    )
    return JSONResponse(result)


class TouchReq(BaseModel):
    session_id: str
    seconds: float | None = None
    force: bool = False


@app.post("/api/touch")
async def touch(req: TouchReq) -> JSONResponse:
    s = _get_session(req.session_id)
    if req.force:
        s.activate(req.seconds)
    return JSONResponse(
        {
            "active": s.is_active(),
            "active_remaining": round(s.active_remaining(), 1),
            "extended": req.force,
        }
    )


class PerformanceEventReq(BaseModel):
    stage: str = Field(min_length=1, max_length=80)
    elapsed_ms: float = Field(ge=0, le=600_000)
    turn_id: str | None = None
    detail: dict = Field(default_factory=dict)


@app.post("/api/session/{session_id}/performance", status_code=204)
async def record_performance(session_id: str, req: PerformanceEventReq) -> Response:
    session = _get_session(session_id)
    session.record_performance(
        req.stage,
        req.elapsed_ms,
        source="frontend",
        turn_id=req.turn_id,
        detail=req.detail,
    )
    return Response(status_code=204)


@app.get("/api/session/{session_id}/performance")
async def session_performance(session_id: str) -> JSONResponse:
    session = _get_session(session_id)
    return JSONResponse({"session_id": session.id, "events": session.performance_snapshot()})


class ChatReq(BaseModel):
    session_id: str
    turn_id: str


@app.post("/api/chat")
async def chat(req: ChatReq) -> JSONResponse:
    s = _get_session(req.session_id)
    try:
        reply = await s.chat(req.turn_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Hermes call failed: {exc}") from exc
    turn = s.get_turn(req.turn_id)
    return JSONResponse(
        {
            "reply": reply.text,
            "tts_instruction": reply.instruction,
            "speaker_id": turn.speaker.speaker_id if turn and turn.speaker else None,
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
