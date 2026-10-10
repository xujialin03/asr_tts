"""Backend for the voice-chat pipeline.

Relays the remote Qwen3-ASR streaming service and the OpenAI-compatible LLM,
keeping per-session state (ASR session id + conversation history) in `session.py`.

Frontend flow:
  POST /api/session/start   -> create session + start ASR  -> {session_id}
  POST /api/chunk           -> stream float32 PCM to ASR    -> {language, text}
  POST /api/finish          -> finish ASR                   -> {language, text}
  POST /api/chat            -> send text to LLM (role prompt)-> {reply}
  POST /api/tts             -> synthesize speech (streaming PCM)
"""

from __future__ import annotations

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

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
        "llm": config.LLM_BASE_URL,
        "tts": config.TTS_BASE_URL,
        "model": config.LLM_MODEL_ID,
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


# ---- LLM chat ----
class ChatReq(BaseModel):
    session_id: str
    text: str


@app.post("/api/chat")
async def chat(req: ChatReq) -> JSONResponse:
    s = _get_session(req.session_id)
    try:
        reply = await s.chat(client, req.text)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")
    return JSONResponse({"reply": reply})


# ---- TTS (streaming PCM) ----
class TtsReq(BaseModel):
    text: str


@app.post("/api/tts")
async def tts(req: TtsReq) -> StreamingResponse:
    """Relay the remote TTS service's streaming PCM (16-bit mono @ sample_rate)."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")

    form = {"text": text}
    if config.TTS_PROMPT:
        form["instruction"] = config.TTS_PROMPT

    # Attach the fixed reference audio + its transcript so the cloned voice
    # stays consistent across every request.
    files = None
    if TTS_REF_BYTES:
        form["ref_text"] = config.TTS_REF_TEXT
        files = {"ref_audio": ("ref_voice.wav", TTS_REF_BYTES, "audio/wav")}

    async def relay():
        async with client.stream(
            "POST",
            f"{config.TTS_BASE_URL}/v1/audio/speech",
            data=form,
            files=files,
            timeout=httpx.Timeout(120.0, connect=10.0),
        ) as upstream:
            if upstream.status_code != 200:
                detail = (await upstream.aread()).decode(errors="replace")
                raise HTTPException(
                    status_code=502, detail=f"TTS failed: {upstream.status_code} {detail}"
                )
            async for chunk in upstream.aiter_raw():
                yield chunk

    return StreamingResponse(
        relay(),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(TTS_SAMPLE_RATE)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
