"""Thin async proxy in front of the remote Qwen3-ASR streaming service.

The remote service (default http://36.212.51.208:9020) exposes a session-based
streaming protocol:

    POST /api/start                       -> {"session_id": "..."}
    POST /api/chunk?session_id=<id>       -> {"language": "...", "text": "..."}
         body: raw little-endian float32 PCM, 16 kHz, mono
    POST /api/finish?session_id=<id>      -> {"language": "...", "text": "..."}

The remote service sends no CORS headers, so a browser cannot call it directly.
This backend relays those calls and adds CORS so the Vue frontend can reach it.
"""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

ASR_BASE_URL = os.getenv("ASR_BASE_URL", "http://36.212.51.208:9020").rstrip("/")
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

# Long read timeout: a chunk call blocks until the remote returns a transcript.
client = httpx.AsyncClient(
    base_url=ASR_BASE_URL,
    timeout=httpx.Timeout(60.0, connect=10.0),
)

app = FastAPI(title="Qwen3-ASR Proxy", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await client.aclose()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "asr_base_url": ASR_BASE_URL}


@app.post("/api/start")
async def start() -> JSONResponse:
    r = await client.post("/api/start")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/api/chunk")
async def chunk(request: Request, session_id: str) -> JSONResponse:
    body = await request.body()
    r = await client.post(
        "/api/chunk",
        params={"session_id": session_id},
        content=body,
        headers={"Content-Type": "application/octet-stream"},
    )
    return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/api/finish")
async def finish(request: Request, session_id: str) -> JSONResponse:
    r = await client.post("/api/finish", params={"session_id": session_id})
    return JSONResponse(r.json(), status_code=r.status_code)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
