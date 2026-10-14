from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status

from app.detector import AudioValidationError, detector
from app.settings import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    detector.load()
    yield


app = FastAPI(
    title="Qwen3 Keyword Spotting Service",
    version="0.1.0",
    description="Replaceable sherpa-onnx acoustic keyword detection service.",
    lifespan=lifespan,
)


@app.get("/health")
async def health(response: Response) -> dict[str, Any]:
    ready = detector.ready if settings.enabled else True
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if ready else "unavailable",
        "enabled": settings.enabled,
        "engine": "sherpa-onnx",
        "model": settings.model_name,
        "sample_rate": settings.sample_rate,
    }


@app.post("/v1/detect")
async def detect(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0]
    if content_type != "application/octet-stream":
        raise HTTPException(
            status_code=415,
            detail="expected raw float32 PCM as application/octet-stream",
        )
    if not settings.enabled:
        return {
            "detected": False,
            "keyword": None,
            "engine": "disabled",
            "duration_ms": 0,
        }
    try:
        samples = detector.parse_pcm(await request.body())
        result = detector.detect(samples)
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "detected": result.detected,
        "keyword": result.keyword,
        "engine": "sherpa-onnx",
        "duration_ms": result.duration_ms,
    }
