from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

import asyncpg
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from app import database
from app.model import AudioValidationError, encoder
from app.settings import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    await database.connect()
    encoder.load()
    yield
    await database.disconnect()


app = FastAPI(
    title="Qwen3 Voiceprint Service",
    version="0.1.0",
    description="Local CAM++ speaker enrollment, verification, identification, and user management.",
    lifespan=lifespan,
)

# Serialize automatic resolution so two simultaneous first utterances cannot both
# observe an empty identity set and create duplicate anonymous speakers.
RESOLVE_LOCK = asyncio.Lock()


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    display_name: str = Field(min_length=1, max_length=128)
    preferred_name: str | None = Field(default=None, max_length=128)
    email: str | None = Field(default=None, max_length=320)
    phone: str | None = Field(default=None, max_length=32)
    locale: str = Field(default="zh-CN", max_length=16)
    timezone: str = Field(default="Asia/Shanghai", max_length=64)
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class UserUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str | None = Field(default=None, min_length=1, max_length=64)
    display_name: str | None = Field(default=None, min_length=1, max_length=128)
    preferred_name: str | None = Field(default=None, max_length=128)
    email: str | None = Field(default=None, max_length=320)
    phone: str | None = Field(default=None, max_length=32)
    locale: str | None = Field(default=None, max_length=16)
    timezone: str | None = Field(default=None, max_length=64)
    notes: str | None = None
    metadata: dict[str, Any] | None = None
    is_active: bool | None = None


async def _audio_embedding(request: Request) -> tuple[list[float], dict[str, Any]]:
    if request.headers.get("content-type", "").split(";", 1)[0] != "application/octet-stream":
        raise HTTPException(status_code=415, detail="expected raw float32 PCM as application/octet-stream")
    try:
        audio = encoder.parse_pcm(await request.body())
        embedding = encoder.embed(audio)
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    quality = {"rms": round(audio.rms, 6), "peak": round(audio.peak, 6)}
    return embedding, {
        "duration_ms": audio.duration_ms,
        "audio_sha256": audio.sha256,
        "quality": quality,
    }


def _score(query: list[float], candidate: list[float]) -> float:
    # Stored and query embeddings are normalized. Normalize defensively for old data.
    left = np.asarray(query, dtype=np.float32)
    right = np.asarray(candidate, dtype=np.float32)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else -1.0


def _decision(score: float, accept_threshold: float, uncertain_threshold: float) -> str:
    if score >= accept_threshold:
        return "matched"
    if score >= uncertain_threshold:
        return "uncertain"
    return "unknown"


def _best_match(
    query: list[float], candidates: list[dict[str, Any]]
) -> tuple[float, dict[str, Any]] | None:
    best_by_user: dict[UUID, tuple[float, dict[str, Any]]] = {}
    for candidate in candidates:
        score = _score(query, candidate["embedding"])
        current = best_by_user.get(candidate["user_id"])
        if current is None or score > current[0]:
            best_by_user[candidate["user_id"]] = (score, candidate)
    return max(best_by_user.values(), key=lambda item: item[0]) if best_by_user else None


def _public_speaker(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate["user_id"],
        "username": candidate["username"],
        "display_name": candidate["display_name"],
        "preferred_name": candidate["preferred_name"],
    }


def _validate_thresholds(accept_threshold: float, uncertain_threshold: float) -> None:
    if not -1 <= uncertain_threshold <= accept_threshold <= 1:
        raise HTTPException(
            status_code=422,
            detail="thresholds must satisfy -1 <= uncertain_threshold <= accept_threshold <= 1",
        )


@app.exception_handler(asyncpg.UniqueViolationError)
async def unique_violation_handler(_: Request, exc: asyncpg.UniqueViolationError):
    return Response(
        content='{"detail":"username, email, or phone already exists"}',
        status_code=409,
        media_type="application/json",
    )


@app.get("/health")
async def health(response: Response) -> dict[str, Any]:
    db_ready = await database.ping()
    ready = db_ready and encoder.ready
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if ready else "unavailable",
        "database": "ok" if db_ready else "unavailable",
        "model": "ok" if encoder.ready else "unavailable",
        "model_id": settings.model_id,
        "model_revision": settings.model_revision,
        "embedding_dimension": settings.embedding_dimension,
    }


@app.post("/v1/users", status_code=status.HTTP_201_CREATED)
async def create_user(req: UserCreate) -> dict[str, Any]:
    return await database.create_user(req.model_dump())


@app.get("/v1/users")
async def list_users() -> list[dict[str, Any]]:
    return await database.list_users()


@app.get("/v1/users/{user_id}")
async def get_user(user_id: UUID) -> dict[str, Any]:
    user = await database.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return user


@app.patch("/v1/users/{user_id}")
async def update_user(user_id: UUID, req: UserUpdate) -> dict[str, Any]:
    changes = req.model_dump(exclude_unset=True)
    user = await database.update_user(user_id, changes)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return user


@app.delete("/v1/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: UUID) -> Response:
    if not await database.delete_user(user_id):
        raise HTTPException(status_code=404, detail="user not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/v1/users/{user_id}/voiceprints", status_code=status.HTTP_201_CREATED)
async def enroll_voiceprint(
    user_id: UUID,
    request: Request,
    label: str | None = Query(default=None, max_length=128),
) -> dict[str, Any]:
    if await database.get_user(user_id) is None:
        raise HTTPException(status_code=404, detail="user not found")
    embedding, audio = await _audio_embedding(request)
    return await database.add_voiceprint(user_id, label, embedding, **audio)


@app.get("/v1/users/{user_id}/voiceprints")
async def list_voiceprints(user_id: UUID) -> list[dict[str, Any]]:
    if await database.get_user(user_id) is None:
        raise HTTPException(status_code=404, detail="user not found")
    return await database.list_voiceprints(user_id)


@app.delete("/v1/users/{user_id}/voiceprints", status_code=200)
async def delete_all_voiceprints(user_id: UUID) -> dict[str, Any]:
    if await database.get_user(user_id) is None:
        raise HTTPException(status_code=404, detail="user not found")
    deleted = await database.delete_all_voiceprints(user_id)
    return {"user_id": user_id, "deleted": deleted}


@app.delete("/v1/users/{user_id}/voiceprints/{voiceprint_id}", status_code=204)
async def delete_voiceprint(user_id: UUID, voiceprint_id: UUID) -> Response:
    if not await database.delete_voiceprint(user_id, voiceprint_id):
        raise HTTPException(status_code=404, detail="voiceprint not found")
    return Response(status_code=204)


@app.post("/v1/resolve")
async def resolve_speaker(
    request: Request,
    accept_threshold: float = Query(default=settings.identify_threshold),
) -> dict[str, Any]:
    """Resolve an utterance to a stable speaker ID, creating one when unknown."""
    if not -1 <= accept_threshold <= 1:
        raise HTTPException(status_code=422, detail="accept_threshold must be between -1 and 1")
    query, audio = await _audio_embedding(request)

    async with RESOLVE_LOCK:
        match = _best_match(query, await database.active_embeddings())
        if match is not None and match[0] >= accept_threshold:
            score, candidate = match
            return {
                "speaker_id": candidate["user_id"],
                "created": False,
                "score": round(score, 6),
                "speaker": _public_speaker(candidate),
                **audio["quality"],
                "duration_ms": audio["duration_ms"],
            }

        speaker = await database.create_anonymous_speaker_with_voiceprint(
            query,
            **audio,
        )
        return {
            "speaker_id": speaker["id"],
            "created": True,
            "score": round(match[0], 6) if match is not None else None,
            "speaker": {
                "id": speaker["id"],
                "username": speaker["username"],
                "display_name": speaker["display_name"],
                "preferred_name": speaker["preferred_name"],
            },
            "voiceprint_id": speaker["voiceprint_id"],
            **audio["quality"],
            "duration_ms": audio["duration_ms"],
        }


@app.post("/v1/identify")
async def identify(
    request: Request,
    accept_threshold: float = Query(default=settings.identify_threshold),
    uncertain_threshold: float = Query(default=settings.uncertain_threshold),
) -> dict[str, Any]:
    _validate_thresholds(accept_threshold, uncertain_threshold)
    query, audio = await _audio_embedding(request)
    candidates = await database.active_embeddings()
    if not candidates:
        return {"decision": "unknown", "score": None, "user": None, **audio["quality"], "duration_ms": audio["duration_ms"]}

    match = _best_match(query, candidates)
    score, candidate = match
    decision = _decision(score, accept_threshold, uncertain_threshold)
    user = _public_speaker(candidate) if decision != "unknown" else None
    return {
        "decision": decision,
        "score": round(score, 6),
        "user": user,
        **audio["quality"],
        "duration_ms": audio["duration_ms"],
    }


@app.post("/v1/users/{user_id}/verify")
async def verify(
    user_id: UUID,
    request: Request,
    accept_threshold: float = Query(default=settings.identify_threshold),
    uncertain_threshold: float = Query(default=settings.uncertain_threshold),
) -> dict[str, Any]:
    _validate_thresholds(accept_threshold, uncertain_threshold)
    if await database.get_user(user_id) is None:
        raise HTTPException(status_code=404, detail="user not found")
    query, audio = await _audio_embedding(request)
    candidates = await database.active_embeddings(user_id)
    if not candidates:
        raise HTTPException(status_code=409, detail="user has no active voiceprints")
    score = max(_score(query, item["embedding"]) for item in candidates)
    return {
        "decision": _decision(score, accept_threshold, uncertain_threshold),
        "score": round(score, 6),
        "user_id": user_id,
        **audio["quality"],
        "duration_ms": audio["duration_ms"],
    }
