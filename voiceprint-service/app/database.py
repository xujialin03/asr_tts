from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import asyncpg

from app.settings import settings

_pool: asyncpg.Pool | None = None


def _decode_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _user_dict(record: asyncpg.Record) -> dict[str, Any]:
    data = dict(record)
    data["metadata"] = _decode_json(data.get("metadata")) or {}
    return data


async def connect() -> None:
    global _pool
    _pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=5)
    migration = Path(__file__).resolve().parent.parent / "migrations" / "001_initial.sql"
    async with _pool.acquire() as conn:
        await conn.execute(migration.read_text(encoding="utf-8"))


async def disconnect() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("database is not connected")
    return _pool


async def ping() -> bool:
    try:
        return await pool().fetchval("SELECT TRUE") is True
    except (asyncpg.PostgresError, RuntimeError):
        return False


async def create_user(data: dict[str, Any]) -> dict[str, Any]:
    row = await pool().fetchrow(
        """
        INSERT INTO users (
            username, display_name, preferred_name, email, phone,
            locale, timezone, notes, metadata, is_active
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10)
        RETURNING *
        """,
        data["username"],
        data["display_name"],
        data.get("preferred_name"),
        data.get("email"),
        data.get("phone"),
        data.get("locale", "zh-CN"),
        data.get("timezone", "Asia/Shanghai"),
        data.get("notes"),
        json.dumps(data.get("metadata", {}), ensure_ascii=False),
        data.get("is_active", True),
    )
    return _user_dict(row)


async def list_users() -> list[dict[str, Any]]:
    rows = await pool().fetch(
        """
        SELECT u.*, COUNT(v.id) FILTER (WHERE v.is_active) AS voiceprint_count
        FROM users u
        LEFT JOIN voiceprints v ON v.user_id = u.id
        GROUP BY u.id
        ORDER BY u.created_at
        """
    )
    return [_user_dict(row) for row in rows]


async def get_user(user_id: UUID) -> dict[str, Any] | None:
    row = await pool().fetchrow(
        """
        SELECT u.*, COUNT(v.id) FILTER (WHERE v.is_active) AS voiceprint_count
        FROM users u
        LEFT JOIN voiceprints v ON v.user_id = u.id
        WHERE u.id = $1
        GROUP BY u.id
        """,
        user_id,
    )
    return _user_dict(row) if row else None


async def update_user(user_id: UUID, changes: dict[str, Any]) -> dict[str, Any] | None:
    if not changes:
        return await get_user(user_id)
    allowed = {
        "username", "display_name", "preferred_name", "email", "phone",
        "locale", "timezone", "notes", "metadata", "is_active",
    }
    unknown = changes.keys() - allowed
    if unknown:
        raise ValueError(f"unsupported fields: {', '.join(sorted(unknown))}")

    values: list[Any] = []
    assignments: list[str] = []
    for index, (name, value) in enumerate(changes.items(), start=1):
        values.append(json.dumps(value, ensure_ascii=False) if name == "metadata" else value)
        cast = "::jsonb" if name == "metadata" else ""
        assignments.append(f"{name} = ${index}{cast}")
    values.append(user_id)
    row = await pool().fetchrow(
        f"UPDATE users SET {', '.join(assignments)} WHERE id = ${len(values)} RETURNING *",
        *values,
    )
    return _user_dict(row) if row else None


async def delete_user(user_id: UUID) -> bool:
    result = await pool().execute("DELETE FROM users WHERE id = $1", user_id)
    return result == "DELETE 1"


async def add_voiceprint(
    user_id: UUID,
    label: str | None,
    embedding: list[float],
    duration_ms: int,
    audio_sha256: str,
    quality: dict[str, Any],
) -> dict[str, Any]:
    row = await pool().fetchrow(
        """
        INSERT INTO voiceprints (
            user_id, label, model_id, model_revision, embedding_dimension,
            embedding, duration_ms, sample_rate, audio_sha256, quality
        ) VALUES ($1, $2, $3, $4, $5, $6::real[], $7, $8, $9, $10::jsonb)
        RETURNING id, user_id, label, model_id, model_revision,
                  embedding_dimension, duration_ms, sample_rate, audio_sha256,
                  quality, is_active, created_at
        """,
        user_id,
        label,
        settings.model_id,
        settings.model_revision,
        len(embedding),
        embedding,
        duration_ms,
        settings.sample_rate,
        audio_sha256,
        json.dumps(quality, ensure_ascii=False),
    )
    data = dict(row)
    data["quality"] = _decode_json(data.get("quality")) or {}
    return data


async def list_voiceprints(user_id: UUID) -> list[dict[str, Any]]:
    rows = await pool().fetch(
        """
        SELECT id, user_id, label, model_id, model_revision,
               embedding_dimension, duration_ms, sample_rate, audio_sha256,
               quality, is_active, created_at
        FROM voiceprints
        WHERE user_id = $1
        ORDER BY created_at
        """,
        user_id,
    )
    return [
        {**dict(row), "quality": _decode_json(row["quality"]) or {}}
        for row in rows
    ]


async def delete_voiceprint(user_id: UUID, voiceprint_id: UUID) -> bool:
    result = await pool().execute(
        "DELETE FROM voiceprints WHERE id = $1 AND user_id = $2",
        voiceprint_id,
        user_id,
    )
    return result == "DELETE 1"


async def active_embeddings(user_id: UUID | None = None) -> list[dict[str, Any]]:
    conditions = ["v.is_active", "u.is_active", "v.model_id = $1", "v.model_revision = $2"]
    args: list[Any] = [settings.model_id, settings.model_revision]
    if user_id is not None:
        args.append(user_id)
        conditions.append(f"u.id = ${len(args)}")
    rows = await pool().fetch(
        f"""
        SELECT v.id AS voiceprint_id, v.user_id, v.embedding,
               u.username, u.display_name, u.preferred_name
        FROM voiceprints v
        JOIN users u ON u.id = v.user_id
        WHERE {' AND '.join(conditions)}
        """,
        *args,
    )
    return [dict(row) for row in rows]
