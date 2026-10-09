"""Per-session state for the voice-chat pipeline.

A Session bundles together:
  - the remote ASR streaming session id (start/chunk/finish), and
  - the running LLM conversation (system role prompt + message history).

Sessions are created/looked-up through the module-level `sessions` manager so
the frontend can keep one `session_id` across ASR and LLM calls.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field

import httpx

import config


@dataclass
class Session:
    id: str
    asr_session_id: str | None = None
    # Conversation history WITHOUT the system prompt (system is prepended per call).
    messages: list[dict] = field(default_factory=list)

    # ---- ASR lifecycle (relayed to the remote ASR service) ----
    async def asr_start(self, client: httpx.AsyncClient) -> str:
        r = await client.post(f"{config.ASR_BASE_URL}/api/start")
        r.raise_for_status()
        self.asr_session_id = r.json()["session_id"]
        return self.asr_session_id

    async def asr_chunk(self, client: httpx.AsyncClient, body: bytes) -> dict:
        # The remote ASR session ends after `finish`; transparently start a new
        # one so the same backend Session can be reused across conversations.
        if not self.asr_session_id:
            await self.asr_start(client)
        r = await client.post(
            f"{config.ASR_BASE_URL}/api/chunk",
            params={"session_id": self.asr_session_id},
            content=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        r.raise_for_status()
        return r.json()

    async def asr_finish(self, client: httpx.AsyncClient) -> dict:
        if not self.asr_session_id:
            raise RuntimeError("ASR session not started")
        r = await client.post(
            f"{config.ASR_BASE_URL}/api/finish",
            params={"session_id": self.asr_session_id},
        )
        r.raise_for_status()
        # The remote session is now closed; drop it so the next recording starts fresh.
        self.asr_session_id = None
        return r.json()

    # ---- LLM turn ----
    async def chat(self, client: httpx.AsyncClient, user_text: str) -> str:
        """Append user_text, call the LLM with the role prompt, return the reply."""
        user_text = (user_text or "").strip()
        if not user_text:
            return ""
        self.messages.append({"role": "user", "content": user_text})

        payload = {
            "model": config.LLM_MODEL_ID,
            "temperature": config.LLM_TEMPERATURE,
            "messages": [
                {"role": "system", "content": config.LLM_ROLE_PROMPT},
                *self.messages,
            ],
        }
        headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
        r = await client.post(
            f"{config.LLM_BASE_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": reply})
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
