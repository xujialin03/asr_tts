"""Scoped runtime skill for inviting a registered speaker into the session.

The model extracts the invitee's spoken name; this skill resolves it against
the voiceprint directory with exact matching only. It never mutates session
state: it returns the resolved speaker_id and Session applies it to
participant_ids.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import httpx

import config

INVITE_SPEAKER = "session.invite_speaker"

_NAME_RE = re.compile(r"^[\w一-鿿]{1,32}$", re.UNICODE)


class InviteStatus(str, Enum):
    INVITED = "invited"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    INVALID_NAME = "invalid_name"


@dataclass(frozen=True)
class InviteSkillResult:
    action: str
    status: InviteStatus
    speaker_id: str | None = None
    display_name: str | None = None
    match_count: int = 0


class SessionInviteSkill:
    """Resolve a spoken name to a registered voiceprint user."""

    @staticmethod
    def action_definition() -> dict:
        """JSON action contract supplied to the direct LLM in its system prompt."""
        return {
            "name": INVITE_SPEAKER,
            "description": (
                "把一名已注册的用户拉入当前会话；仅在用户明确说"
                "“把X拉进会话/邀请X加入”时使用。X 为用户所说的称呼。"
            ),
            "arguments": {
                "type": "object",
                "properties": {
                    "preferred_name": {
                        "type": "string",
                        "description": "要邀请用户的称呼，1 到 32 个中英文、数字或下划线字符。",
                    }
                },
                "required": ["preferred_name"],
                "additionalProperties": False,
            },
        }

    async def execute(self, action: str, arguments: object) -> InviteSkillResult:
        if action != INVITE_SPEAKER:
            raise ValueError(f"unsupported session skill action: {action}")
        if not isinstance(arguments, dict):
            raise ValueError("session skill arguments must be an object")
        name = arguments.get("preferred_name")
        if not isinstance(name, str):
            return InviteSkillResult(INVITE_SPEAKER, InviteStatus.INVALID_NAME)
        name = "".join(name.split())
        if not _NAME_RE.fullmatch(name):
            return InviteSkillResult(INVITE_SPEAKER, InviteStatus.INVALID_NAME)

        async with httpx.AsyncClient(
            base_url=config.VOICEPRINT_BASE_URL,
            timeout=httpx.Timeout(15.0, connect=5.0),
        ) as client:
            response = await client.get("/v1/users")
            response.raise_for_status()
            users = response.json()

        # Disabled accounts must not be invited: they still appear in /v1/users
        # but are already excluded from the active_embeddings match path, so
        # accepting them here would let a disabled identity join participant_ids
        # and then silently fail every subsequent identify call.
        active = [user for user in users if user.get("is_active")]

        matches = [
            user
            for user in active
            if name
            in {(user.get("preferred_name") or "").strip(), (user.get("display_name") or "").strip()}
        ]
        if not matches:
            return InviteSkillResult(INVITE_SPEAKER, InviteStatus.NOT_FOUND)
        if len(matches) > 1:
            return InviteSkillResult(
                INVITE_SPEAKER, InviteStatus.AMBIGUOUS, match_count=len(matches)
            )
        user = matches[0]
        return InviteSkillResult(
            INVITE_SPEAKER,
            InviteStatus.INVITED,
            speaker_id=str(user["id"]),
            display_name=user.get("display_name"),
        )
