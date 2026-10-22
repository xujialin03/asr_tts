"""Scoped runtime skill for direct-LLM voiceprint management.

The model never receives a generic user ID or database access. A session binds
this skill to the speaker that was resolved for the current admitted voice turn,
so an action can only update that speaker's own profile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

import config


SET_CURRENT_PREFERRED_NAME = "voiceprint.set_current_preferred_name"

# Name-like Chinese / Latin / digit strings. Deliberately disallow punctuation so
# a verbose LLM extraction cannot accidentally be stored as the user's name.
_PREFERRED_NAME_RE = re.compile(r"^[\w一-鿿]{1,32}$", re.UNICODE)


@dataclass(frozen=True)
class VoiceprintSkillResult:
    action: str
    preferred_name: str


class VoiceprintManagementSkill:
    """Manage the resolved speaker's profile through the voiceprint service."""

    def __init__(self, speaker_id: str) -> None:
        self._speaker_id = speaker_id

    @staticmethod
    def action_definition() -> dict:
        """JSON action contract supplied to the direct LLM in its system prompt."""
        return {
            "name": SET_CURRENT_PREFERRED_NAME,
            "description": "保存当前说话人的称呼；仅在用户明确说“我叫X”或“叫我X”时使用。",
            "arguments": {
                "type": "object",
                "properties": {
                    "preferred_name": {
                        "type": "string",
                        "description": "用户明确说出的称呼，1 到 32 个中英文、数字或下划线字符。",
                    }
                },
                "required": ["preferred_name"],
                "additionalProperties": False,
            },
        }

    async def execute(self, action: str, arguments: object) -> VoiceprintSkillResult:
        if action != SET_CURRENT_PREFERRED_NAME:
            raise ValueError(f"unsupported voiceprint skill action: {action}")
        if not isinstance(arguments, dict):
            raise ValueError("voiceprint skill arguments must be an object")
        preferred_name = arguments.get("preferred_name")
        if not isinstance(preferred_name, str):
            raise ValueError("preferred_name must be a string")
        preferred_name = "".join(preferred_name.split())
        if not _PREFERRED_NAME_RE.fullmatch(preferred_name):
            raise ValueError("preferred_name must be 1-32 Chinese, Latin, digit, or underscore characters")

        async with httpx.AsyncClient(
            base_url=config.VOICEPRINT_BASE_URL,
            timeout=httpx.Timeout(15.0, connect=5.0),
        ) as client:
            response = await client.patch(
                f"/v1/users/{self._speaker_id}",
                json={"preferred_name": preferred_name},
            )
            response.raise_for_status()

        return VoiceprintSkillResult(
            action=SET_CURRENT_PREFERRED_NAME,
            preferred_name=preferred_name,
        )
