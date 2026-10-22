"""Interaction decisions for completed voice turns.

The decision model (``DecisionModelClient``) is an OpenAI-compatible generative
model that judges "is this addressed to the assistant" (incl. homophone names)
and whether to reply now. ~1s with thinking disabled. Configured via the
``decision_model`` section in ``config.yaml``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

import config

logger = logging.getLogger("uvicorn.error")

RESPONSE_TYPES = frozenset({"normal", "brief", "silent"})
INTERACTION_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_addressed_to_ai": {
            "type": "boolean",
            "description": "Whether the current message is directed to the assistant.",
        },
        "should_respond": {
            "type": "boolean",
            "description": "Whether the assistant should produce a user-visible reply now.",
        },
        "response_type": {
            "type": "string",
            "enum": ["normal", "brief", "silent"],
            "description": "normal, brief, or silent.",
        },
        "is_complete": {
            "type": "boolean",
            "description": (
                "Whether the current message is a complete utterance. False when it "
                "looks like a fragment cut off by a pause (e.g. '今天' or a trailing "
                "wake word) that should wait for the speaker to finish."
            ),
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Decision confidence.",
        },
        "reason": {
            "type": "string",
            "enum": [
                "明确唤醒或直接提问",
                "延续最近的助理对话",
                "需要简短确认",
                "人类之间的对话",
                "无实质信息或重复反馈",
            ],
            "description": "Short structured debugging reason in Chinese.",
        },
    },
    "required": [
        "is_addressed_to_ai",
        "should_respond",
        "response_type",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class InteractionDecision:
    is_addressed_to_ai: bool
    should_respond: bool
    response_type: str
    confidence: float
    reason: str
    is_complete: bool = True
    fallback: bool = False

    @classmethod
    def from_value(cls, value: Any) -> "InteractionDecision":
        if not isinstance(value, dict):
            raise ValueError("decision model output must be an object")
        required = set(INTERACTION_DECISION_SCHEMA["required"])
        allowed = set(INTERACTION_DECISION_SCHEMA["properties"])
        if not required.issubset(value):
            raise ValueError("decision model output is missing required fields")
        if not set(value).issubset(allowed):
            raise ValueError("decision model output has unknown fields")
        if not isinstance(value["is_addressed_to_ai"], bool):
            raise ValueError("is_addressed_to_ai must be a boolean")
        if not isinstance(value["should_respond"], bool):
            raise ValueError("should_respond must be a boolean")
        response_type = value["response_type"]
        if response_type not in RESPONSE_TYPES:
            raise ValueError("invalid response_type")
        confidence = value["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (float, int)):
            raise ValueError("confidence must be a number")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        reason = value["reason"]
        if not isinstance(reason, str):
            raise ValueError("reason must be a string")
        # Optional: default to complete so a model that omits it degrades to the
        # pre-buffer behavior (reply immediately) instead of stalling.
        is_complete = value.get("is_complete", True)
        if not isinstance(is_complete, bool):
            raise ValueError("is_complete must be a boolean")
        should_respond = value["should_respond"]
        if not should_respond:
            response_type = "silent"
        elif response_type == "silent":
            should_respond = False
        return cls(
            is_addressed_to_ai=value["is_addressed_to_ai"],
            should_respond=should_respond,
            response_type=response_type,
            confidence=float(confidence),
            reason=reason.strip()[:200],
            is_complete=is_complete,
        )


class DecisionClient(Protocol):
    async def decide(self, state: dict[str, Any]) -> InteractionDecision: ...


# Mirrors INTERACTION_DECISION_SCHEMA semantics for the decision model. The
# persona line is what lets the model tolerate homophone misrecognitions of the
# assistant's name (诗林/志宁/志凌 → 志玲) instead of relying on wake-word strings.
DECISION_MODEL_SYSTEM = """你是语音助手交互决策器。当前助手人设：__PERSONA__
根据 state 判断，只输出一个 JSON 对象，不要多余文字：
{"is_addressed_to_ai":bool,"should_respond":bool,"response_type":"normal|brief|silent","is_complete":bool,"confidence":0~1,"reason":"简短中文理由"}
规则：
- 人类之间互相聊天、自言自语、与助手无关→is_addressed_to_ai=false,should_respond=false,response_type=silent。
- 对助手说话、呼叫助手（含同音误识别的名字）、或延续助手上一轮话题→should_respond=true。
- is_addressed_to_ai=false 时 should_respond 必须为 false。
- is_complete 表示这句话是否已经说完：明显是停顿切开的残句（如只有"今天"、以唤醒词结尾、句子明显没说完）→is_complete=false；完整问句或陈述→is_complete=true。"""


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in model output: {text[:120]!r}")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("model output JSON is not an object")
    return value


class DecisionModelClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=config.DECISION_MODEL_BASE_URL,
            timeout=httpx.Timeout(config.DECISION_MODEL_TIMEOUT_SECONDS, connect=5.0),
            headers={"Authorization": f"Bearer {config.DECISION_MODEL_API_KEY}"}
            if config.DECISION_MODEL_API_KEY
            else {},
        )
        self._system = DECISION_MODEL_SYSTEM.replace(
            "__PERSONA__", config.AGENT_PERSONA
        )

    async def warmup(self) -> None:
        # Warm the connection pool / first-token latency without a real turn.
        try:
            await self.decide({"current_message": "你好", "recent_messages": []})
        except Exception as exc:
            logger.warning("decision model warmup failed: %s", exc)

    def _payload(self, state: dict[str, Any]) -> dict:
        return {
            "model": config.DECISION_MODEL_NAME,
            "messages": [
                {"role": "system", "content": self._system},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": config.DECISION_MODEL_MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": config.DECISION_MODEL_ENABLE_THINKING},
        }

    async def decide(self, state: dict[str, Any]) -> InteractionDecision:
        started = time.monotonic()
        last: Exception | None = None
        try:
            for attempt in range(config.DECISION_MODEL_MAX_RETRIES):
                try:
                    r = await self._client.post("/chat/completions", json=self._payload(state))
                    r.raise_for_status()
                    content = r.json()["choices"][0]["message"]["content"]
                    return InteractionDecision.from_value(_extract_json(content))
                except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    last = exc
                    if attempt < config.DECISION_MODEL_MAX_RETRIES - 1:
                        await asyncio.sleep(0.6 * (attempt + 1))
            raise last or RuntimeError("decision model failed")
        finally:
            logger.info("decision model elapsed=%.2fs", time.monotonic() - started)


def build_decision_client() -> DecisionClient:
    return DecisionModelClient()
