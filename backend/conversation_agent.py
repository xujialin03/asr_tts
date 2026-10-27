"""Lightweight in-process agent for direct voice conversations.

Session owns voice turns, speaker eligibility, and context persistence. This
agent owns prompt construction, OpenAI-compatible model calls, structured reply
parsing, and the narrow set of skills available to the current speaker.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

import config
from voiceprint_skill import SET_CURRENT_PREFERRED_NAME, VoiceprintManagementSkill

logger = logging.getLogger("uvicorn.error")

MAX_TTS_INSTRUCTION_CHARS = 500

SYSTEM_PROMPT = """你是志玲，一名温柔、自然、愿意陪伴用户聊天的中文语音小助手。
你正在参与由若干已加入成员组成的中文语音对话。当前发言人、是否刚被唤醒加入、以及最近对话会随消息提供。只把当前已加入成员的发言当作对话内容；不要推测、回应或提及会话外的背景人声。说话要像熟悉的朋友一样自然、简洁，避免生硬的客服口吻；除非用户明确要求，不要使用 Markdown。

语音识别会把你的名字识别成同音或近音词。{wake_word_aliases} 都只是用户在呼叫你，绝不是用户把你的名字叫错了。无论用户用了其中哪一个词，直接自然回应即可；绝对不要纠正、解释或强调“我叫志玲，不是……”。

若 VOICE_CONTEXT 中 `should_ask_name` 为 true，说明当前成员刚通过唤醒词加入、但声纹身份仍是匿名访客，本次会话尚未问过称呼。自然回应当前内容后，明确而简短地问一次“我该怎么称呼你？”。若为 false，不要主动重复询问姓名。用户没有回答名字时不要阻塞或追问，后续正常聊天。

当且仅当用户明确自报称呼（例如“我叫许老板”“叫我许老板”）时，必须在 actions 中调用下方项目技能保存称呼；不要只在文字里口头记住。用户没有明确报称呼时 actions 必须为空数组。

可用项目技能：{voiceprint_skill}
"""

RESPONSE_CONTRACT = """[DIRECT_LLM_RESPONSE_CONTRACT]
只输出一个合法 JSON 对象，不要 Markdown 围栏或额外文字：
{"text":"要展示并朗读给用户的完整回复","instruction":"简短中文语气指令","actions":[{"name":"项目技能名","arguments":{}}]}

规则：
- text 是完整、自然的口语回复。
- instruction 是简短中文 TTS 语气指令。
- actions 必须是数组；无动作时为 []。
- 仅可使用系统消息列出的项目技能；不可虚构技能或传入未定义参数。
[/DIRECT_LLM_RESPONSE_CONTRACT]
"""


@dataclass(frozen=True)
class AgentTurn:
    session_id: str
    turn_id: str
    speaker_id: str
    speaker: dict[str, Any]
    text: str
    wake_event: bool
    joined_session: bool
    should_ask_name: bool
    response_type: str


@dataclass(frozen=True)
class AgentReply:
    text: str
    instruction: str = ""
    preferred_name: str | None = None


class VoiceConversationAgent:
    """Direct LLM agent scoped to one admitted voice turn at a time."""

    def _system_message(self) -> str:
        aliases = "、".join(f"“{word}”" for word in config.WAKE_WORDS if word)
        skill = json.dumps(
            VoiceprintManagementSkill.action_definition(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return SYSTEM_PROMPT.format(
            wake_word_aliases=aliases or "任何 ASR 对你名字的近音转写",
            voiceprint_skill=skill,
        ) + "\n" + RESPONSE_CONTRACT

    def _messages(self, turn: AgentTurn, context: list[dict[str, Any]]) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self._system_message()}]
        for entry in context:
            role = entry.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = str(entry.get("text") or "")
            speaker_id = entry.get("speaker_id")
            if role == "user" and speaker_id:
                content = f"[用户 {str(speaker_id)[:8]}] {content}"
            messages.append({"role": role, "content": content})
        voice_context = {
            "session_id": turn.session_id,
            "turn_id": turn.turn_id,
            "wake_event": turn.wake_event,
            "joined_session": turn.joined_session,
            "speaker": turn.speaker,
            "should_ask_name": turn.should_ask_name,
            "response_type": turn.response_type,
        }
        messages.append(
            {
                "role": "user",
                "content": (
                    "[VOICE_CONTEXT]"
                    + json.dumps(voice_context, ensure_ascii=False, separators=(",", ":"))
                    + "[/VOICE_CONTEXT]\n"
                    + ("请仅作一句简短确认或追问。\n" if turn.response_type == "brief" else "")
                    + turn.text
                ),
            }
        )
        return messages

    @staticmethod
    def _parse_reply(raw_reply: str) -> tuple[str, str, list[dict[str, Any]]]:
        raw_reply = (raw_reply or "").strip()
        start = raw_reply.find("{")
        if start < 0:
            raise RuntimeError("direct LLM response did not contain a JSON object")
        try:
            value, _ = json.JSONDecoder().raw_decode(raw_reply[start:])
        except json.JSONDecodeError as exc:
            raise RuntimeError("direct LLM returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError("direct LLM response must be a JSON object")
        text = value.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("direct LLM response text must be a non-empty string")
        instruction = value.get("instruction", "")
        if not isinstance(instruction, str):
            instruction = ""
        actions = value.get("actions", [])
        if not isinstance(actions, list):
            raise RuntimeError("direct LLM actions must be an array")
        if len(actions) > 1:
            raise RuntimeError("direct LLM may execute at most one skill action per turn")
        if actions and not isinstance(actions[0], dict):
            raise RuntimeError("direct LLM action must be an object")
        return text.strip(), " ".join(instruction.split())[:MAX_TTS_INSTRUCTION_CHARS], actions

    async def _execute_actions(self, turn: AgentTurn, actions: list[dict[str, Any]]) -> str | None:
        if not actions:
            return None
        action = actions[0]
        name = action.get("name")
        if not isinstance(name, str):
            raise RuntimeError("direct LLM action name must be a string")
        # The model cannot choose a target speaker: the skill binds to turn.speaker_id.
        result = await VoiceprintManagementSkill(turn.speaker_id).execute(
            name,
            action.get("arguments"),
        )
        if result.action == SET_CURRENT_PREFERRED_NAME:
            return result.preferred_name
        raise RuntimeError(f"unhandled voiceprint skill action: {result.action}")

    async def respond(self, turn: AgentTurn, context: list[dict[str, Any]]) -> AgentReply:
        if not config.DIRECT_LLM_MODEL:
            raise RuntimeError("direct_llm.model is required when conversation.provider is direct_llm")
        payload = {
            "model": config.DIRECT_LLM_MODEL,
            "messages": self._messages(turn, context),
            "temperature": config.DIRECT_LLM_TEMPERATURE,
            "max_tokens": config.DIRECT_LLM_MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": config.DIRECT_LLM_ENABLE_THINKING},
        }
        headers = {"Authorization": f"Bearer {config.DIRECT_LLM_API_KEY}"} if config.DIRECT_LLM_API_KEY else {}
        started = time.monotonic()
        async with httpx.AsyncClient(
            base_url=config.DIRECT_LLM_BASE_URL,
            timeout=httpx.Timeout(config.DIRECT_LLM_TIMEOUT_SECONDS, connect=5.0),
            headers=headers,
        ) as client:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
        try:
            raw_reply = response.json()["choices"][0]["message"]["content"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("direct LLM returned an invalid chat response") from exc
        if not isinstance(raw_reply, str):
            raise RuntimeError("direct LLM returned an empty response")
        text, instruction, actions = self._parse_reply(raw_reply)
        preferred_name = await self._execute_actions(turn, actions)
        logger.info(
            "conversation agent session=%s turn=%s elapsed=%.2fs action=%s",
            turn.session_id,
            turn.turn_id,
            time.monotonic() - started,
            actions[0].get("name") if actions else None,
        )
        return AgentReply(text=text, instruction=instruction, preferred_name=preferred_name)
