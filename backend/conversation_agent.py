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
from teleop_skill import TeleopMCPSkill
from voiceprint_skill import SET_CURRENT_PREFERRED_NAME, VoiceprintManagementSkill

logger = logging.getLogger("uvicorn.error")

MAX_TTS_INSTRUCTION_CHARS = 500

SYSTEM_PROMPT = """你是志玲，一名温柔、自然、愿意陪伴用户聊天的中文语音小助手。
你正在参与由若干已加入成员组成的中文语音对话。当前发言人、是否刚被唤醒加入、以及最近对话会随消息提供。只把当前已加入成员的发言当作对话内容；不要推测、回应或提及会话外的背景人声。说话要像熟悉的朋友一样自然、简洁，避免生硬的客服口吻；除非用户明确要求，不要使用 Markdown。

语音识别会把你的名字识别成同音或近音词。{wake_word_aliases} 都只是用户在呼叫你，绝不是用户把你的名字叫错了。无论用户用了其中哪一个词，直接自然回应即可；绝对不要纠正、解释或强调“我叫志玲，不是……”。

若 VOICE_CONTEXT 中 `should_ask_name` 为 true，说明当前成员刚通过唤醒词加入、但声纹身份仍是匿名访客，本次会话尚未问过称呼。自然回应当前内容后，明确而简短地问一次“我该怎么称呼你？”。若为 false，不要主动重复询问姓名。用户没有回答名字时不要阻塞或追问，后续正常聊天。

当且仅当用户明确自报称呼（例如“我叫许老板”“叫我许老板”）时，必须在 actions 中调用下方项目技能保存称呼；不要只在文字里口头记住。用户没有明确报称呼时 actions 必须为空数组。

可用项目技能：{skills}
{teleop_rules}
"""

TELEOP_RULES = """遥操/推理/视觉技能（teleop.*）操作一台真实的 SO101 机械臂及双摄，务必谨慎：
- 仅当用户明确要求启动/停止/查询某项操作时才调用；闲聊不要调用。
- start_teleoperation 与 start_inference 都会使实体机械臂通电运动；调用前必须确认用户已确认现场有人看护、机械臂周围无人无遮挡；否则先用一句话向用户确认，本轮不要调用。
- 用户说“停止/停下/快停”或提到碰撞、抖动、失控、画面冻结时：遥操中调用 stop_teleoperation，推理中调用 stop_inference。
- start_inference 只需把用户的自然语言任务作为 instruction 传入；不要替用户编造模型或 Corrector，服务端会自动选择。当前仅部署了“紫鸭入盒”一类已注册权重，匹配不到会报错，据实转述即可。
- ask_camera_vision 不移动机械臂，但会把当前双摄截图发到外部视觉模型；用户问“你看到了什么/画面里有什么”一类问题时调用。
- 查询与卸载类（get_*_status / get_joint_curves / unload_inference_model）安全，可直接调用。
- 视频流和看板 URL 很长且带鉴权，不要口头念出；只告诉用户“可以在综合看板查看”。
- 本轮最多执行一个技能动作；需要更多操作时先说明，下一轮再执行。"""

TOOL_RESULT_CONTRACT = """[TOOL_RESULT]
{name}
{payload}
[/TOOL_RESULT]
上一步技能已经真实执行。请用一两句口语、自然地转述上面的关键结果给用户（例如当前是否在运行、参数、数值概况），只输出 JSON 对象 {{"text":"...","instruction":"..."}}，不要 actions，不要念 URL。"""


def _redact_tokens(value: object) -> object:
    """Strip auth tokens from any URL-looking strings before narration."""
    if isinstance(value, str):
        import re

        cleaned = re.sub(r"[?&]token=[^&#\s\"']*", "", value)
        # Collapse separators left by the removed token param: ?& -> ?, && -> &,
        # a leading & (when ? was removed) -> ?, and trailing ?/&.
        cleaned = re.sub(r"\?&", "?", cleaned)
        cleaned = re.sub(r"&&", "&", cleaned)
        cleaned = re.sub(r"(?<!\?)&(?=[^?]*$)", "?", cleaned)
        cleaned = re.sub(r"[?&]$", "", cleaned)
        return cleaned
    if isinstance(value, dict):
        return {key: _redact_tokens(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_tokens(item) for item in value]
    return value


def _narration_payload(action: str, structured: object) -> str:
    """Compact the tool result for the narration pass (cap size, drop noise)."""
    if not isinstance(structured, dict):
        return json.dumps(_redact_tokens(structured), ensure_ascii=False)[:2000]
    payload: dict[str, object] = {}
    if action == "teleop.get_joint_curves":
        payload["seconds"] = structured.get("seconds")
        for side in ("leader", "follower"):
            series = structured.get(side)
            if isinstance(series, dict):
                payload[side] = {
                    axis: (values[-1] if isinstance(values, list) and values else None)
                    for axis, values in series.items()
                }
    if action == "teleop.get_inference_status":
        status = structured.get("status") or {}
        payload["status"] = {
            key: status.get(key)
            for key in ("phase", "running", "resident", "model_warm", "total_dispatches", "error")
        }
        return json.dumps(_redact_tokens(payload), ensure_ascii=False)[:2000]
    if action == "teleop.ask_camera_vision":
        # Pass the answer through; cap size and strip any incidental token URLs.
        payload["vision_status"] = structured.get("vision_status")
        payload["answer"] = structured.get("answer")
        payload["capture_source"] = structured.get("capture_source")
        return json.dumps(_redact_tokens(payload), ensure_ascii=False)[:2000]
    status = structured.get("status")
    if isinstance(status, dict):
        payload["status"] = {
            key: status.get(key)
            for key in ("mode", "phase", "running", "pid", "episode", "error", "teleop_config")
        }
    if action == "teleop.get_teleoperation_status":
        payload["has_dashboard"] = bool(structured.get("dashboard_url"))
    return json.dumps(_redact_tokens(payload), ensure_ascii=False)[:2000]

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
        skills = [VoiceprintManagementSkill.action_definition()]
        teleop_rules = ""
        if config.TELEOP_ENABLED:
            skills.extend(TeleopMCPSkill.definitions())
            teleop_rules = "\n" + TELEOP_RULES
        return SYSTEM_PROMPT.format(
            wake_word_aliases=aliases or "任何 ASR 对你名字的近音转写",
            skills=json.dumps(skills, ensure_ascii=False, separators=(",", ":"), indent=1),
            teleop_rules=teleop_rules,
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

    async def _execute_actions(
        self, turn: AgentTurn, actions: list[dict[str, Any]]
    ) -> tuple[str | None, str, dict[str, Any] | None]:
        """Execute the single action. Returns (preferred_name, narration_text, narration_meta).

        Voiceprint is a silent side effect (the model's own text covers it).
        Teleop actions return data the user must hear, so narration_text is the
        skill's fallback summary; respond() upgrades it via a second model pass.
        """
        if not actions:
            return None, "", None
        action = actions[0]
        name = action.get("name")
        if not isinstance(name, str):
            raise RuntimeError("direct LLM action name must be a string")
        if name == SET_CURRENT_PREFERRED_NAME:
            # The model cannot choose a target speaker: the skill binds to turn.speaker_id.
            result = await VoiceprintManagementSkill(turn.speaker_id).execute(
                name,
                action.get("arguments"),
            )
            if result.action == SET_CURRENT_PREFERRED_NAME:
                return result.preferred_name, "", None
            raise RuntimeError(f"unhandled voiceprint skill action: {result.action}")
        if config.TELEOP_ENABLED and TeleopMCPSkill.is_supported(name):
            try:
                result = await TeleopMCPSkill().execute(name, action.get("arguments"))
            except Exception as exc:  # network/contract failure: tell the user, don't crash the turn
                logger.warning("teleop skill %s failed: %s", name, exc)
                return None, "抱歉，我这边连不上遥操服务，稍后再试。", None
            meta = {
                "action": result.action,
                "ok": result.ok,
                "payload": _narration_payload(result.action, result.structured),
            }
            return None, result.fallback_summary, meta
        raise RuntimeError(f"unknown skill action: {name}")

    async def _chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": config.DIRECT_LLM_MODEL,
            "messages": messages,
            "temperature": config.DIRECT_LLM_TEMPERATURE,
            "max_tokens": config.DIRECT_LLM_MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": config.DIRECT_LLM_ENABLE_THINKING},
        }
        headers = {"Authorization": f"Bearer {config.DIRECT_LLM_API_KEY}"} if config.DIRECT_LLM_API_KEY else {}
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
        return raw_reply

    async def _narrate_tool_result(
        self, turn: AgentTurn, messages: list[dict[str, Any]], meta: dict[str, Any]
    ) -> tuple[str, str] | None:
        """Second model pass: turn the executed tool's data into a spoken update."""
        narration_messages = messages + [
            {
                "role": "user",
                "content": TOOL_RESULT_CONTRACT.format(name=meta["action"], payload=meta["payload"]),
            }
        ]
        try:
            raw = await self._chat(narration_messages)
            text, instruction, actions = self._parse_reply(raw)
        except Exception as exc:
            logger.warning("teleop narration pass failed: %s", exc)
            return None
        if actions:
            logger.warning("teleop narration pass returned actions; ignoring")
        return text, instruction

    async def respond(self, turn: AgentTurn, context: list[dict[str, Any]]) -> AgentReply:
        if not config.DIRECT_LLM_MODEL:
            raise RuntimeError("direct_llm.model is required")
        started = time.monotonic()
        messages = self._messages(turn, context)
        text, instruction, actions = self._parse_reply(await self._chat(messages))
        preferred_name, tool_text, tool_meta = await self._execute_actions(turn, actions)
        if tool_text:
            # Teleop result: prefer a model-narrated update over the raw summary.
            if tool_meta and tool_meta["ok"]:
                narrated = await self._narrate_tool_result(turn, messages, tool_meta)
                if narrated:
                    text, instruction = narrated
                else:
                    text = tool_text
            else:
                text = tool_text
        logger.info(
            "conversation agent session=%s turn=%s elapsed=%.2fs action=%s",
            turn.session_id,
            turn.turn_id,
            time.monotonic() - started,
            actions[0].get("name") if actions else None,
        )
        return AgentReply(text=text, instruction=instruction, preferred_name=preferred_name)
