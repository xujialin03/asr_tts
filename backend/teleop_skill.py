"""LeRobot SO101 teleop/inference/vision MCP skill for the direct-LLM voice agent.

The LeRobot server speaks MCP over Streamable-HTTP: single JSON-RPC POSTs to one
endpoint, sessionless (verified against the live deployment). We keep a thin
client here instead of pulling in the MCP SDK. All nine remote tools are exposed
as project skills on one `teleop.` prefix (they share one server, one token, one
enable flag):

Teleop (moves a physical arm):
- start_teleoperation  (DANGEROUS: powers and moves a physical robot arm)
- stop_teleoperation
- get_teleoperation_status
- get_joint_curves
Inference (VLA task, also moves a physical arm):
- start_inference  (DANGEROUS: powers and moves a physical robot arm)
- stop_inference
- get_inference_status
- unload_inference_model
Vision (no arm motion, but sends camera frames to an external model):
- ask_camera_vision

Schemas below mirror the live tools/list response (descriptions translated for
the system prompt). Like the voiceprint skill, the model only emits a
{name, arguments} action; this class owns transport and argument validation. The
server result is narrated back by the agent's second model pass (see
conversation_agent), so the caller decides what the user hears — skill summaries
here are only a fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

import config

START_TELEOPERATION = "teleop.start_teleoperation"
STOP_TELEOPERATION = "teleop.stop_teleoperation"
GET_TELEOPERATION_STATUS = "teleop.get_teleoperation_status"
GET_JOINT_CURVES = "teleop.get_joint_curves"
START_INFERENCE = "teleop.start_inference"
STOP_INFERENCE = "teleop.stop_inference"
GET_INFERENCE_STATUS = "teleop.get_inference_status"
UNLOAD_INFERENCE_MODEL = "teleop.unload_inference_model"
ASK_CAMERA_VISION = "teleop.ask_camera_vision"

ACTION_NAMES = (
    START_TELEOPERATION,
    STOP_TELEOPERATION,
    GET_TELEOPERATION_STATUS,
    GET_JOINT_CURVES,
    START_INFERENCE,
    STOP_INFERENCE,
    GET_INFERENCE_STATUS,
    UNLOAD_INFERENCE_MODEL,
    ASK_CAMERA_VISION,
)

# Per-action remote tool name and MCP input schema, mirroring tools/list from
# the live server (descriptions translated/condensed for the system prompt).
_SPECS: dict[str, dict[str, Any]] = {
    START_TELEOPERATION: {
        "tool": "start_teleoperation",
        "description": (
            "启动 SO101 主从臂遥操，会使实体机械臂通电运动。仅在用户明确要求启动、"
            "且已确认现场有人看护、工作区无人无遮挡时使用。参数可省略，默认 front_fps=30、third_fps=30、servo_hz=60。"
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "front_fps": {"type": "integer", "minimum": 1, "maximum": 60, "description": "腕部摄像头 FPS，默认 30。"},
                "third_fps": {"type": "integer", "minimum": 1, "maximum": 60, "description": "场地摄像头 FPS，默认 30。"},
                "servo_hz": {"type": "integer", "minimum": 1, "maximum": 120, "description": "主从臂遥操循环频率，默认 60。"},
            },
        },
    },
    STOP_TELEOPERATION: {
        "tool": "stop_teleoperation",
        "description": "停止当前遥操，并执行从臂扭矩关闭校验。用户要求停止或出现异常时用。",
        "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
    },
    GET_TELEOPERATION_STATUS: {
        "tool": "get_teleoperation_status",
        "description": "查询遥操运行状态、最新主从臂六轴曲线及摄像头视频流/看板地址。",
        "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
    },
    GET_JOINT_CURVES: {
        "tool": "get_joint_curves",
        "description": "获取最近若干秒的主臂、从臂六轴时间序列。seconds 取 1～120，默认 30。",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "seconds": {"type": "number", "minimum": 1, "maximum": 120, "description": "回溯秒数，默认 30。"},
            },
        },
    },
    START_INFERENCE: {
        "tool": "start_inference",
        "description": (
            "启动本机 VLA 推理，会使实体机械臂通电运动。instruction 是用户的自然语言任务指令，"
            "服务端据此映射规范提示词、权重及其预绑定的 Corrector；不能由调用方指定模型或 Corrector。"
            "仅在用户明确要求执行某任务、且已确认现场有人看护、工作区无人无遮挡时使用。"
            "resident 可省略，默认 false（任务结束即卸载模型）；true 表示停止后保留模型热驻。"
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["instruction"],
            "properties": {
                "instruction": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 2000,
                    "description": "自然语言任务指令，例如“把紫色的鸭子夹到纸盒里”。",
                },
                "resident": {
                    "type": "boolean",
                    "description": "任务停止后是否保留模型热驻，默认 false。",
                },
            },
        },
    },
    STOP_INFERENCE: {
        "tool": "stop_inference",
        "description": "停止当前推理并执行硬件清理；长驻模式下保留模型热加载。用户要求停止或出现异常时用。",
        "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
    },
    GET_INFERENCE_STATUS: {
        "tool": "get_inference_status",
        "description": "查询推理阶段、所用模型、Corrector、长驻和硬件清理状态。",
        "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
    },
    UNLOAD_INFERENCE_MODEL: {
        "tool": "unload_inference_model",
        "description": "在任务已停止后显式卸载长驻模型；不移动机械臂。",
        "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
    },
    ASK_CAMERA_VISION: {
        "tool": "ask_camera_vision",
        "description": (
            "获取当前 front/third 双摄截图，交给服务端配置的外部视觉模型回答 question。"
            "不会启动、通电或移动机械臂；但会把画面截图发送到外部模型接口。"
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["question"],
            "properties": {
                "question": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 2000,
                    "description": "针对当前双摄画面的问题，例如“你看到了什么？”。",
                },
            },
        },
    },
}


@dataclass(frozen=True)
class TeleopSkillResult:
    action: str
    ok: bool
    structured: Any
    fallback_summary: str


class TeleopMCPSkill:
    """Call the LeRobot teleop/inference/vision MCP server for the current turn."""

    @staticmethod
    def is_supported(action: str) -> bool:
        return action in ACTION_NAMES

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        """Action contracts injected into the direct-LLM system prompt."""
        return [
            {
                "name": action,
                "description": spec["description"],
                "arguments": spec["inputSchema"],
            }
            for action, spec in _SPECS.items()
        ]

    def _validate(self, action: str, arguments: object) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError(f"{action} arguments must be an object")
        schema = _SPECS[action]["inputSchema"]
        allowed = set(schema["properties"])
        extra = set(arguments) - allowed
        if extra:
            raise ValueError(f"{action} got unexpected arguments: {', '.join(sorted(extra))}")
        missing = set(schema.get("required", ())) - set(arguments)
        if missing:
            raise ValueError(f"{action} missing required argument(s): {', '.join(sorted(missing))}")
        clean: dict[str, Any] = {}
        for name, bound in schema["properties"].items():
            if name not in arguments:
                continue
            value = arguments[name]
            btype = bound.get("type")
            if btype == "integer":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{action}.{name} must be an integer")
            elif btype == "number":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{action}.{name} must be a number")
            elif btype == "string":
                if not isinstance(value, str):
                    raise ValueError(f"{action}.{name} must be a string")
            elif btype == "boolean":
                if not isinstance(value, bool):
                    raise ValueError(f"{action}.{name} must be a boolean")
            if btype in ("integer", "number"):
                low, high = bound.get("minimum"), bound.get("maximum")
                if low is not None and value < low:
                    raise ValueError(f"{action}.{name} must be >= {low}")
                if high is not None and value > high:
                    raise ValueError(f"{action}.{name} must be <= {high}")
            if btype == "string":
                min_len, max_len = bound.get("minLength"), bound.get("maxLength")
                if min_len is not None and len(value) < min_len:
                    raise ValueError(f"{action}.{name} must be at least {min_len} characters")
                if max_len is not None and len(value) > max_len:
                    raise ValueError(f"{action}.{name} must be at most {max_len} characters")
            clean[name] = value
        return clean

    async def execute(self, action: str, arguments: object) -> TeleopSkillResult:
        if action not in ACTION_NAMES:
            raise ValueError(f"unsupported teleop action: {action}")
        if not config.TELEOP_MCP_URL or not config.TELEOP_MCP_TOKEN:
            raise RuntimeError("teleop MCP url/token is not configured")
        payload_args = self._validate(action, arguments)

        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": _SPECS[action]["tool"], "arguments": payload_args},
        }
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(config.TELEOP_TIMEOUT_SECONDS, connect=5.0)
        ) as client:
            response = await client.post(
                config.TELEOP_MCP_URL,
                json=body,
                headers={
                    "Authorization": f"Bearer {config.TELEOP_MCP_TOKEN}",
                    "Accept": "application/json, text/event-stream",
                },
            )
            response.raise_for_status()
            data = _decode_response(response.text)

        if "error" in data:
            message = (data["error"] or {}).get("message", "teleop MCP returned an error")
            return TeleopSkillResult(action, ok=False, structured=None, fallback_summary=f"调用失败：{message}")

        result = data.get("result") or {}
        structured = result.get("structuredContent")
        text_parts = [
            block.get("text", "")
            for block in result.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        is_error = bool(result.get("isError"))
        fallback = _summarize(action, structured, text_parts, is_error)
        return TeleopSkillResult(
            action=action,
            ok=not is_error,
            structured=structured,
            fallback_summary=fallback,
        )


def _decode_response(raw: str) -> dict[str, Any]:
    """Return the JSON-RPC object, tolerating an SSE ``data:`` framed body."""
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            try:
                return json.loads(line[len("data:"):].strip())
            except json.JSONDecodeError:
                continue
    raise RuntimeError("teleop MCP returned an unparseable response")


def _summarize(action: str, structured: Any, text_parts: list[str], is_error: bool) -> str:
    """Condensed fallback narration; the agent's second model pass normally replaces this."""
    if is_error:
        detail = " ".join(text_parts).strip()
        return detail or "设备操作失败。"

    if not isinstance(structured, dict):
        joined = " ".join(part.strip() for part in text_parts if part.strip())
        return joined or "已完成设备操作。"

    if action == GET_TELEOPERATION_STATUS:
        status = structured.get("status") or {}
        mode = status.get("mode", "unknown")
        running = bool(status.get("running"))
        parts = [f"当前遥操状态：{mode}（{'运行中' if running else '未运行'}）"]
        cfg = status.get("teleop_config") or {}
        if cfg:
            parts.append(
                f"参数 front {cfg.get('front_fps')}fps、third {cfg.get('third_fps')}fps、舵机 {cfg.get('servo_hz')}hz"
            )
        return "；".join(parts) + "。"

    if action == GET_JOINT_CURVES:
        seconds = structured.get("seconds", 30)
        return f"已获取最近 {seconds} 秒的主从臂六轴曲线数据。"

    if action == START_TELEOPERATION:
        cfg = structured.get("teleop_config")
        if not isinstance(cfg, dict):
            fb = structured.get("feedback") if isinstance(structured.get("feedback"), dict) else {}
            cfg = fb.get("parameters") if isinstance(fb.get("parameters"), dict) else {}
        tail = f"，参数 front {cfg.get('front_fps')}fps、third {cfg.get('third_fps')}fps、舵机 {cfg.get('servo_hz')}hz" if cfg else ""
        return "遥操已启动，机械臂开始跟随主臂动作" + tail + "。"

    if action == STOP_TELEOPERATION:
        return "遥操已停止，从臂扭矩已关闭。"

    if action == START_INFERENCE:
        start_status = structured.get("start_status", "queued")
        return f"推理启动请求已提交（当前阶段 {start_status}）。"

    if action == STOP_INFERENCE:
        return "推理已停止，硬件已清理。"

    if action == GET_INFERENCE_STATUS:
        status = structured.get("status") or {}
        phase = status.get("phase", "unknown")
        running = bool(status.get("running"))
        parts = [f"当前推理阶段：{phase}（{'运行中' if running else '未运行'}）"]
        if status.get("resident"):
            parts.append("模型长驻中")
        if status.get("total_dispatches") is not None:
            parts.append(f"累计下发 {status.get('total_dispatches')} 次")
        return "；".join(parts) + "。"

    if action == UNLOAD_INFERENCE_MODEL:
        return "长驻模型已卸载。"

    if action == ASK_CAMERA_VISION:
        answer = str(structured.get("answer") or "").strip()
        return answer or "已获取双摄画面的视觉回答。"

    return "已完成遥操操作。"
