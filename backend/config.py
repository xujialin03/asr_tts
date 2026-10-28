"""Load service config from the repo-root config.yaml.

config.yaml lives one level above the backend package. It holds the ASR/TTS
service addresses, the direct-LLM chat settings, and the TTS voice prompt.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

# backend/config.py -> repo root is the parent of the backend dir.
_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = os.getenv("CONFIG_PATH", str(_ROOT / "config.yaml"))


def _load() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_cfg = _load()


def _url(svc: dict) -> str:
    return f"http://{svc['host']}:{svc['port']}"


services = _cfg.get("services", {})
_asr_service = services["asr"]
# The ASR gateway is OpenAI-compatible; base_url already includes the /v1 prefix
# when provided, otherwise host:port is used. An api_key is sent as Bearer auth.
ASR_BASE_URL = (_asr_service.get("base_url") or _url(_asr_service)).rstrip("/")
ASR_API_KEY = _asr_service.get("api_key", "")
ASR_MODEL = _asr_service.get("model", "qwen3-asr")
ASR_TIMEOUT_SECONDS = float(_asr_service.get("timeout_seconds", 60.0))
TTS_BASE_URL = _url(services["tts"])
_voiceprint_service = services.get("voiceprint", {"host": "127.0.0.1", "port": 8002})
VOICEPRINT_BASE_URL = _url(_voiceprint_service)
_kws_service = services.get("kws", {})
KWS_ENABLED = bool(_kws_service.get("enabled", False))
KWS_BASE_URL = _url(_kws_service) if _kws_service.get("host") else ""
KWS_FAIL_OPEN = bool(_kws_service.get("fail_open", True))

_postgresql = _cfg.get("database", {}).get("postgresql", {})
POSTGRES_HOST = _postgresql.get("host", "127.0.0.1")
POSTGRES_PORT = int(_postgresql.get("port", 5432))
POSTGRES_DATABASE = _postgresql.get("database", "qwen3_asr")
POSTGRES_USER = _postgresql.get("user", "qwen3_asr")
POSTGRES_PASSWORD = _postgresql.get("password", "")

# Conversation replies are served by the in-process direct-LLM agent, which calls
# an OpenAI-compatible chat model (see direct_llm below); the backend owns all
# conversation state and history.
_direct_llm = _cfg.get("direct_llm", {})
DIRECT_LLM_BASE_URL = _direct_llm.get("base_url", "http://127.0.0.1:9090/v1").rstrip("/")
DIRECT_LLM_MODEL = _direct_llm.get("model", "")
DIRECT_LLM_API_KEY = _direct_llm.get("api_key", "")
DIRECT_LLM_TIMEOUT_SECONDS = float(_direct_llm.get("timeout_seconds", 30))
DIRECT_LLM_MAX_TOKENS = int(_direct_llm.get("max_tokens", 512))
DIRECT_LLM_TEMPERATURE = float(_direct_llm.get("temperature", 0.7))
DIRECT_LLM_ENABLE_THINKING = bool(_direct_llm.get("enable_thinking", False))

# LeRobot SO101 teleop MCP (Streamable-HTTP JSON-RPC). The direct-LLM agent can
# start/stop teleop and read status/joint curves through this server. It moves a
# physical robot, so TELEOP_ENABLED requires both an explicit opt-in and a token.
# The token lives in config.yaml (gitignored) or TELEOP_MCP_TOKEN in the env.
_teleop = _cfg.get("teleop", {})
TELEOP_MCP_URL = str(_teleop.get("url", "")).strip()
TELEOP_MCP_TOKEN = os.getenv("TELEOP_MCP_TOKEN") or str(_teleop.get("token", "")).strip()
TELEOP_TIMEOUT_SECONDS = float(_teleop.get("timeout_seconds", 30))
TELEOP_ENABLED = bool(_teleop.get("enabled", False)) and bool(
    TELEOP_MCP_URL and TELEOP_MCP_TOKEN
)

_agent = _cfg.get("agent", {})
AGENT_NAME = _agent.get("name", "志玲")
AGENT_ROLE = _agent.get("role", "语音助理")
# Persona injected into the decision-model state so the model knows who the
# assistant is when judging whether a turn is addressed to it.
AGENT_PERSONA = " ".join(str(_agent.get("persona", "")).split()) or (
    f"你叫{AGENT_NAME}，是一名{AGENT_ROLE}，正在多人语音房间中通过麦克风听取用户发言。"
)

_assistant = _cfg.get("assistant", {})
WAKE_WORDS = _assistant.get("wake_words", ["志玲", "志玲姐姐", "姐姐", "小助手"])
# Continuation-fallback gate: when the decision model marks a participant turn
# as addressed to the assistant but chooses not to respond, we still reply if the utterance looks
# like a follow-up question. Markers are interrogative/continuation cues; finals
# are sentence-final particles that usually signal a question.
FOLLOWUP_MARKERS = _assistant.get(
    "followup_markers",
    ["是不是", "有没有", "能不能", "要不要", "怎么", "为什么", "什么", "多少", "几点", "哪里", "哪个", "哪些", "吗", "呢", "吧"],
)
FOLLOWUP_SENTENCE_FINALS = _assistant.get("followup_sentence_finals", ["?", "？", "吗", "呢", "吧", "啊", "呀"])
ACTIVE_WINDOW_SECONDS = float(_assistant.get("active_window_seconds", 20))

# Input buffer: VAD pauses can split one utterance into several ASR segments.
# When the decision model flags a segment as an incomplete fragment, we hold it
# and merge following segments from the same speaker until the utterance is
# complete or the buffer times out.
_buffer = _cfg.get("buffer", {})
BUFFER_ENABLED = bool(_buffer.get("enabled", True))
BUFFER_MAX_WAIT_SECONDS = float(_buffer.get("max_wait_seconds", 3.0))
BUFFER_MAX_SEGMENTS = int(_buffer.get("max_segments", 4))
BUFFER_MAX_CHARS = int(_buffer.get("max_chars", 200))
INTERRUPT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in _assistant.get(
        "interrupt_patterns",
        [
            r"等(?:一)?下",
            r"等一等",
            r"等会(?:儿)?",
            r"稍等(?:一下)?",
            r"停(?:一)?下",
            r"暂停(?:一下)?",
            r"慢着",
            r"先别说(?:了)?",
            r"别说(?:了)?",
            r"打住",
        ],
    )
]

_tts = _cfg.get("tts", {})
TTS_PROMPT = _tts.get("prompt", "")
TTS_CFG_SCALE = max(1.0, min(float(_tts.get("cfg_scale", 4)), 10.0))
TTS_REF_TEXT = _tts.get("ref_text", "")

# ref_audio is stored relative to the backend dir (e.g. assets/ref_voice.wav).
_ref = _tts.get("ref_audio", "")
TTS_REF_AUDIO = str(Path(__file__).resolve().parent / _ref) if _ref else ""

# Interaction decision model: an OpenAI-compatible generative model that judges
# whether a turn is addressed to the assistant and whether to reply now.
_decision_model = _cfg.get("decision_model", {})
DECISION_MODEL_BASE_URL = _decision_model.get(
    "base_url", "http://mo.sy.cn:8000/v1"
)
DECISION_MODEL_NAME = _decision_model.get("model", "qwen3.6-35b-awq")
DECISION_MODEL_API_KEY = _decision_model.get("api_key", "")
DECISION_MODEL_TIMEOUT_SECONDS = float(_decision_model.get("timeout_seconds", 8.0))
DECISION_MODEL_MAX_TOKENS = int(_decision_model.get("max_tokens", 512))
# Thinking mode emits a long reasoning trace before the JSON, burning the output
# budget (truncation) and adding ~3s of latency; disabling it keeps ~1s.
DECISION_MODEL_ENABLE_THINKING = bool(_decision_model.get("enable_thinking", False))
DECISION_MODEL_MAX_RETRIES = int(_decision_model.get("max_retries", 3))
