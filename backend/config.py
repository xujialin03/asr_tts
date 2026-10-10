"""Load service config from the repo-root config.yaml.

config.yaml lives one level above the backend package. It holds the ASR/TTS/LLM
service addresses plus the LLM role prompt and TTS voice prompt.
"""

from __future__ import annotations

import os
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
ASR_BASE_URL = _url(services["asr"])
TTS_BASE_URL = _url(services["tts"])
LLM_BASE_URL = _url(services["llm"])

_llm = _cfg.get("llm", {})
LLM_MODEL_ID = _llm.get("model_id", "")
LLM_API_KEY = _llm.get("api_key", "empty")
LLM_TEMPERATURE = _llm.get("temperature", 0.7)
LLM_ROLE_PROMPT = _llm.get("role_prompt", "")

_assistant = _cfg.get("assistant", {})
WAKE_WORDS = _assistant.get("wake_words", ["志玲", "志玲姐姐", "姐姐", "小助手"])
ACTIVE_WINDOW_SECONDS = float(_assistant.get("active_window_seconds", 20))
DIRECT_QUESTION_HINTS = _assistant.get(
    "direct_question_hints",
    ["你觉得", "帮我", "怎么办", "为什么", "能不能", "可以不可以", "陪我", "给我"],
)

_tts = _cfg.get("tts", {})
TTS_PROMPT = _tts.get("prompt", "")
TTS_REF_TEXT = _tts.get("ref_text", "")

# ref_audio is stored relative to the backend dir (e.g. assets/ref_voice.wav).
_ref = _tts.get("ref_audio", "")
TTS_REF_AUDIO = str(Path(__file__).resolve().parent / _ref) if _ref else ""
