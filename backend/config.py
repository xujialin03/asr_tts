"""Load service config from the repo-root config.yaml.

config.yaml lives one level above the backend package. It holds the ASR/TTS
service addresses, Hermes client settings, and TTS voice prompt.
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
ASR_BASE_URL = _url(services["asr"])
TTS_BASE_URL = _url(services["tts"])
_voiceprint_service = services.get("voiceprint", {"host": "127.0.0.1", "port": 8002})
VOICEPRINT_BASE_URL = _url(_voiceprint_service)

_postgresql = _cfg.get("database", {}).get("postgresql", {})
POSTGRES_HOST = _postgresql.get("host", "127.0.0.1")
POSTGRES_PORT = int(_postgresql.get("port", 5432))
POSTGRES_DATABASE = _postgresql.get("database", "qwen3_asr")
POSTGRES_USER = _postgresql.get("user", "qwen3_asr")
POSTGRES_PASSWORD = _postgresql.get("password", "")

_hermes = _cfg.get("hermes", {})
HERMES_CLIENT_PATH = os.path.expanduser(
    _hermes.get("client_path", "~/worker_agent_client.py")
)
HERMES_TIMEOUT_SECONDS = float(_hermes.get("timeout_seconds", 180))

_assistant = _cfg.get("assistant", {})
WAKE_WORDS = _assistant.get("wake_words", ["志玲", "志玲姐姐", "姐姐", "小助手"])
ACTIVE_WINDOW_SECONDS = float(_assistant.get("active_window_seconds", 20))
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
TTS_REF_TEXT = _tts.get("ref_text", "")

# ref_audio is stored relative to the backend dir (e.g. assets/ref_voice.wav).
_ref = _tts.get("ref_audio", "")
TTS_REF_AUDIO = str(Path(__file__).resolve().parent / _ref) if _ref else ""
