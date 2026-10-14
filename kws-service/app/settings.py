from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KWS_", extra="ignore")

    enabled: bool = True
    model_root: Path = Path("/models")
    model_name: str = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    keywords_file: Path = Path("/app/keywords.txt")
    sample_rate: int = 16000
    num_threads: int = 2
    keywords_score: float = 1.5
    keywords_threshold: float = 0.20
    num_trailing_blanks: int = 1
    max_audio_seconds: float = 30.0

    @property
    def model_dir(self) -> Path:
        return self.model_root / self.model_name


settings = Settings()
