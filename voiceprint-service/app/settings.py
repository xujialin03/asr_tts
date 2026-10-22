from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOICEPRINT_", extra="ignore")

    database_url: str = "postgresql://qwen3_asr:change-me@postgres:5432/qwen3_asr"
    model_id: str = "iic/speech_campplus_sv_zh-cn_16k-common"
    model_revision: str = "v1.0.0"
    model_dir: str = "/models"
    device: str = "cpu"
    sample_rate: int = 16000
    embedding_dimension: int = 192
    min_audio_seconds: float = 1.5
    max_audio_seconds: float = 30.0
    identify_threshold: float = 0.55
    uncertain_threshold: float = 0.45
    # When resolve scores a segment in the uncertain band against an existing
    # identity, treat it as the same person and append the segment as an extra
    # template instead of creating a new anonymous speaker. Makes a returning
    # user progressively easier to recognize (self-improving enrollment).
    auto_enroll_uncertain: bool = True


settings = Settings()
