from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from modelscope.hub.snapshot_download import snapshot_download

from app.settings import settings
from app.speakerlab.models.campplus.DTDNN import CAMPPlus
from app.speakerlab.process.fbank import FBank

MODEL_FILE = "campplus_cn_common.bin"


class AudioValidationError(ValueError):
    pass


@dataclass(frozen=True)
class AudioInfo:
    samples: np.ndarray
    duration_ms: int
    sha256: str
    rms: float
    peak: float


class SpeakerEncoder:
    def __init__(self) -> None:
        self.device = torch.device(settings.device)
        self.model: CAMPPlus | None = None
        self.feature_extractor = FBank(sample_rate=settings.sample_rate)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        cache_dir = snapshot_download(
            settings.model_id,
            revision=settings.model_revision,
            cache_dir=settings.model_dir,
        )
        checkpoint_path = Path(cache_dir) / MODEL_FILE
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"CAM++ checkpoint not found: {checkpoint_path}")

        model = CAMPPlus(feat_dim=80, embedding_size=settings.embedding_dimension)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model

    def parse_pcm(self, body: bytes) -> AudioInfo:
        if not body:
            raise AudioValidationError("audio body is empty")
        if len(body) % 4:
            raise AudioValidationError("audio must be little-endian float32 PCM")

        samples = np.frombuffer(body, dtype="<f4")
        if not np.isfinite(samples).all():
            raise AudioValidationError("audio contains non-finite samples")

        duration = samples.size / settings.sample_rate
        if duration < settings.min_audio_seconds:
            raise AudioValidationError(
                f"audio is too short; need at least {settings.min_audio_seconds:.1f} seconds"
            )
        if duration > settings.max_audio_seconds:
            raise AudioValidationError(
                f"audio is too long; maximum is {settings.max_audio_seconds:.1f} seconds"
            )

        samples = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        peak = float(np.max(np.abs(samples)))
        if rms < 0.001:
            raise AudioValidationError("audio level is too low")

        return AudioInfo(
            samples=samples,
            duration_ms=round(duration * 1000),
            sha256=hashlib.sha256(body).hexdigest(),
            rms=rms,
            peak=peak,
        )

    def embed(self, audio: AudioInfo) -> list[float]:
        if self.model is None:
            raise RuntimeError("speaker model is not loaded")
        wav = torch.from_numpy(audio.samples.copy()).unsqueeze(0)
        feat = self.feature_extractor(wav).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            embedding = self.model(feat).squeeze(0)
            embedding = torch.nn.functional.normalize(embedding, dim=0)
        values = embedding.cpu().tolist()
        if len(values) != settings.embedding_dimension or not all(math.isfinite(v) for v in values):
            raise RuntimeError("speaker model returned an invalid embedding")
        return values


encoder = SpeakerEncoder()
