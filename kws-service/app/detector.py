from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sherpa_onnx

from app.settings import settings


class AudioValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Detection:
    detected: bool
    keyword: str | None
    duration_ms: int


class KeywordDetector:
    def __init__(self) -> None:
        self.spotter: sherpa_onnx.KeywordSpotter | None = None

    @property
    def ready(self) -> bool:
        return self.spotter is not None

    def load(self) -> None:
        if not settings.enabled:
            return
        root = settings.model_dir
        self.spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(root / "tokens.txt"),
            encoder=str(root / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            decoder=str(root / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
            joiner=str(root / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            keywords_file=str(settings.keywords_file),
            num_threads=settings.num_threads,
            sample_rate=settings.sample_rate,
            keywords_score=settings.keywords_score,
            keywords_threshold=settings.keywords_threshold,
            num_trailing_blanks=settings.num_trailing_blanks,
            provider="cpu",
        )

    def parse_pcm(self, body: bytes) -> np.ndarray:
        if not body:
            raise AudioValidationError("audio body is empty")
        if len(body) % 4:
            raise AudioValidationError("audio must be little-endian float32 PCM")
        samples = np.frombuffer(body, dtype="<f4")
        if not np.isfinite(samples).all():
            raise AudioValidationError("audio contains non-finite samples")
        duration = samples.size / settings.sample_rate
        if duration > settings.max_audio_seconds:
            raise AudioValidationError(
                f"audio is too long; maximum is {settings.max_audio_seconds:.1f} seconds"
            )
        return np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)

    def detect(self, samples: np.ndarray) -> Detection:
        if self.spotter is None:
            raise RuntimeError("keyword spotter is not loaded")
        stream = self.spotter.create_stream()
        stream.accept_waveform(settings.sample_rate, samples)
        # Flush the streaming decoder after a complete VAD utterance.
        tail = np.zeros(round(0.66 * settings.sample_rate), dtype=np.float32)
        stream.accept_waveform(settings.sample_rate, tail)
        stream.input_finished()

        keyword = None
        while self.spotter.is_ready(stream):
            self.spotter.decode_stream(stream)
            result = self.spotter.get_result(stream)
            if result:
                keyword = result
                self.spotter.reset_stream(stream)
                break
        return Detection(
            detected=keyword is not None,
            keyword=keyword,
            duration_ms=math.floor(samples.size * 1000 / settings.sample_rate),
        )


detector = KeywordDetector()
