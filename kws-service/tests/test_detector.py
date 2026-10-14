import numpy as np
import pytest

from app.detector import AudioValidationError, KeywordDetector


def test_parse_pcm_accepts_float32():
    samples = np.ones(16000, dtype="<f4") * 0.1
    parsed = KeywordDetector().parse_pcm(samples.tobytes())
    assert parsed.dtype == np.float32
    assert parsed.size == 16000


def test_parse_pcm_rejects_unaligned_body():
    with pytest.raises(AudioValidationError, match="float32"):
        KeywordDetector().parse_pcm(b"123")


def test_parse_pcm_rejects_long_audio():
    samples = np.zeros(16000 * 31, dtype="<f4")
    with pytest.raises(AudioValidationError, match="too long"):
        KeywordDetector().parse_pcm(samples.tobytes())
