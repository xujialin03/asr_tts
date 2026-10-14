import numpy as np
import pytest

from app.model import AudioValidationError, SpeakerEncoder


def pcm(seconds: float, amplitude: float = 0.1) -> bytes:
    count = round(16000 * seconds)
    samples = np.full(count, amplitude, dtype="<f4")
    return samples.tobytes()


def test_parse_pcm_accepts_valid_audio():
    audio = SpeakerEncoder().parse_pcm(pcm(2.0))
    assert audio.duration_ms == 2000
    assert audio.rms == pytest.approx(0.1)
    assert audio.peak == pytest.approx(0.1)
    assert len(audio.sha256) == 64


def test_parse_pcm_rejects_short_audio():
    with pytest.raises(AudioValidationError, match="too short"):
        SpeakerEncoder().parse_pcm(pcm(1.0))


def test_parse_pcm_rejects_silence():
    with pytest.raises(AudioValidationError, match="too low"):
        SpeakerEncoder().parse_pcm(pcm(2.0, 0.0))
