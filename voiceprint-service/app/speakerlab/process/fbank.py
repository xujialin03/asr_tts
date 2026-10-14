# Adapted from ModelScope 3D-Speaker's speakerlab/process/processor.py.
# Licensed under the Apache License, Version 2.0.

import torch
import torchaudio.compliance.kaldi as kaldi


class FBank:
    def __init__(self, n_mels: int = 80, sample_rate: int = 16000, mean_nor: bool = True):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            wav = wav[:1, :]
        feat = kaldi.fbank(
            wav,
            num_mel_bins=self.n_mels,
            sample_frequency=self.sample_rate,
            dither=0,
        )
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat
