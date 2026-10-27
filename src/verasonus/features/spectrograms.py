"""Time-frequency representations for the deep models [aspect 4] (Phase 1-2)."""

from __future__ import annotations

import numpy as np


def mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    log: bool = True,
) -> np.ndarray:
    """Return a ``(n_mels, frames)`` mel spectrogram, in dB when ``log`` is true."""
    raise NotImplementedError("Phase 1")


def cqt_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    hop_length: int = 512,
) -> np.ndarray:
    """Return a ``(n_bins, frames)`` constant-Q spectrogram in dB (pitch-aligned bins)."""
    raise NotImplementedError("Phase 1")
