"""Time-frequency representations for the deep models [aspect 4] (Phase 1-2)."""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np

from verasonus.config import Config


def mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    log: bool = True,
    fmax: float | None = None,
) -> np.ndarray:
    """Return a ``(n_mels, frames)`` mel spectrogram, in dB when ``log`` is true.

    Frames are not centred (no padding at either end), so ``frames = 1 + (len - n_fft) //
    hop_length``. ``fmax`` defaults to half the sample rate.
    """
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=fmax,
        center=False,
    )
    return librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None) if log else mel


def mel_settings(cfg: Config) -> dict[str, Any]:
    """Arguments for :func:`window_mel`, from the ``spectrogram`` section of the config."""
    spec = cfg.raw["spectrogram"]
    if spec["seconds"] > cfg.windows.min_last_seconds:
        raise ValueError("spectrogram.seconds must not exceed windows.min_last_seconds")
    sample_rate = cfg.audio.target_sample_rate
    n_samples = round(spec["seconds"] * sample_rate)
    return {
        "n_mels": int(spec["n_mels"]),
        "n_fft": int(spec["n_fft"]),
        "hop_length": int(spec["hop_length"]),
        "fmax": float(cfg.audio.lowpass_hz or sample_rate / 2),
        "n_frames": 1 + (n_samples - int(spec["n_fft"])) // int(spec["hop_length"]),
    }


def window_mel(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    fmax: float,
    n_frames: int,
) -> np.ndarray:
    """Log-mel spectrogram of the start of one window, exactly ``n_frames`` long (float16)."""
    n_samples = (n_frames - 1) * hop_length + n_fft
    if len(audio) < n_samples:
        raise ValueError(f"window has {len(audio)} samples, needs {n_samples}")
    mel = mel_spectrogram(audio[:n_samples], sample_rate, n_mels, n_fft, hop_length, fmax=fmax)
    return mel.astype(np.float16)


def cqt_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    hop_length: int = 512,
) -> np.ndarray:
    """Return a ``(n_bins, frames)`` constant-Q spectrogram in dB (pitch-aligned bins)."""
    raise NotImplementedError("Phase 1")
