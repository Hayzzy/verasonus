"""Audio loading, down-mixing, resampling and saving (implemented in Phase 0).

Arrays are float32, shaped ``(samples,)`` for mono or ``(channels, samples)`` otherwise.
Decoding uses libsndfile through ``soundfile`` (WAV, FLAC, OGG/Opus and MP3). Resampling uses
``soxr``, which is fast, high quality and needs no ffmpeg. Formats libsndfile cannot read,
such as AAC/M4A, must be converted with ffmpeg first; that arrives with the codec tools in
Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


@dataclass(frozen=True)
class AudioInfo:
    """File properties recorded in the metadata table before any resampling."""

    path: str
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    format: str  # container, e.g. "WAV", "FLAC", "MP3"
    subtype: str  # encoding, e.g. "PCM_16", "MPEG_LAYER_III"


def audio_info(path: str | Path) -> AudioInfo:
    """Read a file's sample rate, channels, length and codec without decoding it."""
    info = sf.info(str(path))
    return AudioInfo(
        path=str(path),
        sample_rate=info.samplerate,
        channels=info.channels,
        frames=info.frames,
        duration_seconds=info.frames / info.samplerate,
        format=info.format,
        subtype=info.subtype,
    )


def load_audio(
    path: str | Path, target_sr: int | None = None, mono: bool = True
) -> tuple[np.ndarray, int]:
    """Decode a file to float32 in [-1, 1], optionally down-mix and resample it.

    Returns ``(audio, sample_rate)``. ``audio`` is ``(samples,)`` when ``mono`` is true,
    otherwise ``(channels, samples)`` (also for single-channel files).
    """
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.T  # soundfile gives (samples, channels)
    if mono:
        audio = to_mono(audio)
    if target_sr is not None and target_sr != sample_rate:
        audio = resample(audio, sample_rate, target_sr)
        sample_rate = target_sr
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Average the channels of ``(channels, samples)`` audio. 1-D input is returned as is."""
    audio = np.asarray(audio)
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {audio.shape}")
    return audio.mean(axis=0).astype(np.float32)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int, quality: str = "HQ") -> np.ndarray:
    """Resample ``(samples,)`` or ``(channels, samples)`` audio with soxr."""
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive")
    audio = np.asarray(audio, dtype=np.float32)
    if orig_sr == target_sr:
        return audio
    if audio.ndim == 1:
        return soxr.resample(audio, orig_sr, target_sr, quality=quality)
    if audio.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {audio.shape}")
    # soxr works on (samples, channels)
    out = soxr.resample(np.ascontiguousarray(audio.T), orig_sr, target_sr, quality=quality)
    return np.ascontiguousarray(out.T)


def save_audio(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    format: str | None = None,
    subtype: str | None = None,
) -> Path:
    """Write ``(samples,)`` or ``(channels, samples)`` audio; format follows the extension.

    Samples outside [-1, 1] are clipped unless the subtype is FLOAT or DOUBLE.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    if subtype not in ("FLOAT", "DOUBLE"):
        audio = np.clip(audio, -1.0, 1.0)
    data = audio.T if audio.ndim == 2 else audio
    sf.write(str(path), data, sample_rate, format=format, subtype=subtype)
    return path
