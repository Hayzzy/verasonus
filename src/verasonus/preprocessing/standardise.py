"""Bring every clip to one format so models cannot learn file format instead of "AI" (Phase 1).

For each clip, real or AI, the same steps run in the same order:
    decode -> mono -> resample to cfg.audio.target_sample_rate
    -> cut long clips (full songs) to one excerpt (cfg.audio.long_clip_seconds)
    -> low-pass at cfg.audio.lowpass_hz -> scale to cfg.audio.target_rms_dbfs
    -> shared lossy codec round trip (cfg.audio.shared_codec), the same for every clip
    -> save as FLAC / PCM_16 under data/interim
Afchar et al. (2024) show that detectors happily learn such shortcuts, so also audit the
result: bandwidth, loudness and duration should not separate the classes on their own.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from verasonus.config import AudioConfig
from verasonus.preprocessing.audio_io import AudioInfo, audio_info, load_audio, save_audio

_EXTENSIONS = {"FLAC": ".flac", "WAV": ".wav"}


def interim_path(raw_path: str, audio_cfg: AudioConfig) -> str:
    """Map a metadata ``path`` under ``raw/`` to its standardised copy under ``interim/``.

    ``raw/fma/fma_small/000/000002.mp3`` becomes ``interim/fma/fma_small/000/000002.flac``.
    Both paths are relative to the data root.
    """
    path = PurePosixPath(raw_path)
    if not path.parts or path.parts[0] != "raw":
        raise ValueError(f"expected a path under raw/, got {raw_path!r}")
    suffix = _EXTENSIONS.get(audio_cfg.storage_format.upper())
    if suffix is None:
        raise ValueError(f"unsupported storage_format {audio_cfg.storage_format!r}")
    return PurePosixPath("interim", *path.parts[1:]).with_suffix(suffix).as_posix()


def standardise_file(src: Path, dst: Path, audio_cfg: AudioConfig) -> AudioInfo:
    """Standardise one file and return the properties of the written file.

    Clips longer than ``audio_cfg.long_clip_seconds`` are cut to one excerpt first.
    """
    audio, sample_rate = load_audio(
        src, target_sr=audio_cfg.target_sample_rate, mono=audio_cfg.mono
    )
    audio = standardise_audio(excerpt(audio, sample_rate, audio_cfg), sample_rate, audio_cfg)
    save_audio(
        dst,
        audio,
        sample_rate,
        format=audio_cfg.storage_format,
        subtype=audio_cfg.storage_subtype,
    )
    return audio_info(dst)


def excerpt(audio: np.ndarray, sample_rate: int, audio_cfg: AudioConfig) -> np.ndarray:
    """Cut a long clip to ``excerpt_seconds`` starting 40% in; shorter clips are unchanged."""
    if not audio_cfg.long_clip_seconds or len(audio) <= audio_cfg.long_clip_seconds * sample_rate:
        return audio
    length = round(audio_cfg.excerpt_seconds * sample_rate)
    start = int(0.4 * (len(audio) - length))
    return audio[start : start + length]


def standardise_audio(audio: np.ndarray, sample_rate: int, audio_cfg: AudioConfig) -> np.ndarray:
    """Low-pass, loudness and shared codec for audio already at the target rate (in memory).

    The same steps run for training clips (standardise_file) and for uploads in the API.
    """
    if audio_cfg.lowpass_hz:
        audio = lowpass(audio, sample_rate, audio_cfg.lowpass_hz)
    if audio_cfg.target_rms_dbfs is not None:
        audio = match_loudness(audio, audio_cfg.target_rms_dbfs)
    if audio_cfg.shared_codec:
        from verasonus.augmentation.codecs import reencode

        codec = audio_cfg.shared_codec
        audio = reencode(audio, sample_rate, codec["codec"], int(codec["bitrate_kbps"]))
    return audio


def lowpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Steep zero-phase low-pass filter (12th-order Butterworth, run forwards and backwards)."""
    sos = butter(12, cutoff_hz, btype="low", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def match_loudness(audio: np.ndarray, target_rms_dbfs: float, max_peak: float = 0.99) -> np.ndarray:
    """Scale ``audio`` to an RMS level of ``target_rms_dbfs``, turned down if it would clip."""
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms == 0:
        return audio
    gain = 10 ** (target_rms_dbfs / 20) / rms
    peak = float(np.abs(audio).max()) * gain
    if peak > max_peak:
        gain *= max_peak / peak
    return (audio * gain).astype(np.float32)


def effective_bandwidth_hz(path: Path, floor_db: float = -60.0) -> float:
    """Estimate the highest frequency carrying real energy, to audit bandwidth shortcuts.

    For example, MusicGen audio stops at 16 kHz while a 44.1 kHz WAV reaches 22 kHz.
    Returns the highest frequency whose average power is within ``floor_db`` of the
    loudest frequency, or 0.0 for silence.
    """
    audio, sample_rate = load_audio(path, mono=True)
    if not np.any(audio):
        return 0.0
    freqs, power = welch(audio, fs=sample_rate, nperseg=min(4096, len(audio)))
    level_db = 10 * np.log10(power + 1e-20)
    above = np.nonzero(level_db >= level_db.max() + floor_db)[0]
    return float(freqs[above[-1]])
