"""Bring every clip to one format so models cannot learn file format instead of "AI" (Phase 1).

For each clip, real or AI, the same steps run in the same order:
    decode -> mono -> resample to cfg.audio.target_sample_rate
    -> optional shared lossy codec round trip (cfg.audio.shared_codec)
    -> save as FLAC / PCM_16 under data/interim
Afchar et al. (2024) show that detectors happily learn such shortcuts, so also audit the
result: bandwidth, loudness and duration should not separate the classes on their own.
"""

from __future__ import annotations

from pathlib import Path

from verasonus.config import AudioConfig
from verasonus.preprocessing.audio_io import AudioInfo


def standardise_file(src: Path, dst: Path, audio_cfg: AudioConfig) -> AudioInfo:
    """Standardise one file and return the properties of the written file."""
    raise NotImplementedError("Phase 1")


def effective_bandwidth_hz(path: Path, floor_db: float = -60.0) -> float:
    """Estimate the highest frequency carrying real energy, to audit bandwidth shortcuts.

    For example, MusicGen audio stops at 16 kHz while a 44.1 kHz WAV reaches 22 kHz.
    """
    raise NotImplementedError("Phase 1")
