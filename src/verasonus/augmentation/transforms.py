"""Signal edits for the robustness suite and training-time augmentation (Phase 1).

A published detector was fooled by a 2-semitone pitch shift, so every edit here is also
used to build per-edit test reports (verasonus.evaluation.protocols).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal

import numpy as np


def pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    """Shift pitch by ``semitones`` without changing duration (e.g. librosa.effects)."""
    raise NotImplementedError("Phase 1")


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Speed up (rate > 1) or slow down (rate < 1) without changing pitch."""
    raise NotImplementedError("Phase 1")


def add_noise(
    audio: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    colour: Literal["white", "pink"] = "white",
) -> np.ndarray:
    """Add noise at a given signal-to-noise ratio in dB."""
    raise NotImplementedError("Phase 1")


def normalise_loudness(audio: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
    """Normalise integrated loudness (ITU-R BS.1770) to ``target_lufs``."""
    raise NotImplementedError("Phase 1")


def apply_suite(
    audio: np.ndarray,
    sample_rate: int,
    suite: Mapping[str, Any],
    rng: np.random.Generator,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(edit_name, edited_audio)`` for every edit in configs/augment.yaml.

    Edit names are stable, e.g. "mp3_128", "pitch_+2", "noise_20db", "phone".
    """
    raise NotImplementedError("Phase 1")
