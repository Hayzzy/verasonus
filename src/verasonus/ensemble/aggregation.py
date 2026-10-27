"""Window scores -> one track score plus a segment timeline [aspects 3 and 4] (Phase 2).

This feeds the web app's timeline: each window's AI probability maps back to its time span
(see verasonus.preprocessing.windowing.Windows), and overlapping windows are averaged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class TrackScore:
    ai_probability: float
    segments: list[tuple[float, float, float]]  # (start_seconds, end_seconds, ai_probability)


def aggregate(
    window_probs: np.ndarray,
    starts_seconds: np.ndarray,
    length_seconds: float,
    duration_seconds: float,
    method: Literal["mean", "max", "top_k_mean"] = "mean",
    k: int = 3,
) -> TrackScore:
    """Combine window probabilities into a track score and non-overlapping segments."""
    raise NotImplementedError("Phase 2")
