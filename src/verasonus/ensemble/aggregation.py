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
    """Combine window probabilities into a track score and non-overlapping segments.

    The track score is the mean, the max, or the mean of the ``k`` highest window scores.
    The timeline is split at every window start and end; each piece gets the mean score of
    the windows covering it, so overlapping windows are averaged.
    """
    probs = np.asarray(window_probs, dtype=float)
    starts = np.asarray(starts_seconds, dtype=float)
    if probs.shape != starts.shape or probs.size == 0:
        raise ValueError("need one start time per window score, and at least one window")
    if method == "mean":
        score = probs.mean()
    elif method == "max":
        score = probs.max()
    elif method == "top_k_mean":
        score = np.sort(probs)[-k:].mean()
    else:
        raise ValueError(f"unknown method {method!r}")

    ends = np.minimum(starts + length_seconds, duration_seconds)
    edges = np.unique(np.concatenate([starts, ends]))
    segments = []
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        covering = (starts < end) & (ends > start)
        if covering.any():
            segments.append((float(start), float(end), float(probs[covering].mean())))
    return TrackScore(ai_probability=float(score), segments=segments)
