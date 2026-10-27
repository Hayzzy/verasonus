"""Probability calibration so that a score of 0.8 means roughly 80% (Phase 2).

Temperature scaling for neural-network logits and isotonic regression for other scores, both
fitted on the validation split only. Check the result with reliability diagrams and ECE
(verasonus.evaluation.metrics).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sklearn.isotonic import IsotonicRegression


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find the temperature that minimises validation log-loss."""
    raise NotImplementedError("Phase 2")


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrated probabilities from logits."""
    raise NotImplementedError("Phase 2")


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    """Monotonic mapping from raw scores to calibrated probabilities."""
    raise NotImplementedError("Phase 2")
