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
    """Find the temperature that minimises validation log-loss.

    Searched between 0.05 and 20: with perfectly separated classes the loss keeps falling as
    the temperature shrinks, so the bound keeps the scores from collapsing to exactly 0 or 1.
    """
    from scipy.optimize import minimize_scalar

    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)

    def loss(t: float) -> float:
        p = np.clip(apply_temperature(logits, t), 1e-12, 1 - 1e-12)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    return float(minimize_scalar(loss, bounds=(0.05, 20.0), method="bounded").x)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrated probabilities from logits."""
    return 1 / (1 + np.exp(-np.asarray(logits, dtype=float) / temperature))


def to_logits(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Logits from probabilities, so averaged model scores can be temperature-scaled."""
    p = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    """Monotonic mapping from raw scores to calibrated probabilities."""
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
