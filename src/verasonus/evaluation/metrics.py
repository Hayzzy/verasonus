"""Classification and calibration metrics (Phase 1).

Labels are 1 for AI and 0 for real; scores are AI probabilities.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Accuracy, precision, recall, F1 and ROC-AUC."""
    raise NotImplementedError("Phase 1")


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return ``(eer, threshold)`` where the false-positive and false-negative rates meet."""
    raise NotImplementedError("Phase 1")


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Average gap between confidence and accuracy across probability bins (ECE)."""
    raise NotImplementedError("Phase 1")


def false_positive_rate(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    """Share of real tracks flagged as AI. Reported separately: it is the costliest error,
    because it accuses a human artist."""
    raise NotImplementedError("Phase 1")


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Confusion matrix (rows: true, columns: predicted), also for generator attribution."""
    raise NotImplementedError("Phase 1")
