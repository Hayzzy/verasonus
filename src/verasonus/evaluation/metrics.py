"""Classification and calibration metrics (Phase 1).

Labels are 1 for AI and 0 for real; scores are AI probabilities.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn import metrics as skm


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Accuracy, precision, recall, F1 and ROC-AUC."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "precision": float(skm.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(skm.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(skm.f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(skm.roc_auc_score(y_true, y_score))
        if len(np.unique(y_true)) == 2
        else float("nan"),
    }


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return ``(eer, threshold)`` where the false-positive and false-negative rates meet."""
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2), float(thresholds[i])


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Average gap between confidence and accuracy across probability bins (ECE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def false_positive_rate(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    """Share of real tracks flagged as AI. Reported separately: it is the costliest error,
    because it accuses a human artist."""
    real = np.asarray(y_true) == 0
    if not real.any():
        return float("nan")
    return float((np.asarray(y_score)[real] >= threshold).mean())


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Confusion matrix (rows: true, columns: predicted), also for generator attribution."""
    return skm.confusion_matrix(y_true, y_pred, labels=list(labels))
