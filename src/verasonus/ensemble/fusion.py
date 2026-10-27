"""Combine model families [aspect 3: hybrid / ensemble] (Phase 2).

Weighted voting (weights tuned on validation) and stacking (a logistic-regression meta-model
trained on out-of-fold window scores from the classical, deep and probe models).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression


def weighted_vote(scores: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """Weighted average of each model's AI probabilities."""
    raise NotImplementedError("Phase 2")


def fit_stacker(
    oof_scores: Mapping[str, np.ndarray], labels: np.ndarray, seed: int = 42
) -> LogisticRegression:
    """Train the meta-model on out-of-fold scores (never on scores from training data)."""
    raise NotImplementedError("Phase 2")


def stack(stacker: LogisticRegression, scores: Mapping[str, np.ndarray]) -> np.ndarray:
    """Apply the meta-model to new window scores."""
    raise NotImplementedError("Phase 2")
