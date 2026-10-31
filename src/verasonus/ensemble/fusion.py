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
    total = sum(weights[name] for name in scores)
    if total <= 0:
        raise ValueError("weights must add up to more than 0")
    return sum(weights[name] * np.asarray(s, dtype=float) for name, s in scores.items()) / total


def tune_weights(
    scores: Mapping[str, np.ndarray],
    labels: np.ndarray,
    step: float = 0.1,
    min_weight: float = 0.0,
) -> dict[str, float]:
    """Weights for any number of models, picked on val by ROC-AUC.

    Every combination of weights in multiples of ``step`` that sum to 1 is tried; ties go to
    the most even mix. ``min_weight`` keeps each model in the mix even when val prefers one
    alone.
    """
    from itertools import product

    from sklearn.metrics import roc_auc_score

    names = list(scores)
    steps = round(1 / step)
    low = round(min_weight / step)
    best: tuple[float, float, dict[str, float]] | None = None
    for combo in product(range(low, steps + 1), repeat=len(names)):
        if sum(combo) != steps:
            continue
        weights = {n: round(c * step, 6) for n, c in zip(names, combo, strict=True)}
        auc = roc_auc_score(labels, weighted_vote(scores, weights))
        key = (auc, -float(np.std(list(weights.values()))))
        if best is None or key > best[:2]:
            best = (*key, weights)
    if best is None:
        raise ValueError("min_weight is too large for this many models")
    return best[2]


def fit_stacker(
    oof_scores: Mapping[str, np.ndarray], labels: np.ndarray, seed: int = 42
) -> LogisticRegression:
    """Train the meta-model on out-of-fold scores (never on scores from training data)."""
    from sklearn.linear_model import LogisticRegression

    x = np.column_stack([oof_scores[name] for name in sorted(oof_scores)])
    return LogisticRegression(random_state=seed).fit(x, labels)


def stack(stacker: LogisticRegression, scores: Mapping[str, np.ndarray]) -> np.ndarray:
    """Apply the meta-model to new window scores."""
    x = np.column_stack([scores[name] for name in sorted(scores)])
    return stacker.predict_proba(x)[:, 1]
