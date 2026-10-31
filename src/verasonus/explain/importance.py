"""Feature importance for the classical models [aspect 4] (Phase 1-2).

Permutation importance on the validation split (works for any model), plus per-track
contributions for the explanation panel, e.g. "strong periodic spectral peaks pushed this
towards AI".
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def permutation_importance_table(
    model: Any,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Sequence[str],
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Mean and std drop in validation ROC-AUC when each feature is shuffled."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, x_val, y_val, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=-1
    )
    table = pd.DataFrame(
        {"importance_mean": result.importances_mean, "importance_std": result.importances_std},
        index=pd.Index(list(feature_names), name="feature"),
    )
    return table.sort_values("importance_mean", ascending=False)


def explain_prediction(
    model: Any,
    x: np.ndarray,
    feature_names: Sequence[str],
    reference: np.ndarray,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Top features pushing one prediction towards AI (+) or real (-).

    ``x`` holds one or more windows of a track. Each feature in turn is reset to its
    ``reference`` value (e.g. the median of the real training windows); how far the mean AI
    score falls is that feature's contribution. Works for any model.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n_windows, n = x.shape
    base = float(model.predict_proba(x)[:, 1].mean())
    reset = np.repeat(x[None], n, axis=0)  # (feature reset, window, feature)
    reset[np.arange(n), :, np.arange(n)] = np.asarray(reference, dtype=float)[:, None]
    scores = model.predict_proba(reset.reshape(-1, n))[:, 1].reshape(n, n_windows).mean(axis=1)
    contributions = base - scores
    order = np.argsort(-np.abs(contributions))[:top_k]
    return [(feature_names[i], float(contributions[i])) for i in order]
