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
    raise NotImplementedError("Phase 1")


def explain_prediction(
    model: Any, x: np.ndarray, feature_names: Sequence[str], top_k: int = 5
) -> list[tuple[str, float]]:
    """Top features pushing one prediction towards AI (+) or real (-)."""
    raise NotImplementedError("Phase 2")
