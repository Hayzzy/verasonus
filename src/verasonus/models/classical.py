"""Classical baselines on handcrafted features [aspects 1 (supervised) and 2 (ML)] (Phase 1).

Fast, explainable baselines: an RBF SVM, a random forest and gradient boosting, each wrapped
in a scikit-learn Pipeline with feature scaling and tuned on the validation split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

ClassicalModel = Literal["svm", "random_forest", "gradient_boosting"]


def build_classical(name: ClassicalModel, seed: int = 42, **params: Any) -> Pipeline:
    """Return an untrained scaling + classifier Pipeline."""
    raise NotImplementedError("Phase 1")


def train_classical(
    name: ClassicalModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> tuple[Pipeline, dict[str, float]]:
    """Tune hyperparameters on the validation split; return the model and its val metrics."""
    raise NotImplementedError("Phase 1")
