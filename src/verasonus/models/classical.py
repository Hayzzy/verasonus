"""Classical baselines on handcrafted features [aspects 1 (supervised) and 2 (ML)] (Phase 1).

Fast, explainable baselines: an RBF SVM, a random forest and gradient boosting, each wrapped
in a scikit-learn Pipeline with feature scaling and tuned on the validation split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from verasonus.evaluation.metrics import classification_metrics

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

ClassicalModel = Literal["svm", "random_forest", "gradient_boosting"]

# Small grids: each candidate is trained on train and scored (ROC-AUC) on val.
PARAM_GRIDS: dict[str, list[dict[str, Any]]] = {
    "svm": [{"C": c} for c in (0.3, 1.0, 3.0, 10.0)],
    "random_forest": [{"max_depth": d} for d in (None, 20, 10)],
    "gradient_boosting": [{"learning_rate": lr} for lr in (0.05, 0.1, 0.2)],
}


def build_classical(name: ClassicalModel, seed: int = 42, **params: Any) -> Pipeline:
    """Return an untrained scaling + classifier Pipeline."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if name == "svm":
        # Calibration turns SVM decision values into probabilities.
        svc = SVC(kernel="rbf", random_state=seed, **params)
        model = CalibratedClassifierCV(svc, ensemble=False)
    elif name == "random_forest":
        params = {"n_estimators": 300, "n_jobs": -1, **params}
        model = RandomForestClassifier(random_state=seed, **params)
    elif name == "gradient_boosting":
        model = HistGradientBoostingClassifier(random_state=seed, **params)
    else:
        raise ValueError(f"unknown classical model {name!r}")
    return make_pipeline(StandardScaler(), model)


def train_classical(
    name: ClassicalModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> tuple[Pipeline, dict[str, float]]:
    """Tune hyperparameters on the validation split; return the model and its val metrics."""
    best = None
    for params in PARAM_GRIDS[name]:
        model = build_classical(name, seed=seed, **params).fit(x_train, y_train)
        metrics = classification_metrics(y_val, model.predict_proba(x_val)[:, 1])
        if best is None or metrics["roc_auc"] > best[1]["roc_auc"]:
            best = (model, metrics)
    assert best is not None
    return best
