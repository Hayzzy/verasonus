"""Linear and MLP probes on frozen MERT / CLAP embeddings [aspect 2] (Phase 2).

Probes test how much "AI-ness" pretrained audio representations already capture, and they
train in seconds once embeddings are cached.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin


def train_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    kind: Literal["linear", "mlp"] = "linear",
    seed: int = 42,
) -> ClassifierMixin:
    """Fit a class-balanced logistic regression or small MLP on embeddings."""
    raise NotImplementedError("Phase 2")


def compare_layers(
    train_layers: Mapping[int, np.ndarray],
    y_train: np.ndarray,
    val_layers: Mapping[int, np.ndarray],
    y_val: np.ndarray,
) -> dict[int, float]:
    """Validation ROC-AUC of a linear probe per MERT layer, to pick the best layer."""
    raise NotImplementedError("Phase 2")
