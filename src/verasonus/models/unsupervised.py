"""Unsupervised analysis [aspect 1: unsupervised learning] (Phase 2).

* Embedding maps (PCA or t-SNE; UMAP if you add umap-learn) show how generators cluster.
* Clustering checks whether clusters match generators without using labels.
* An outlier detector fitted on known generators reports tracks from unfamiliar generators
  as "unknown" instead of forcing them into a known class.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


def embedding_map(
    embeddings: np.ndarray, method: Literal["pca", "tsne"] = "pca", seed: int = 42
) -> np.ndarray:
    """Project embeddings to 2-D for plotting: ``(n, 2)``."""
    raise NotImplementedError("Phase 2")


def cluster(embeddings: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    """Cluster labels for each embedding (e.g. k-means)."""
    raise NotImplementedError("Phase 2")


def fit_unknown_detector(known_embeddings: np.ndarray, contamination: float = 0.01) -> Any:
    """Fit an outlier detector (e.g. IsolationForest) on embeddings of known generators."""
    raise NotImplementedError("Phase 2")


def is_unknown(detector: Any, embeddings: np.ndarray) -> np.ndarray:
    """Boolean mask: True where an embedding looks unlike every known generator."""
    raise NotImplementedError("Phase 2")
