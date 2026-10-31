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
    from sklearn.preprocessing import StandardScaler

    x = StandardScaler().fit_transform(embeddings)
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(x)
    from sklearn.manifold import TSNE

    return TSNE(n_components=2, random_state=seed, init="pca").fit_transform(x)


def cluster(embeddings: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    """Cluster labels for each embedding (k-means on standardised values)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    x = StandardScaler().fit_transform(embeddings)
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(x)


def fit_unknown_detector(known_embeddings: np.ndarray, contamination: float = 0.01) -> Any:
    """Fit an outlier detector (IsolationForest) on embeddings of known generators."""
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    return make_pipeline(StandardScaler(), forest).fit(known_embeddings)


def is_unknown(detector: Any, embeddings: np.ndarray) -> np.ndarray:
    """Boolean mask: True where an embedding looks unlike every known generator."""
    return detector.predict(embeddings) == -1
