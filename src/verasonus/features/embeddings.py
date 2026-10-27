"""Pretrained audio embeddings for linear and MLP probes [aspect 2] (Phase 2).

* MERT, m-a-p/MERT-v1-95M: expects 24 kHz audio and loads with trust_remote_code=True. If its
  remote code breaks on transformers 5.x, pin transformers below 5 in requirements/ml.txt.
* CLAP, laion/larger_clap_music: expects 48 kHz audio.

Compute embeddings once per window on Colab and cache them under data/processed, so probes
train in seconds.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

EmbeddingModel = Literal["m-a-p/MERT-v1-95M", "laion/larger_clap_music"]


class EmbeddingExtractor:
    """Wraps one pretrained model; resamples windows to the model's rate internally."""

    def __init__(self, model_name: EmbeddingModel, device: str = "cpu") -> None:
        raise NotImplementedError("Phase 2")

    def embed(self, windows: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return ``(n_windows, dim)`` embeddings (for MERT, a layer chosen on validation)."""
        raise NotImplementedError("Phase 2")
