"""Saliency maps over the spectrogram [aspect 4] (Phase 2).

Grad-CAM for the CNN and integrated gradients for the Transformer, so the web app can show
which regions of the spectrogram pushed the score towards "AI".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import torch


def saliency_map(
    model: torch.nn.Module,
    spectrogram: np.ndarray,
    method: Literal["gradcam", "integrated_gradients"] = "gradcam",
    target_class: int = 1,
) -> np.ndarray:
    """Return a map with the spectrogram's shape, scaled to [0, 1]."""
    raise NotImplementedError("Phase 2")
