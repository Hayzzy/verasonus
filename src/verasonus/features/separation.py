"""Optional source separation with Demucs [aspect 4] (Phase 2).

Scoring vocals and accompaniment separately helps with hybrid tracks, where only some stems
are AI-generated. Demucs is heavy: run it on Colab and cache the stems.
"""

from __future__ import annotations

import numpy as np


def separate_stems(
    audio: np.ndarray, sample_rate: int, model: str = "htdemucs", device: str = "cpu"
) -> dict[str, np.ndarray]:
    """Split a mix into {"vocals", "drums", "bass", "other"} stems."""
    raise NotImplementedError("Phase 2")
