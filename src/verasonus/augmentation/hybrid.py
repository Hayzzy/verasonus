"""Hybrid human + AI tracks built from MUSDB18-HQ stems (Phase 1-2).

Following Rigaud et al. (2026), chosen stems (e.g. vocals) are passed through a neural audio
codec such as EnCodec, so they carry generator-like decoder artefacts, then remixed with the
untouched stems. Because we know which stems were processed, we also know which time regions
are "AI", which gives labels for the segment timeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

STEMS = ("vocals", "drums", "bass", "other")


def codec_resynthesise(
    stem: np.ndarray,
    sample_rate: int,
    model_name: str = "facebook/encodec_24khz",
    bandwidth_kbps: float = 6.0,
) -> np.ndarray:
    """Encode and decode one stem with a neural audio codec (EnCodec via transformers)."""
    raise NotImplementedError("Phase 2")


def make_hybrid(
    stems: Mapping[str, np.ndarray],
    sample_rate: int,
    ai_stems: Sequence[str],
    **codec_kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remix ``stems`` with ``ai_stems`` codec-resynthesised.

    Returns the mixture and label info: which stems are AI and, per window, the share of
    energy that comes from AI stems.
    """
    raise NotImplementedError("Phase 2")
