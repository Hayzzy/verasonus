"""Handcrafted features for the classical baselines [aspects 2 and 4] (Phase 1).

One fixed-length vector per window, so SVM, random forest and gradient boosting train in
minutes and can be explained with feature importance.
"""

from __future__ import annotations

import numpy as np


def mfcc_stats(audio: np.ndarray, sample_rate: int, n_mfcc: int = 20) -> dict[str, float]:
    """Mean and standard deviation of MFCCs and their deltas."""
    raise NotImplementedError("Phase 1")


def spectral_stats(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Spectral centroid, bandwidth, roll-off, flatness, contrast and zero-crossing rate."""
    raise NotImplementedError("Phase 1")


def high_band_energy(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 6000.0) -> float:
    """Share of energy above ``cutoff_hz``. Useful, but check it is not a bandwidth shortcut."""
    raise NotImplementedError("Phase 1")


def decoder_artefact_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Periodic spectral peaks left by the upsampling layers of neural decoders.

    Afchar et al. (2024) show these artefacts make detection "easy" in the lab, and that
    simple edits can remove them, so report robustness for these features separately.
    """
    raise NotImplementedError("Phase 1")


def extract_handcrafted(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """All handcrafted features for one window, with a stable key order."""
    raise NotImplementedError("Phase 1")
