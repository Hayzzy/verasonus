"""Simulated phone-speaker playback and re-recording (Phase 1).

Approximates someone playing a track on a phone and recording it with another device: a
band-limited speaker response, optional room reverb, then microphone noise. Matters for the
PWA's microphone recording mode.
"""

from __future__ import annotations

import numpy as np


def simulate_phone_playback(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    speaker_band_hz: tuple[float, float] = (300.0, 8000.0),
    room_reverb: bool = True,
    mic_noise_snr_db: float = 35.0,
) -> np.ndarray:
    """Return ``audio`` as if played through a phone speaker and re-recorded."""
    raise NotImplementedError("Phase 1")
