"""Source separation with Demucs [aspect 4] (Phase 2).

Scoring vocals and accompaniment separately helps with hybrid tracks, where only some stems
are AI-generated (for example AI vocals over a produced beat). Demucs (Meta, MIT licence) is
heavy: on a GPU a song takes a few seconds, on a CPU about half a minute.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np


@cache
def _demucs(name: str) -> Any:
    from demucs.pretrained import get_model

    return get_model(name).eval()


def separate_stems(
    audio: np.ndarray, sample_rate: int, model: str = "htdemucs", device: str = "cpu"
) -> dict[str, np.ndarray]:
    """Split a mix into {"vocals", "drums", "bass", "other"} stems.

    ``audio`` is mono ``(samples,)`` or ``(channels, samples)``. Stems come back mono at
    ``sample_rate``. Demucs runs at its own rate (44.1 kHz, stereo), so the audio is
    resampled on the way in and out.
    """
    import soxr
    import torch
    from demucs.apply import apply_model

    separator = _demucs(model)
    rate = separator.samplerate
    audio = np.atleast_2d(np.asarray(audio, dtype=np.float32))
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    audio = audio[:2]
    if sample_rate != rate:
        audio = soxr.resample(audio.T, sample_rate, rate).T
    x = torch.from_numpy(np.ascontiguousarray(audio))
    mono = x.mean(0)
    mean, std = mono.mean(), mono.std().clamp_min(1e-8)  # Demucs expects normalised input
    with torch.inference_mode():
        out = apply_model(separator, ((x - mean) / std)[None], device=device, progress=False)[0]
    out = out * std + mean
    stems = {}
    for name, stem in zip(separator.sources, out, strict=True):
        mono_stem = stem.mean(0).cpu().numpy()
        if sample_rate != rate:
            mono_stem = soxr.resample(mono_stem, rate, sample_rate)
        stems[name] = mono_stem.astype(np.float32)
    return stems


def vocals_and_instrumental(
    audio: np.ndarray, sample_rate: int, device: str = "cpu"
) -> dict[str, np.ndarray]:
    """Two stems: the vocals, and everything else summed (drums, bass, other)."""
    stems = separate_stems(audio, sample_rate, device=device)
    vocals = stems.pop("vocals")
    return {"vocals": vocals, "instrumental": np.sum(list(stems.values()), axis=0)}
