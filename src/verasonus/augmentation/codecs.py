"""Lossy re-encoding for the robustness suite and training augmentation (Phase 1).

Streaming services and tab capture (the Phase 4 extension) deliver compressed audio, so the
detector has to survive MP3, AAC and Opus at several bitrates. MP3 is encoded in memory with
soundfile (libsndfile 1.1+ includes LAME), so it needs no ffmpeg. AAC and Opus still need
ffmpeg and are not implemented yet.
"""

from __future__ import annotations

import io
from typing import Literal

import numpy as np
import soundfile as sf

Codec = Literal["mp3", "aac", "opus"]


def _mp3_level(sample_rate: int, bitrate_kbps: int) -> float:
    """soundfile's MP3 compression level (0 = best) for roughly ``bitrate_kbps``.

    Constant-bitrate MP3 spans 32-320 kbps at 32 kHz and above, and 8-160 kbps below.
    """
    low, high = (32, 320) if sample_rate >= 32_000 else (8, 160)
    return float(np.clip((high - bitrate_kbps) / (high - low), 0.0, 0.95))


def reencode(audio: np.ndarray, sample_rate: int, codec: Codec, bitrate_kbps: int) -> np.ndarray:
    """Encode mono ``audio`` with ``codec`` at ``bitrate_kbps`` and decode it back.

    Returns float32 audio at the original sample rate and length; the encoder delay is
    trimmed from the start so windows stay aligned with the original.
    """
    if codec != "mp3":
        raise NotImplementedError(f"{codec} needs ffmpeg; only mp3 works so far")
    audio = np.asarray(audio, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.clip(audio, -1, 1),
        sample_rate,
        format="MP3",
        subtype="MPEG_LAYER_III",
        compression_level=_mp3_level(sample_rate, bitrate_kbps),
        bitrate_mode="CONSTANT",
    )
    buffer.seek(0)
    decoded, _ = sf.read(buffer, dtype="float32")
    if decoded.ndim > 1:
        decoded = decoded.mean(axis=1)
    # Align: find the encoder delay by cross-correlating the first second.
    n = min(len(audio), sample_rate)
    lag_range = min(4096, len(decoded) - n) if len(decoded) > n else 0
    if lag_range > 0 and np.any(audio[:n]):
        corr = [float(np.dot(audio[:n], decoded[lag : lag + n])) for lag in range(lag_range)]
        delay = int(np.argmax(corr))
    else:
        delay = 0
    out = decoded[delay : delay + len(audio)]
    return np.pad(out, (0, len(audio) - len(out))).astype(np.float32)
