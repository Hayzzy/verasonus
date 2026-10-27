"""Lossy re-encoding for the robustness suite (Phase 1).

Streaming services and tab capture (the Phase 4 extension) deliver compressed audio, so the
detector has to survive MP3, AAC and Opus at several bitrates. Uses the ffmpeg command line,
which must be installed locally (Colab already has it).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Codec = Literal["mp3", "aac", "opus"]


def reencode(audio: np.ndarray, sample_rate: int, codec: Codec, bitrate_kbps: int) -> np.ndarray:
    """Encode mono ``audio`` with ``codec`` at ``bitrate_kbps`` and decode it back.

    Returns float32 audio at the original sample rate and length; trim or pad the encoder
    delay so windows stay aligned with the original.
    """
    raise NotImplementedError("Phase 1")
