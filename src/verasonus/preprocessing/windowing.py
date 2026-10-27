"""Cut audio into fixed-length, overlapping windows (implemented in Phase 0).

The models score each window; window scores are later combined into a track score and the
segment timeline (verasonus.ensemble.aggregation). Start times are kept so every score maps
back to a position in the track.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Windows:
    """Windows cut from one track.

    Attributes:
        data: float32 array ``(n_windows, window_samples)``.
        starts_seconds: start time of each window in the original track.
        length_seconds: window length (the final window may end in zero padding).
        sample_rate: sample rate of ``data``.
        n_padded_samples: zeros appended to the final window (0 if none).
    """

    data: np.ndarray
    starts_seconds: np.ndarray
    length_seconds: float
    sample_rate: int
    n_padded_samples: int = 0

    def __len__(self) -> int:
        return len(self.data)

    @property
    def ends_seconds(self) -> np.ndarray:
        return self.starts_seconds + self.length_seconds


def frame_windows(
    audio: np.ndarray,
    sample_rate: int,
    length_seconds: float,
    hop_seconds: float,
    pad_last: bool = True,
    min_last_seconds: float = 0.0,
) -> Windows:
    """Cut mono ``audio`` into windows of ``length_seconds`` every ``hop_seconds``.

    Full windows are taken along the hop grid. If audio remains after the last full window
    and ``pad_last`` is true, one more window is added at the next grid position and padded
    with zeros, provided it holds at least ``min_last_seconds`` of real audio. A clip shorter
    than one window therefore becomes a single padded window (or none).
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio shaped (samples,), got shape {audio.shape}")
    if sample_rate <= 0 or length_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("sample_rate, length_seconds and hop_seconds must be positive")
    if not 0 <= min_last_seconds <= length_seconds:
        raise ValueError("min_last_seconds must be between 0 and length_seconds")

    win = round(length_seconds * sample_rate)
    hop = round(hop_seconds * sample_rate)
    n = audio.shape[0]

    starts = list(range(0, n - win + 1, hop))
    covered = starts[-1] + win if starts else 0
    n_padded = 0
    if pad_last and covered < n:
        tail_start = starts[-1] + hop if starts else 0
        remaining = n - tail_start
        if remaining >= max(1, round(min_last_seconds * sample_rate)):
            starts.append(tail_start)
            n_padded = win - remaining

    data = np.zeros((len(starts), win), dtype=np.float32)
    for i, start in enumerate(starts):
        chunk = audio[start : start + win]
        data[i, : chunk.shape[0]] = chunk
    return Windows(
        data=data,
        starts_seconds=np.asarray(starts, dtype=np.float64) / sample_rate,
        length_seconds=win / sample_rate,
        sample_rate=sample_rate,
        n_padded_samples=n_padded,
    )
