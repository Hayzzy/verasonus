"""Compute per-window features for every clip once and save them under data/processed.

Each clip's standardised copy (interim/) is cut into windows with the `windows` settings,
a feature function turns each window into a dict of numbers, and the result is saved as:

    processed/<name>/features.npy   float32 matrix, one row per window
    processed/<name>/windows.csv    one row per window: clip_id, window, start_seconds,
                                    real_seconds, plus split, label, generator, group_id
                                    and source_dataset from the metadata table
    processed/<name>/columns.txt    feature names, one per line, in column order

The zero padding at the end of a final window is cut off before the feature function sees
it. Only real clips would otherwise carry that padding, which would be a shortcut.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from verasonus.config import Config
from verasonus.preprocessing.audio_io import load_audio
from verasonus.preprocessing.standardise import interim_path
from verasonus.preprocessing.windowing import frame_windows

FeatureFn = Callable[[np.ndarray, int], dict[str, float]]

# Columns copied from the metadata table into windows.csv.
META_COLUMNS = ("split", "label", "generator", "group_id", "source_dataset")


def clip_windows(path: Path, cfg: Config) -> list[tuple[float, np.ndarray]]:
    """Cut one standardised clip into windows; return ``(start_seconds, audio)`` pairs.

    The audio of a padded final window is trimmed to its real samples.
    """
    audio, sample_rate = load_audio(path, target_sr=cfg.audio.target_sample_rate)
    w = cfg.windows
    windows = frame_windows(
        audio, sample_rate, w.length_seconds, w.hop_seconds, w.pad_last, w.min_last_seconds
    )
    out = []
    for i, (start, data) in enumerate(zip(windows.starts_seconds, windows.data, strict=True)):
        if i == len(windows) - 1 and windows.n_padded_samples:
            data = data[: len(data) - windows.n_padded_samples]
        out.append((float(start), data))
    return out


def clip_features(path: Path, cfg: Config, feature_fn: FeatureFn) -> list[dict[str, float]]:
    """Features for every window of one clip, each with ``start_seconds`` and ``real_seconds``."""
    rate = cfg.audio.target_sample_rate
    rows = []
    for start, audio in clip_windows(path, cfg):
        features = feature_fn(audio, rate)
        rows.append({"start_seconds": start, "real_seconds": len(audio) / rate, **features})
    return rows


def _clip_job(job: tuple[Path, Config, FeatureFn]) -> list[dict[str, float]]:
    return clip_features(*job)


def build_feature_table(
    metadata: pd.DataFrame, cfg: Config, feature_fn: FeatureFn, workers: int = 1
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Features for every window of every clip in ``metadata``.

    Returns ``(windows, features, columns)``: the per-window table, a float32 matrix with
    one row per window, and the feature names. ``feature_fn`` must be a module-level
    function when ``workers > 1`` so it can be sent to worker processes.
    """
    root = cfg.paths.data_root
    jobs = [(root / interim_path(p, cfg.audio), cfg, feature_fn) for p in metadata["path"]]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            per_clip = list(pool.map(_clip_job, jobs, chunksize=8))
    else:
        per_clip = [_clip_job(job) for job in jobs]

    columns: Sequence[str] | None = None
    window_rows, matrix = [], []
    for (_, clip), rows in zip(metadata.iterrows(), per_clip, strict=True):
        for i, row in enumerate(rows):
            names = [k for k in row if k not in ("start_seconds", "real_seconds")]
            if columns is None:
                columns = names
            elif names != list(columns):
                raise ValueError(f"{clip['clip_id']}: feature names differ between windows")
            window_rows.append(
                {
                    "clip_id": clip["clip_id"],
                    "window": i,
                    "start_seconds": row["start_seconds"],
                    "real_seconds": row["real_seconds"],
                    **{c: clip.get(c, "") for c in META_COLUMNS},
                }
            )
            matrix.append([row[c] for c in names])
    if columns is None:
        raise ValueError("no windows: check the clips and the windows settings")
    return pd.DataFrame(window_rows), np.asarray(matrix, dtype=np.float32), list(columns)


def save_feature_table(
    out_dir: Path, windows: pd.DataFrame, features: np.ndarray, columns: list[str]
) -> None:
    """Write the three files described in the module docstring."""
    if len(windows) != len(features) or features.shape[1] != len(columns):
        raise ValueError("windows, features and columns do not match in size")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", features)
    windows.to_csv(out_dir / "windows.csv", index=False)
    (out_dir / "columns.txt").write_text("\n".join(columns) + "\n", encoding="utf-8")


def load_feature_table(out_dir: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Read back what :func:`save_feature_table` wrote."""
    windows = pd.read_csv(out_dir / "windows.csv")
    features = np.load(out_dir / "features.npy")
    columns = (out_dir / "columns.txt").read_text(encoding="utf-8").split()
    return windows, features, columns
