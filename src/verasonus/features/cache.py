"""Compute per-window features for every clip once and save them under data/processed.

Each clip's standardised copy (interim/) is cut into windows with the `windows` settings,
a feature function turns each window into a dict of numbers, and the result is saved as:

    processed/<name>/features.npy   float32 matrix, one row per window
    processed/<name>/windows.csv    one row per window: clip_id, window, start_seconds,
                                    real_seconds, plus split, label, generator, group_id
                                    and source_dataset from the metadata table
    processed/<name>/columns.txt    feature names, one per line, in column order

Spectrograms for the deep models are cached the same way, but as a 3-D array that is written
to disk as it is computed (it does not fit comfortably in memory):

    processed/<name>/spectrograms.npy   float16, (windows, n_mels, frames)
    processed/<name>/windows.csv        as above
    processed/<name>/settings.json      the spectrogram settings used

The zero padding at the end of a final window is cut off before the feature function sees
it. Only real clips would otherwise carry that padding, which would be a shortcut.

Training-time edits: with ``edit_probability`` > 0, each train clip gets one random everyday
edit (speed change and/or MP3, augmentation.transforms.random_training_edit) with that
probability. The edit is made on the raw file, which is then standardised as usual and
cropped or repeated to the unedited length, so window counts never change. The choice is
seeded by the clip id, so the handcrafted and spectrogram caches see the same edits. Val and
test clips are never edited.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from verasonus.config import Config
from verasonus.features.spectrograms import mel_settings, window_mel
from verasonus.preprocessing.audio_io import load_audio
from verasonus.preprocessing.standardise import excerpt, interim_path, standardise_audio
from verasonus.preprocessing.windowing import frame_windows

FeatureFn = Callable[[np.ndarray, int], dict[str, float]]

# Columns copied from the metadata table into windows.csv.
META_COLUMNS = ("split", "label", "generator", "group_id", "source_dataset")


Edit = tuple[Path, int, float] | None  # raw file, seed, probability; None = never edit


def edit_for(clip: pd.Series, cfg: Config, probability: float) -> Edit:
    """The edit settings for one metadata row: only train clips can be edited."""
    if probability <= 0 or clip.get("split") != "train":
        return None
    seed = zlib.crc32(str(clip["clip_id"]).encode()) ^ cfg.seed
    return cfg.paths.data_root / clip["path"], seed, probability


def load_clip(path: Path, cfg: Config, edit: Edit = None) -> np.ndarray:
    """A clip's standardised audio, with its random training edit if it gets one."""
    import soxr

    from verasonus.augmentation.transforms import random_training_edit

    rate = cfg.audio.target_sample_rate
    audio, _ = load_audio(path, target_sr=rate)
    if edit is None:
        return audio
    raw, seed, probability = edit
    rng = np.random.default_rng(seed)
    if rng.random() >= probability:
        return audio
    # Edit at twice the target rate, so a slowed clip still fills the band up to the low-pass.
    source, _ = load_audio(raw, target_sr=2 * rate)
    _, edited = random_training_edit(source, 2 * rate, rng)
    edited = soxr.resample(edited, 2 * rate, rate)
    edited = standardise_audio(excerpt(edited, rate, cfg.audio), rate, cfg.audio)
    if len(edited) >= len(audio):
        return edited[: len(audio)]
    return np.tile(edited, -(-len(audio) // len(edited)))[: len(audio)]


def clip_windows(path: Path, cfg: Config, edit: Edit = None) -> list[tuple[float, np.ndarray]]:
    """Cut one standardised clip into windows; return ``(start_seconds, audio)`` pairs.

    The audio of a padded final window is trimmed to its real samples.
    """
    return split_windows(load_clip(path, cfg, edit), cfg.audio.target_sample_rate, cfg)


def split_windows(
    audio: np.ndarray, sample_rate: int, cfg: Config
) -> list[tuple[float, np.ndarray]]:
    """Cut standardised audio into windows, as ``clip_windows`` does for a file."""
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


def clip_features(
    path: Path, cfg: Config, feature_fn: FeatureFn, edit: Edit = None
) -> list[dict[str, float]]:
    """Features for every window of one clip, each with ``start_seconds`` and ``real_seconds``."""
    rate = cfg.audio.target_sample_rate
    rows = []
    for start, audio in clip_windows(path, cfg, edit):
        features = feature_fn(audio, rate)
        rows.append({"start_seconds": start, "real_seconds": len(audio) / rate, **features})
    return rows


def _clip_job(job: tuple[Path, Config, FeatureFn, Edit]) -> list[dict[str, float]]:
    return clip_features(*job)


def build_feature_table(
    metadata: pd.DataFrame,
    cfg: Config,
    feature_fn: FeatureFn,
    workers: int = 1,
    edit_probability: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Features for every window of every clip in ``metadata``.

    Returns ``(windows, features, columns)``: the per-window table, a float32 matrix with
    one row per window, and the feature names. ``feature_fn`` must be a module-level
    function when ``workers > 1`` so it can be sent to worker processes.
    """
    root = cfg.paths.data_root
    jobs = [
        (
            root / interim_path(clip["path"], cfg.audio),
            cfg,
            feature_fn,
            edit_for(clip, cfg, edit_probability),
        )
        for _, clip in metadata.iterrows()
    ]
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


def count_windows(path: Path, cfg: Config) -> int:
    """Number of windows :func:`clip_windows` will cut from a standardised clip."""
    w = cfg.windows
    n_samples = sf.info(path).frames
    zeros = np.zeros(n_samples, dtype=np.float32)
    rate = cfg.audio.target_sample_rate
    return len(
        frame_windows(zeros, rate, w.length_seconds, w.hop_seconds, w.pad_last, w.min_last_seconds)
    )


def _spectrogram_job(
    job: tuple[Path, Config, dict[str, Any], Edit],
) -> tuple[list[tuple[float, float]], np.ndarray]:
    path, cfg, settings, edit = job
    rate = cfg.audio.target_sample_rate
    windows = clip_windows(path, cfg, edit)
    times = [(start, len(audio) / rate) for start, audio in windows]
    specs = [window_mel(audio, rate, **settings) for _, audio in windows]
    return times, np.stack(specs) if specs else np.empty((0,), dtype=np.float16)


def build_spectrogram_cache(
    metadata: pd.DataFrame,
    cfg: Config,
    out_dir: Path,
    workers: int = 1,
    edit_probability: float = 0.0,
) -> pd.DataFrame:
    """Log-mel spectrograms for every window of every clip, written straight to ``out_dir``.

    Returns the per-window table (also saved as windows.csv).
    """
    settings = mel_settings(cfg)
    root = cfg.paths.data_root
    paths = [root / interim_path(p, cfg.audio) for p in metadata["path"]]
    counts = [count_windows(path, cfg) for path in paths]
    if not sum(counts):
        raise ValueError("no windows: check the clips and the windows settings")

    out_dir.mkdir(parents=True, exist_ok=True)
    shape = (sum(counts), settings["n_mels"], settings["n_frames"])
    specs = np.lib.format.open_memmap(
        out_dir / "spectrograms.npy", mode="w+", dtype=np.float16, shape=shape
    )
    edits = [edit_for(clip, cfg, edit_probability) for _, clip in metadata.iterrows()]
    jobs = [(path, cfg, settings, edit) for path, edit in zip(paths, edits, strict=True)]
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    results = pool.map(_spectrogram_job, jobs, chunksize=8) if pool else map(_spectrogram_job, jobs)
    rows, i = [], 0
    try:
        for (_, clip), count, (times, clip_specs) in zip(
            metadata.iterrows(), counts, results, strict=True
        ):
            if len(times) != count:
                raise ValueError(f"{clip['clip_id']}: expected {count} windows, got {len(times)}")
            specs[i : i + count] = clip_specs
            i += count
            for w, (start, seconds) in enumerate(times):
                rows.append(
                    {
                        "clip_id": clip["clip_id"],
                        "window": w,
                        "start_seconds": start,
                        "real_seconds": seconds,
                        **{c: clip.get(c, "") for c in META_COLUMNS},
                    }
                )
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    specs.flush()
    del specs
    windows = pd.DataFrame(rows)
    windows.to_csv(out_dir / "windows.csv", index=False)
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return windows


def load_spectrogram_cache(
    out_dir: Path, mmap: bool = False
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Read back what :func:`build_spectrogram_cache` wrote. ``mmap`` leaves the array on disk."""
    windows = pd.read_csv(out_dir / "windows.csv")
    specs = np.load(out_dir / "spectrograms.npy", mmap_mode="r" if mmap else None)
    settings = json.loads((out_dir / "settings.json").read_text(encoding="utf-8"))
    return windows, specs, settings
