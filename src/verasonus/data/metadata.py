"""The per-clip metadata table: one row per clip, the single source of truth for experiments.

It records where each clip came from, its label and generator, and its original sample rate
and codec, so format shortcuts can be audited. Stored as CSV at ``cfg.paths.metadata`` and
written by scripts/build_metadata.py.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath, PureWindowsPath

import pandas as pd

LABELS = ("real", "ai", "partly_ai")


@dataclass
class ClipRecord:
    clip_id: str  # unique, e.g. "fma/000002" or "fakemusiccaps/musicgen/-0Gj8-vB1q4"
    track_id: str  # the track a clip (and each of its windows) comes from
    group_id: str  # rows that must share a split: the artist where known, else the track
    source_dataset: str  # key in configs/datasets.yaml, e.g. "sonics"
    label: str  # one of LABELS
    generator: str  # e.g. "suno_v3.5", "musicgen"; "human" for real music
    path: str  # audio path relative to the data root, with forward slashes
    original_sample_rate: int
    codec: str  # original encoding, e.g. "MPEG_LAYER_III", "PCM_16"
    duration_seconds: float
    licence: str
    split: str = ""  # filled in by verasonus.data.splits
    genre: str = ""


METADATA_COLUMNS = tuple(f.name for f in fields(ClipRecord))
# Columns that must never be empty (split and genre may be)
REQUIRED_COLUMNS = (
    "clip_id",
    "track_id",
    "group_id",
    "source_dataset",
    "label",
    "generator",
    "path",
    "codec",
    "licence",
)


def records_to_frame(records: Iterable[ClipRecord]) -> pd.DataFrame:
    """Turn ClipRecords into a DataFrame with the standard column order."""
    return pd.DataFrame([asdict(r) for r in records], columns=list(METADATA_COLUMNS))


def validate_metadata(df: pd.DataFrame, data_root: str | Path | None = None) -> None:
    """Raise ValueError listing every problem in the metadata table.

    Checks that all columns are present, required values are filled in, labels are allowed
    (generator "human" exactly for real clips), clip ids are unique, sample rates and
    durations are positive, paths are relative, and each track has one group and one label.
    With ``data_root``, it also checks that every audio file exists.
    """
    missing = [c for c in METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"metadata table is missing columns: {missing}")

    problems: list[str] = []

    def check(bad: pd.Series, what: str) -> None:
        if bad.any():
            examples = df.loc[bad, "clip_id"].head(3).tolist()
            problems.append(f"{int(bad.sum())} rows {what}, e.g. {examples}")

    for column in REQUIRED_COLUMNS:
        check(df[column].isna() | (df[column].astype(str).str.strip() == ""), f"have no {column}")
    check(~df["label"].isin(LABELS), f"have a label outside {LABELS}")
    check(
        (df["label"] == "real") != (df["generator"] == "human"),
        'disagree on label "real" and generator "human"',
    )
    check(df["clip_id"].duplicated(keep=False), "share a clip_id")
    for column in ("original_sample_rate", "duration_seconds"):
        check(~(pd.to_numeric(df[column], errors="coerce") > 0), f"have no positive {column}")
    check(df["path"].astype(str).map(_is_absolute), "have an absolute path")
    for column in ("group_id", "label"):
        per_track = df.groupby("track_id")[column].nunique()
        check(
            df["track_id"].isin(per_track.index[per_track > 1]),
            f"belong to a track with more than one {column}",
        )
    if data_root is not None:
        root = Path(data_root)
        check(
            ~df["path"].astype(str).map(lambda p: (root / p).is_file()),
            f"point to files missing under {root}",
        )
    if problems:
        raise ValueError("metadata table has problems:\n- " + "\n- ".join(problems))


def build_metadata(source: str, root: Path) -> pd.DataFrame:
    """Scan one downloaded source (``<data_root>/raw/<source>``) into validated rows."""
    from verasonus.data.sources import scan  # imported here because sources imports this module

    df = records_to_frame(scan(source, root))
    if df.empty:
        raise ValueError(f"no audio found for {source!r} under {root}")
    validate_metadata(df)
    return df


def _is_absolute(path: str) -> bool:
    return PurePosixPath(path).is_absolute() or bool(PureWindowsPath(path).drive)
