"""The per-clip metadata table: one row per clip, the single source of truth for experiments.

It records where each clip came from, its label and generator, and its original sample rate
and codec, so format shortcuts can be audited. Stored as CSV at ``cfg.paths.metadata``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd

LABELS = ("real", "ai", "partly_ai")


@dataclass
class ClipRecord:
    clip_id: str  # unique, e.g. "sonics/fake_53858_suno_0"
    track_id: str  # groups clips/windows of the same song; splits keep it together
    source_dataset: str  # key in configs/datasets.yaml, e.g. "sonics"
    label: str  # one of LABELS
    generator: str  # e.g. "suno_v3.5", "musicgen"; "human" for real music
    path: str  # audio path relative to the data root
    original_sample_rate: int
    codec: str  # original encoding, e.g. "MPEG_LAYER_III", "PCM_16"
    duration_seconds: float
    licence: str
    split: str = ""  # filled in by verasonus.data.splits
    genre: str = ""


METADATA_COLUMNS = tuple(f.name for f in fields(ClipRecord))


def records_to_frame(records: Iterable[ClipRecord]) -> pd.DataFrame:
    """Turn ClipRecords into a DataFrame with the standard column order."""
    return pd.DataFrame([asdict(r) for r in records], columns=list(METADATA_COLUMNS))


def validate_metadata(df: pd.DataFrame) -> None:
    """Check columns, allowed labels, unique clip ids and positive durations (Phase 1)."""
    raise NotImplementedError("Phase 1")


def build_metadata(source: str, root: Path) -> pd.DataFrame:
    """Scan one downloaded source and return its rows of the metadata table (Phase 1)."""
    raise NotImplementedError("Phase 1")
