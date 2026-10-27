"""Train/val/test splits that keep every window of a track together (implemented in Phase 0).

* group_split: random split by track, stratified by label (and genre, when given).
* generator_holdout_split: whole generators go to "test_unseen"; the rest is group-split.
* leave_one_generator_out: one generator_holdout_split per AI generator.

Each function takes the metadata table (one row per clip or window) and returns a Series of
split names aligned with its index, so ``df["split"] = group_split(df)`` just works.

Unseen-generator evaluation uses the "test_unseen" rows (AI audio from held-out generators)
together with the real rows of "test", so both classes are present.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_RATIOS: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}
UNSEEN = "test_unseen"


def group_split(
    df: pd.DataFrame,
    ratios: Mapping[str, float] | None = None,
    *,
    group_column: str = "track_id",
    stratify_columns: Sequence[str] = ("label",),
    seed: int = 42,
) -> pd.Series:
    """Assign each row a split so that all rows of a track (``group_column``) share it.

    Tracks are shuffled with ``seed`` and cut by ``ratios`` separately within each stratum,
    i.e. each combination of ``stratify_columns`` taken from the track's first row, so every
    split keeps roughly the same label (and genre) balance.
    """
    ratios = _check_ratios(ratios)
    stratify_columns = list(stratify_columns)
    _check_columns(df, [group_column, *stratify_columns])
    if df[group_column].isna().any():
        raise ValueError(f"column {group_column!r} has missing values")

    tracks = df.drop_duplicates(group_column).set_index(group_column).sort_index()
    if stratify_columns:
        strata = tracks[stratify_columns].astype(str).agg("|".join, axis=1)
    else:
        strata = pd.Series("all", index=tracks.index)

    rng = np.random.default_rng(seed)
    track_split: dict[Any, str] = {}
    for stratum in sorted(strata.unique()):
        members = rng.permutation(strata.index[(strata == stratum).to_numpy()].to_numpy())
        bounds = np.cumsum([0, *_allocate(len(members), ratios)])
        for name, start, stop in zip(ratios, bounds[:-1], bounds[1:], strict=True):
            track_split.update(dict.fromkeys(members[start:stop], name))
    return df[group_column].map(track_split).rename("split")


def generator_holdout_split(
    df: pd.DataFrame,
    holdout_generators: Sequence[str],
    *,
    generator_column: str = "generator",
    ratios: Mapping[str, float] | None = None,
    group_column: str = "track_id",
    stratify_columns: Sequence[str] = ("label",),
    seed: int = 42,
) -> pd.Series:
    """Put every row of ``holdout_generators`` in "test_unseen" and group-split the rest.

    Held-out generators never appear in train, val or test, which is what the
    unseen-generator protocol needs.
    """
    holdout = set(holdout_generators)
    if not holdout:
        raise ValueError("holdout_generators is empty")
    _check_columns(df, [generator_column, group_column])
    unknown = holdout - set(df[generator_column].dropna())
    if unknown:
        raise ValueError(f"generators not in the table: {sorted(unknown)}")
    held = df[generator_column].isin(holdout)
    mixed = set(df.loc[held, group_column]) & set(df.loc[~held, group_column])
    if mixed:
        raise ValueError(f"tracks mix held-out and other generators: {sorted(mixed)[:5]}")

    seen = group_split(
        df[~held],
        ratios,
        group_column=group_column,
        stratify_columns=stratify_columns,
        seed=seed,
    )
    return seen.reindex(df.index, fill_value=UNSEEN)


def leave_one_generator_out(
    df: pd.DataFrame,
    *,
    generator_column: str = "generator",
    label_column: str = "label",
    ai_label: str = "ai",
    **split_kwargs: Any,
) -> Iterator[tuple[str, pd.Series]]:
    """Yield ``(generator, splits)`` once per AI generator, holding that generator out."""
    _check_columns(df, [generator_column, label_column])
    generators = sorted(df.loc[df[label_column] == ai_label, generator_column].dropna().unique())
    for generator in generators:
        splits = generator_holdout_split(
            df, [generator], generator_column=generator_column, **split_kwargs
        )
        yield generator, splits


def _allocate(n: int, ratios: Mapping[str, float]) -> list[int]:
    """Split ``n`` items into counts proportional to ``ratios`` (the counts sum to ``n``)."""
    edges = np.round(np.cumsum(list(ratios.values())) * n).astype(int)
    edges[-1] = n
    return np.diff(edges, prepend=0).tolist()


def _check_ratios(ratios: Mapping[str, float] | None) -> dict[str, float]:
    ratios = dict(DEFAULT_RATIOS if ratios is None else ratios)
    if not ratios or any(r < 0 for r in ratios.values()) or abs(sum(ratios.values()) - 1) > 1e-6:
        raise ValueError(f"ratios must be non-negative and sum to 1, got {ratios}")
    return ratios


def _check_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"metadata table is missing columns: {missing}")
