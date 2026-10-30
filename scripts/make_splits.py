"""Add a "split" column to the metadata table (works now; needs data/metadata.csv).

Usage:
    python scripts/make_splits.py                     # split by group_id, stratified by label
    python scripts/make_splits.py --holdout musicgen  # also hold a generator out as test_unseen
"""

import argparse
from pathlib import Path

import pandas as pd

from verasonus.config import load_config
from verasonus.data.splits import generator_holdout_split, group_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign train/val/test splits by group.")
    parser.add_argument("--metadata", type=Path, help="defaults to <data_root>/metadata.csv")
    parser.add_argument("--holdout", nargs="*", help="generators to hold out as test_unseen")
    parser.add_argument("--out", type=Path, help="defaults to overwriting --metadata")
    args = parser.parse_args()

    cfg = load_config()
    path = args.metadata or cfg.paths.metadata
    df = pd.read_csv(path)
    s = cfg.splits
    holdout = args.holdout if args.holdout is not None else list(s.holdout_generators)
    common = {
        "ratios": s.ratios,
        "group_column": s.group_column,
        "stratify_columns": s.stratify_columns,
        "seed": cfg.seed,
    }
    if holdout:
        df["split"] = generator_holdout_split(
            df, holdout, generator_column=s.generator_column, **common
        )
    else:
        df["split"] = group_split(df, **common)

    out = args.out or path
    df.to_csv(out, index=False)
    tracks = df.drop_duplicates(s.group_column)
    print(tracks.groupby(["split", "label"]).size().unstack(fill_value=0))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
