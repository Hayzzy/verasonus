"""Scan downloaded sources and write the per-clip metadata table.

Usage:
    python scripts/build_metadata.py --sources fma fakemusiccaps
    python scripts/build_metadata.py --sources fma sonics --append

Scanners exist for fma, fakemusiccaps and sonics. The table goes to <data_root>/metadata.csv.
Rebuilding it clears the split column, so run scripts/make_splits.py afterwards.

--append keeps every existing row and its split, and adds only clips not yet in the table
(e.g. fma_medium after fma_small, or a new source). A new clip whose group (an FMA artist)
already has a split joins that split; new groups are split as make_splits.py would. Clips
whose source fixes a split (SONICS) keep it.
"""

import argparse

import pandas as pd

from verasonus.config import load_config
from verasonus.data.metadata import build_metadata, validate_metadata
from verasonus.data.splits import group_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata.csv from raw sources.")
    parser.add_argument("--sources", nargs="+", required=True, help="keys from datasets.yaml")
    parser.add_argument("--append", action="store_true", help="add new clips, keep splits")
    args = parser.parse_args()

    cfg = load_config()
    tables = [build_metadata(source, cfg.paths.raw / source) for source in args.sources]
    table = pd.concat(tables, ignore_index=True)

    if args.append:
        old = pd.read_csv(cfg.paths.metadata, keep_default_na=False)
        new = table[~table["clip_id"].isin(old["clip_id"])].copy()
        s = cfg.splits
        known = old.drop_duplicates(s.group_column).set_index(s.group_column)["split"]
        given = new["split"].fillna("") != ""  # e.g. SONICS songs keep their own split
        new["split"] = new["split"].where(given, new[s.group_column].map(known))
        fresh = new["split"].isna()
        if fresh.any():
            new.loc[fresh, "split"] = group_split(
                new[fresh],
                s.ratios,
                group_column=s.group_column,
                stratify_columns=s.stratify_columns,
                seed=cfg.seed,
            )
        print(f"adding {len(new)} clips ({int((~fresh).sum())} join an existing group's split)")
        table = pd.concat([old, new], ignore_index=True)

    validate_metadata(table, data_root=cfg.paths.data_root)
    cfg.paths.metadata.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.paths.metadata, index=False)
    print(table.groupby(["source_dataset", "label", "generator"]).size().to_string())
    if args.append:
        print(table.groupby(["split", "source_dataset"]).size().unstack(fill_value=0))
    print(f"Wrote {len(table)} rows to {cfg.paths.metadata}")


if __name__ == "__main__":
    main()
