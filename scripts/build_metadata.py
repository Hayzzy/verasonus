"""Scan downloaded sources and write the per-clip metadata table.

Usage:
    python scripts/build_metadata.py --sources fma fakemusiccaps

Scanners exist for fma and fakemusiccaps so far. The table goes to <data_root>/metadata.csv.
Rebuilding it clears the split column, so run scripts/make_splits.py afterwards.
"""

import argparse

import pandas as pd

from verasonus.config import load_config
from verasonus.data.metadata import build_metadata, validate_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata.csv from raw sources.")
    parser.add_argument("--sources", nargs="+", required=True, help="keys from datasets.yaml")
    args = parser.parse_args()

    cfg = load_config()
    tables = [build_metadata(source, cfg.paths.raw / source) for source in args.sources]
    table = pd.concat(tables, ignore_index=True)
    validate_metadata(table, data_root=cfg.paths.data_root)
    cfg.paths.metadata.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.paths.metadata, index=False)
    print(table.groupby(["source_dataset", "label", "generator"]).size().to_string())
    print(f"Wrote {len(table)} rows to {cfg.paths.metadata}")


if __name__ == "__main__":
    main()
