"""Scan downloaded sources and write the per-clip metadata table (Phase 1).

Usage:
    python scripts/build_metadata.py --sources sonics fakemusiccaps fma
"""

import argparse

import pandas as pd

from verasonus.config import load_config
from verasonus.data.metadata import build_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/metadata.csv from raw sources.")
    parser.add_argument("--sources", nargs="+", required=True, help="keys from datasets.yaml")
    args = parser.parse_args()

    cfg = load_config()
    tables = [build_metadata(source, cfg.paths.raw / source) for source in args.sources]
    table = pd.concat(tables, ignore_index=True)
    cfg.paths.metadata.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.paths.metadata, index=False)
    print(f"Wrote {len(table)} rows to {cfg.paths.metadata}")


if __name__ == "__main__":
    main()
