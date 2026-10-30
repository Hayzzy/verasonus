"""Compute features for every window of every clip and save them to <data_root>/processed.

Needs metadata.csv with a split column (make_splits.py) and the standardised audio
(standardise_audio.py). Writes processed/<features>/features.npy, windows.csv and
columns.txt (see verasonus.features.cache).

Usage:
    python scripts/extract_features.py                     # handcrafted features, all clips
    python scripts/extract_features.py --limit 200         # quick trial on 200 random clips
    python scripts/extract_features.py --workers 8
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from verasonus.config import load_config
from verasonus.features.cache import build_feature_table, save_feature_table
from verasonus.features.handcrafted import extract_handcrafted

FEATURES = {"handcrafted": extract_handcrafted}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache per-window features.")
    parser.add_argument("--features", choices=FEATURES, default="handcrafted")
    parser.add_argument("--metadata", type=Path, help="defaults to <data_root>/metadata.csv")
    parser.add_argument("--limit", type=int, help="only this many random clips (a trial run)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    cfg = load_config()
    df = pd.read_csv(args.metadata or cfg.paths.metadata)
    if df["split"].isna().any():
        raise SystemExit("metadata.csv has clips without a split: run make_splits.py first")
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=cfg.seed)

    name = args.features if not args.limit else f"{args.features}_trial"
    started = time.perf_counter()
    try:
        windows, features, columns = build_feature_table(
            df, cfg, FEATURES[args.features], workers=args.workers
        )
    except NotImplementedError as exc:
        raise SystemExit(
            f"The {args.features} feature functions are not written yet ({exc}). "
            "Implement them in src/verasonus/features/, then run this again."
        ) from None
    out_dir = cfg.paths.processed / name
    save_feature_table(out_dir, windows, features, columns)
    print(
        f"{len(df)} clips -> {len(windows)} windows x {len(columns)} features "
        f"in {time.perf_counter() - started:.0f} s; wrote {out_dir}"
    )


if __name__ == "__main__":
    main()
