"""Compute features for every window of every clip and save them to <data_root>/processed.

Needs metadata.csv with a split column (make_splits.py) and the standardised audio
(standardise_audio.py). Writes processed/<features>/features.npy, windows.csv and
columns.txt, or for mel spectrograms.npy, windows.csv and settings.json
(see verasonus.features.cache).

Usage:
    python scripts/extract_features.py                     # handcrafted features, all clips
    python scripts/extract_features.py --limit 200         # quick trial on 200 random clips
    python scripts/extract_features.py --workers 8
    python scripts/extract_features.py --features mel      # log-mel input for the CNN (~7 GB)
    python scripts/extract_features.py --features mel --edit --name mel_v2
                                        # train clips get random speed/MP3 edits
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from verasonus.config import REPO_ROOT, load_config, load_yaml
from verasonus.features.cache import (
    build_feature_table,
    build_spectrogram_cache,
    save_feature_table,
)
from verasonus.features.handcrafted import extract_handcrafted

FEATURES = {"handcrafted": extract_handcrafted}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache per-window features.")
    parser.add_argument("--features", choices=[*FEATURES, "mel"], default="handcrafted")
    parser.add_argument("--metadata", type=Path, help="defaults to <data_root>/metadata.csv")
    parser.add_argument("--limit", type=int, help="only this many random clips (a trial run)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--edit",
        action="store_true",
        help="give train clips random speed/MP3 edits (configs/augment.yaml training)",
    )
    parser.add_argument("--name", help="output folder under processed/ (default: --features)")
    args = parser.parse_args()

    cfg = load_config()
    df = pd.read_csv(args.metadata or cfg.paths.metadata)
    if df["split"].isna().any():
        raise SystemExit("metadata.csv has clips without a split: run make_splits.py first")
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=cfg.seed)

    name = args.name or args.features
    if args.limit:
        name += "_trial"
    edit_probability = 0.0
    if args.edit:
        edit_probability = float(
            load_yaml(REPO_ROOT / "configs" / "augment.yaml")["training"]["probability"]
        )
    started = time.perf_counter()
    out_dir = cfg.paths.processed / name
    if args.features == "mel":
        windows = build_spectrogram_cache(
            df, cfg, out_dir, workers=args.workers, edit_probability=edit_probability
        )
        print(
            f"{len(df)} clips -> {len(windows)} spectrograms "
            f"in {time.perf_counter() - started:.0f} s; wrote {out_dir}"
        )
        return
    try:
        windows, features, columns = build_feature_table(
            df,
            cfg,
            FEATURES[args.features],
            workers=args.workers,
            edit_probability=edit_probability,
        )
    except NotImplementedError as exc:
        raise SystemExit(
            f"The {args.features} feature functions are not written yet ({exc}). "
            "Implement them in src/verasonus/features/, then run this again."
        ) from None
    save_feature_table(out_dir, windows, features, columns)
    print(
        f"{len(df)} clips -> {len(windows)} windows x {len(columns)} features "
        f"in {time.perf_counter() - started:.0f} s; wrote {out_dir}"
    )


if __name__ == "__main__":
    main()
