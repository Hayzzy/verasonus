"""Write a standardised copy of every clip in the metadata table to <data_root>/interim.

Every clip, real or AI, is decoded, down-mixed, resampled and saved the same way (see
verasonus.preprocessing.standardise and the `audio` section of configs/default.yaml).
Copies mirror raw/: raw/fma/fma_small/000/000002.mp3 -> interim/fma/fma_small/000/000002.flac.
Existing copies are skipped, so an interrupted run can simply be started again.

Usage:
    python scripts/standardise_audio.py                # all clips, all CPU cores
    python scripts/standardise_audio.py --workers 8
    python scripts/standardise_audio.py --overwrite    # redo copies, e.g. after a config change
    python scripts/standardise_audio.py --audit 200    # also compare bandwidth per generator
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from verasonus.config import AudioConfig, load_config
from verasonus.preprocessing.standardise import (
    effective_bandwidth_hz,
    interim_path,
    standardise_file,
)


def _work(job: tuple[Path, Path, AudioConfig]) -> str | None:
    src, dst, audio_cfg = job
    try:
        standardise_file(src, dst, audio_cfg)
    except Exception as exc:  # report and carry on; one bad file should not stop the run
        dst.unlink(missing_ok=True)
        return f"{src}: {exc}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardise every clip in metadata.csv.")
    parser.add_argument("--metadata", type=Path, help="defaults to <data_root>/metadata.csv")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--overwrite", action="store_true", help="redo existing copies")
    parser.add_argument(
        "--audit",
        type=int,
        default=0,
        metavar="N",
        help="measure bandwidth of N standardised clips per generator",
    )
    args = parser.parse_args()

    cfg = load_config()
    root = cfg.paths.data_root
    df = pd.read_csv(args.metadata or cfg.paths.metadata)
    dst_paths = df["path"].map(lambda p: interim_path(p, cfg.audio))

    jobs = [
        (root / src, root / dst, cfg.audio)
        for src, dst in zip(df["path"], dst_paths, strict=True)
        if args.overwrite or not (root / dst).exists()
    ]
    print(f"{len(df)} clips, {len(df) - len(jobs)} already standardised, {len(jobs)} to do")
    errors = []
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for error in tqdm(pool.map(_work, jobs, chunksize=16), total=len(jobs)):
                if error:
                    errors.append(error)
    for error in errors:
        print("FAILED", error)
    print(f"Done: {len(jobs) - len(errors)} written, {len(errors)} failed, in {cfg.paths.interim}")

    if args.audit:
        sample = (
            df.assign(interim=dst_paths)
            .sample(frac=1, random_state=cfg.seed)
            .groupby("generator")
            .head(args.audit)
        )
        sample["bandwidth_hz"] = [effective_bandwidth_hz(root / p) for p in sample["interim"]]
        print("Effective bandwidth (Hz) of standardised clips by generator:")
        print(sample.groupby("generator")["bandwidth_hz"].describe()[["50%", "min", "max"]])

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
