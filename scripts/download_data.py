"""Download a data source into <data_root>/raw/<source> (downloads arrive in Phase 1).

Usage:
    python scripts/download_data.py --list
    python scripts/download_data.py --source fakemusiccaps
"""

import argparse

from verasonus.config import load_config
from verasonus.data.sources import download, load_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a VeraSonus data source.")
    parser.add_argument("--source", help="key from configs/datasets.yaml, e.g. sonics")
    parser.add_argument("--subset", default=None, help="optional subset, e.g. fma_small")
    parser.add_argument("--list", action="store_true", help="list sources and licences, then exit")
    args = parser.parse_args()

    registry = load_registry()
    if args.list or not args.source:
        for key, source in registry.items():
            print(f"{key:18} {source.role:42} {source.licence}")
        return
    if args.source not in registry:
        parser.error(f"unknown source {args.source!r}; run with --list")

    cfg = load_config()
    folder = download(args.source, cfg.paths.raw / args.source, subset=args.subset)
    print(f"Downloaded {args.source} to {folder}")


if __name__ == "__main__":
    main()
