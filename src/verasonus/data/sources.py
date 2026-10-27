"""Registry of data sources (configs/datasets.yaml) plus per-source download/scan stubs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from verasonus.config import REPO_ROOT, load_yaml
from verasonus.data.metadata import ClipRecord

DATASETS_CONFIG = REPO_ROOT / "configs" / "datasets.yaml"


@dataclass(frozen=True)
class DataSource:
    """One entry of configs/datasets.yaml."""

    key: str
    title: str
    role: str
    labels: tuple[str, ...]
    generators: tuple[str, ...]
    size: str
    access: str
    licence: str
    url: str | None = None
    paper: str | None = None


def load_registry(path: str | Path = DATASETS_CONFIG) -> dict[str, DataSource]:
    """Read the dataset registry, keyed by source name (e.g. "sonics")."""
    registry = {}
    for key, entry in load_yaml(path).items():
        entry = {
            **entry,
            "labels": tuple(entry["labels"]),
            "generators": tuple(entry["generators"]),
        }
        registry[key] = DataSource(key=key, **entry)
    return registry


def download(key: str, dest: Path, subset: str | None = None) -> Path:
    """Download one source into ``dest`` and return the folder (Phase 1).

    Script the direct downloads (SONICS AI songs, FakeMusicCaps, FMA, MUSDB18-HQ) with
    resumable, checksum-verified transfers. For sources that need manual steps (SONICS real
    songs, Melody or Machine), print the instructions instead of scraping.
    """
    raise NotImplementedError("Phase 1")


def scan(key: str, root: Path) -> Iterator[ClipRecord]:
    """Yield one ClipRecord per audio file of a downloaded source (Phase 1)."""
    raise NotImplementedError("Phase 1")
