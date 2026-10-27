"""Registry of data sources (configs/datasets.yaml) and scanners for downloaded audio.

``scan`` lists the clips of a downloaded source as ClipRecords. It is implemented for FMA and
FakeMusicCaps so far (Phase 1). ``download`` is still a stub; docs/DATA.md has the steps.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from verasonus.config import REPO_ROOT, load_yaml
from verasonus.data.metadata import ClipRecord
from verasonus.preprocessing.audio_io import audio_info

DATASETS_CONFIG = REPO_ROOT / "configs" / "datasets.yaml"
AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".ogg"})

# FakeMusicCaps folder name -> generator name in configs/datasets.yaml
FAKEMUSICCAPS_FOLDERS = {
    "MusicGen_medium": "musicgen",
    "audioldm2": "audioldm2",
    "musicldm": "musicldm",
    "mustango": "mustango",
    "stable_audio_open": "stable_audio_open",
}


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
    exclude: tuple[str, ...] = ()  # file names (without extension) that scan() skips


def load_registry(path: str | Path = DATASETS_CONFIG) -> dict[str, DataSource]:
    """Read the dataset registry, keyed by source name (e.g. "sonics")."""
    registry = {}
    for key, entry in load_yaml(path).items():
        entry = {
            **entry,
            "labels": tuple(entry["labels"]),
            "generators": tuple(entry["generators"]),
            "exclude": tuple(str(name) for name in entry.get("exclude", ())),
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


def scan(key: str, root: str | Path) -> Iterator[ClipRecord]:
    """Return one ClipRecord per audio file of a downloaded source, in a stable order.

    ``root`` is the source's download folder, ``<data_root>/raw/<key>``. Record paths are
    relative to the data root, so the table works on any machine. Files listed under
    ``exclude`` in datasets.yaml are skipped; unreadable files are skipped with a warning.
    """
    registry = load_registry()
    if key not in registry:
        raise KeyError(f"unknown source {key!r}; see configs/datasets.yaml")
    if key not in _SCANNERS:
        raise NotImplementedError(f"Phase 1: no scanner for {key!r} yet")
    return _SCANNERS[key](registry[key], Path(root).resolve())


def _scan_fma(source: DataSource, root: Path) -> Iterator[ClipRecord]:
    """FMA: ``fma_<subset>/<NNN>/<track id>.mp3``, described by ``fma_metadata/tracks.csv``.

    A clip's group is its artist, so a split never puts one artist's songs in two places.
    """
    tracks_csv = root / "fma_metadata" / "tracks.csv"
    if not tracks_csv.is_file():
        raise FileNotFoundError(f"{tracks_csv} not found: download and unzip fma_metadata.zip")
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1], low_memory=False)
    artists = tracks[("artist", "id")].to_dict()
    genres = tracks[("track", "genre_top")].to_dict()
    licences = tracks[("track", "license")].to_dict()

    seen: dict[str, Path] = {}
    for path in _audio_files(root):
        name = path.stem
        if name in source.exclude:
            continue
        if name in seen:
            raise ValueError(f"FMA track {name} appears twice: {seen[name]} and {path}")
        seen[name] = path
        track = int(name) if name.isdigit() else None
        if track not in artists:
            warnings.warn(f"skipping {path}: not a track in tracks.csv", stacklevel=2)
            continue
        clip_id = f"fma/{name}"
        artist = artists[track]
        record = _record(
            path,
            root,
            clip_id=clip_id,
            track_id=clip_id,
            group_id=clip_id if pd.isna(artist) else f"fma/artist/{int(artist)}",
            source_dataset=source.key,
            label="real",
            generator="human",
            licence=_text(licences[track], "unknown"),
            genre=_text(genres[track]),
        )
        if record is not None:
            yield record


def _scan_fakemusiccaps(source: DataSource, root: Path) -> Iterator[ClipRecord]:
    """FakeMusicCaps: ``<generator folder>/<MusicCaps YouTube id>.wav``.

    All five generators were given the same 5,521 MusicCaps captions, so each file name appears
    once per generator. Each clip is its own track and group: putting a caption's five clips in
    one group would leave leave-one-generator-out nothing to split. The caption's YouTube id is
    the last part of ``clip_id``.
    """
    for path in _audio_files(root):
        folder = path.relative_to(root).parts[0]
        if folder not in FAKEMUSICCAPS_FOLDERS:
            raise ValueError(f"unexpected FakeMusicCaps folder {folder!r} under {root}")
        generator = FAKEMUSICCAPS_FOLDERS[folder]
        clip_id = f"fakemusiccaps/{generator}/{path.stem}"
        record = _record(
            path,
            root,
            clip_id=clip_id,
            track_id=clip_id,
            group_id=clip_id,
            source_dataset=source.key,
            label="ai",
            generator=generator,
            licence=source.licence,
        )
        if record is not None:
            yield record


_SCANNERS: dict[str, Callable[[DataSource, Path], Iterator[ClipRecord]]] = {
    "fma": _scan_fma,
    "fakemusiccaps": _scan_fakemusiccaps,
}


def _audio_files(root: Path) -> list[Path]:
    """Audio files under ``root`` in a stable order, skipping macOS metadata files.

    Zips made on macOS carry ``__MACOSX/`` folders of ``._*`` files, some named ``*.wav``.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found: download the source first (docs/DATA.md)")
    files = (
        p
        for p in root.rglob("*")
        if p.suffix.lower() in AUDIO_SUFFIXES
        and not any(part.startswith((".", "__MACOSX")) for part in p.relative_to(root).parts)
        and p.is_file()
    )
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def _record(path: Path, root: Path, **fields: str) -> ClipRecord | None:
    """Build a record from the file's header, or warn and return None if it cannot be read."""
    try:
        info = audio_info(path)
    except RuntimeError as exc:  # soundfile raises LibsndfileError, a RuntimeError
        warnings.warn(f"skipping unreadable file {path}: {exc}", stacklevel=3)
        return None
    return ClipRecord(
        path=path.relative_to(root.parent.parent).as_posix(),
        original_sample_rate=info.sample_rate,
        codec=info.subtype,
        duration_seconds=round(info.duration_seconds, 3),
        **fields,
    )


def _text(value: object, default: str = "") -> str:
    """A CSV cell as text, with ``default`` for an empty cell."""
    return default if pd.isna(value) else str(value)
