"""Load the YAML configuration into typed, read-only settings (implemented in Phase 0).

Example:
    >>> from verasonus.config import load_config
    >>> cfg = load_config(overrides={"audio.target_sample_rate": 32000})
    >>> cfg.audio.target_sample_rate
    32000

Precedence, lowest to highest: the YAML file, the VERASONUS_DATA_ROOT environment variable
(also read from a `.env` file at the repository root), then explicit ``overrides``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# src/verasonus/config.py -> repository root. The package is installed in editable mode
# (pip install -e .), so this always points at the checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
DATA_ROOT_ENV = "VERASONUS_DATA_ROOT"


@dataclass(frozen=True)
class PathsConfig:
    """Where data, checkpoints and report outputs live (absolute paths)."""

    data_root: Path
    checkpoints: Path
    reports: Path

    @property
    def raw(self) -> Path:
        """Downloaded datasets, untouched."""
        return self.data_root / "raw"

    @property
    def interim(self) -> Path:
        """Standardised audio: one format and one sample rate."""
        return self.data_root / "interim"

    @property
    def processed(self) -> Path:
        """Features and embeddings ready for training."""
        return self.data_root / "processed"

    @property
    def metadata(self) -> Path:
        """The per-clip metadata table."""
        return self.data_root / "metadata.csv"


@dataclass(frozen=True)
class AudioConfig:
    target_sample_rate: int = 16_000
    mono: bool = True
    storage_format: str = "FLAC"
    storage_subtype: str = "PCM_16"
    shared_codec: dict[str, Any] | None = None


@dataclass(frozen=True)
class WindowConfig:
    length_seconds: float = 10.0
    hop_seconds: float = 5.0
    pad_last: bool = True
    min_last_seconds: float = 8.0


@dataclass(frozen=True)
class SplitConfig:
    ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    group_column: str = "group_id"
    stratify_columns: tuple[str, ...] = ("label",)
    generator_column: str = "generator"
    holdout_generators: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    """All settings. ``raw`` keeps the merged YAML for sections without a typed class yet."""

    seed: int
    paths: PathsConfig
    audio: AudioConfig
    windows: WindowConfig
    splits: SplitConfig
    raw: dict[str, Any]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict (an empty file gives ``{}``)."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    path: str | Path = DEFAULT_CONFIG, overrides: Mapping[str, Any] | None = None
) -> Config:
    """Load ``path``, apply overrides such as ``{"windows.hop_seconds": 2.5}``, and validate.

    Unknown keys raise ``TypeError`` so typos in the YAML are caught early.
    """
    load_dotenv(REPO_ROOT / ".env", override=False)
    data = load_yaml(path)
    if env_root := os.environ.get(DATA_ROOT_ENV):
        data.setdefault("paths", {})["data_root"] = env_root
    for dotted_key, value in (overrides or {}).items():
        _set_dotted(data, dotted_key, value)

    paths = {key: _resolve(value) for key, value in data.get("paths", {}).items()}
    splits = dict(data.get("splits", {}))
    for key in ("stratify_columns", "holdout_generators"):
        if key in splits:
            splits[key] = tuple(splits[key] or ())

    cfg = Config(
        seed=int(data.get("seed", 42)),
        paths=PathsConfig(**paths),
        audio=AudioConfig(**data.get("audio", {})),
        windows=WindowConfig(**data.get("windows", {})),
        splits=SplitConfig(**splits),
        raw=data,
    )
    _validate(cfg)
    return cfg


def _set_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    *parents, leaf = dotted_key.split(".")
    node = data
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _validate(cfg: Config) -> None:
    if cfg.audio.target_sample_rate <= 0:
        raise ValueError("audio.target_sample_rate must be positive")
    windows = cfg.windows
    if windows.length_seconds <= 0 or windows.hop_seconds <= 0:
        raise ValueError("windows.length_seconds and windows.hop_seconds must be positive")
    if not 0 <= windows.min_last_seconds <= windows.length_seconds:
        raise ValueError("windows.min_last_seconds must be between 0 and length_seconds")
    ratios = cfg.splits.ratios
    if any(r < 0 for r in ratios.values()) or abs(sum(ratios.values()) - 1) > 1e-6:
        raise ValueError(f"splits.ratios must be non-negative and sum to 1, got {ratios}")
