"""End-to-end analysis used by the API (Phase 3).

analyze_file will:
    1. load and resample the audio, then cut windows (preprocessing.audio_io, .windowing)
    2. extract features and embeddings (features.*)
    3. score windows with each model and fuse them (models.*, ensemble.fusion)
    4. aggregate into a calibrated track score and timeline (ensemble.aggregation, .calibration)
    5. guess the generator or "unknown" (models.unsupervised) and explain (explain.*)

Target: a 30-second clip analysed in a few seconds on CPU (export models to ONNX where it
helps).
"""

from __future__ import annotations

from pathlib import Path

from verasonus.api.schemas import AnalysisResult
from verasonus.config import Config


def analyze_file(path: Path, config: Config | None = None) -> AnalysisResult:
    """Run the full pipeline on one audio file."""
    raise NotImplementedError("Phase 3")
