"""Report-ready figures and tables saved under reports/ (Phase 1 onwards).

Figures go to reports/figures/<name>.png (300 dpi) and .pdf; tables go to
reports/tables/<name>.csv, so they drop straight into the final report and slides.
Use the slide palette for plots: #FF7A00, #D56A46, #F7A760 on dark or white backgrounds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def save_figure(fig: Figure, name: str, reports_dir: Path | None = None) -> list[Path]:
    """Save ``fig`` as PNG and PDF; defaults to cfg.paths.reports / "figures"."""
    raise NotImplementedError("Phase 1")


def save_table(df: pd.DataFrame, name: str, reports_dir: Path | None = None) -> Path:
    """Save ``df`` as CSV; defaults to cfg.paths.reports / "tables"."""
    raise NotImplementedError("Phase 1")


def plot_roc(curves: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> Figure:
    """ROC curves, one per model: {name: (false_positive_rates, true_positive_rates)}."""
    raise NotImplementedError("Phase 1")


def plot_confusion(matrix: np.ndarray, labels: Sequence[str]) -> Figure:
    """Annotated confusion matrix."""
    raise NotImplementedError("Phase 1")


def plot_reliability(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Figure:
    """Reliability diagram for calibration."""
    raise NotImplementedError("Phase 2")
