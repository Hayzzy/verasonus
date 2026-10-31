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
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

PALETTE = ["#FF7A00", "#D56A46", "#F7A760", "#6A6A6A", "#B3B3B3", "#30180F", "#FFD1A6", "#7A2E12"]


def _default_dir(sub: str, reports_dir: Path | None) -> Path:
    if reports_dir is None:
        from verasonus.config import load_config

        reports_dir = load_config().paths.reports
    out = reports_dir / sub
    out.mkdir(parents=True, exist_ok=True)
    return out


def new_figure(figsize: tuple[float, float] = (5, 4)) -> tuple[Figure, Axes]:
    """A figure and axes drawn off-screen, so scripts run without a display."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=figsize, layout="constrained")


def save_figure(fig: Figure, name: str, reports_dir: Path | None = None) -> list[Path]:
    """Save ``fig`` as PNG and PDF; defaults to cfg.paths.reports / "figures"."""
    import matplotlib.pyplot as plt

    out = _default_dir("figures", reports_dir)
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    return paths


def save_table(df: pd.DataFrame, name: str, reports_dir: Path | None = None) -> Path:
    """Save ``df`` as CSV; defaults to cfg.paths.reports / "tables"."""
    path = _default_dir("tables", reports_dir) / f"{name}.csv"
    df.to_csv(path)
    return path


def plot_roc(curves: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> Figure:
    """ROC curves, one per model: {name: (false_positive_rates, true_positive_rates)}."""
    fig, ax = new_figure()
    for (name, (fpr, tpr)), colour in zip(curves.items(), PALETTE, strict=False):
        ax.plot(fpr, tpr, label=name, color=colour, linewidth=2)
    ax.plot([0, 1], [0, 1], color="#B3B3B3", linestyle="--", linewidth=1)
    ax.set(xlabel="False-positive rate (real flagged as AI)", ylabel="True-positive rate")
    ax.legend(loc="lower right")
    return fig


def plot_confusion(matrix: np.ndarray, labels: Sequence[str]) -> Figure:
    """Annotated confusion matrix."""
    fig, ax = new_figure((4.5, 4))
    ax.imshow(matrix, cmap="Oranges")
    ticks = range(len(labels))
    ax.set(xticks=ticks, yticks=ticks, xlabel="Predicted", ylabel="True")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    high = matrix.max() / 2
    for (i, j), value in np.ndenumerate(matrix):
        colour = "white" if value > high else "black"
        ax.text(j, i, f"{value:g}", ha="center", va="center", color=colour)
    return fig


def plot_reliability(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Figure:
    """Reliability diagram for calibration."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    bins = np.clip((y_prob * n_bins).astype(int), 0, n_bins - 1)
    used = [b for b in range(n_bins) if (bins == b).any()]
    fig, ax = new_figure()
    ax.plot([0, 1], [0, 1], color="#B3B3B3", linestyle="--", linewidth=1)
    ax.plot(
        [y_prob[bins == b].mean() for b in used],
        [y_true[bins == b].mean() for b in used],
        marker="o",
        color=PALETTE[0],
    )
    ax.set(xlabel="Predicted AI probability", ylabel="Share actually AI")
    return fig
