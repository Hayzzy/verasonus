"""Compare models on the leave-one-generator-out test (reads scripts/unseen_generators.py output).

Saves reports/tables/unseen_generators_summary.csv and a bar chart of the share of each
held-out generator's clips caught, seen versus unseen, per model.

Usage:
    python scripts/plot_unseen.py
    python scripts/plot_unseen.py --runs gradient_boosting_handcrafted cnn_mel
"""

import argparse
import json

import numpy as np
import pandas as pd

from verasonus.config import load_config
from verasonus.evaluation.reports import PALETTE, new_figure, save_figure, save_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise the unseen-generator test.")
    parser.add_argument("--runs", nargs="+", default=["gradient_boosting_handcrafted", "cnn_mel"])
    args = parser.parse_args()
    cfg = load_config()
    tables = cfg.paths.reports / "tables"

    rows = []
    for run in args.runs:
        result = json.loads((tables / f"{run}_unseen_generators.json").read_text(encoding="utf-8"))
        for generator, fold in result["folds"].items():
            rows.append(
                {
                    "model": result["model"],
                    "held_out": generator,
                    "seen_roc_auc": fold["seen_clip"]["roc_auc"],
                    "unseen_roc_auc": fold["unseen_clip"]["roc_auc"],
                    "seen_recall": fold["seen_clip"]["recall"],
                    "unseen_caught": fold["unseen_flagged_ai"],
                    "real_flagged": fold["unseen_clip"]["fpr_real"],
                }
            )
    summary = pd.DataFrame(rows).set_index(["model", "held_out"])
    save_table(summary, "unseen_generators_summary")
    print(summary.round(3).to_string())

    caught = summary["unseen_caught"].unstack("model")
    fig, ax = new_figure((7, 4))
    width = 0.8 / len(caught.columns)
    x = np.arange(len(caught))
    for i, (model, colour) in enumerate(zip(caught.columns, PALETTE, strict=False)):
        ax.bar(x + i * width, caught[model], width, label=model, color=colour)
    ax.set_xticks(x + width * (len(caught.columns) - 1) / 2, caught.index, rotation=20)
    ax.set(ylabel="Held-out clips caught", ylim=(0, 1.05))
    ax.set_title("Catching a generator never seen in training")
    ax.legend()
    save_figure(fig, "unseen_generators_caught")


if __name__ == "__main__":
    main()
