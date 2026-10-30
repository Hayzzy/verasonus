"""Train one model family (Phases 1-2). Runs locally or in a Colab notebook cell.

Classical models train on the cached handcrafted features (scripts/extract_features.py):
hyperparameters are tuned on val, then the model is scored on test, per window and per clip
(the mean of its window scores). The model goes to checkpoints/ and the metrics to
reports/tables/.

Usage:
    python scripts/train.py --model random_forest
    python scripts/train.py --model cnn --epochs 20
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from verasonus.config import load_config
from verasonus.evaluation.metrics import (
    classification_metrics,
    equal_error_rate,
    expected_calibration_error,
    false_positive_rate,
)
from verasonus.features.cache import load_feature_table
from verasonus.models.classical import train_classical

CLASSICAL = ["svm", "random_forest", "gradient_boosting"]
MODELS = [*CLASSICAL, "cnn", "transformer", "probe_mert"]


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """All binary metrics for one set of AI scores (labels: 1 = AI)."""
    eer, _ = equal_error_rate(y_true, y_score)
    return {
        **classification_metrics(y_true, y_score),
        "eer": eer,
        "ece": expected_calibration_error(y_true, y_score),
        "fpr_real": false_positive_rate(y_true, y_score),
        "n": int(len(y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VeraSonus model.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--epochs", type=int, default=20, help="deep models only")
    parser.add_argument("--features", default="handcrafted", help="folder under processed/")
    args = parser.parse_args()
    if args.model not in CLASSICAL:
        raise NotImplementedError("Phase 2: deep models and probes (verasonus.models.deep)")

    cfg = load_config()
    windows, features, columns = load_feature_table(cfg.paths.processed / args.features)
    y = (windows["label"] == "ai").to_numpy(dtype=int)
    split = windows["split"].to_numpy()
    if not np.isfinite(features).all():
        raise SystemExit("features contain NaN or infinity; check the feature functions")

    def part(name: str) -> tuple[np.ndarray, np.ndarray]:
        mask = split == name
        return features[mask], y[mask]

    print(f"{len(columns)} features; windows per split: {dict(pd.Series(split).value_counts())}")
    model, val_metrics = train_classical(args.model, *part("train"), *part("val"), seed=cfg.seed)
    print(f"Chosen settings: {model[-1].get_params()} (val ROC-AUC {val_metrics['roc_auc']:.4f})")

    test = windows[split == "test"].copy()
    test["score"] = model.predict_proba(features[split == "test"])[:, 1]
    test["y"] = y[split == "test"]
    clips = test.groupby("clip_id").agg(score=("score", "mean"), y=("y", "first"))
    clips["generator"] = test.groupby("clip_id")["generator"].first()

    results = {
        "model": args.model,
        "features": args.features,
        "val_window": val_metrics,
        "test_window": evaluate(test["y"].to_numpy(), test["score"].to_numpy()),
        "test_clip": evaluate(clips["y"].to_numpy(), clips["score"].to_numpy()),
        # Share of each generator's clips scored as AI (for "human": the false-positive rate).
        "test_clip_flagged_ai_by_generator": (
            (clips["score"] >= 0.5).groupby(clips["generator"]).mean().round(4).to_dict()
        ),
    }
    print(json.dumps(results, indent=2))

    cfg.paths.checkpoints.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "columns": columns}, cfg.paths.checkpoints / f"{args.model}.joblib"
    )
    out = cfg.paths.reports / "tables" / f"{args.model}_{args.features}_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved the model to {cfg.paths.checkpoints} and the metrics to {out}")


if __name__ == "__main__":
    main()
