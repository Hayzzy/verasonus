"""Train one model family (Phases 1-2). Runs locally or in a Colab notebook cell.

Classical models train on the cached handcrafted features and the CNN on the cached log-mel
spectrograms (scripts/extract_features.py). Hyperparameters are tuned (classical) or the
best epoch is kept (CNN) on val, then the model is scored on test, per window and per clip
(the mean of its window scores). A clip counts as AI above a threshold picked on val clips
(evaluation.max_fpr_real in the config); test is also scored at 0.5 for comparison. The model
and its threshold go to checkpoints/ and the metrics to reports/tables/.

Usage:
    python scripts/train.py --model random_forest
    python scripts/train.py --model cnn                # settings in the config's deep section
    python scripts/train.py --model cnn --epochs 2     # a quick check
"""

import argparse
import json
from collections.abc import Callable
from typing import Any

import joblib
import numpy as np
import pandas as pd

from verasonus.config import Config, load_config
from verasonus.evaluation.metrics import (
    classification_metrics,
    equal_error_rate,
    expected_calibration_error,
    false_positive_rate,
    threshold_at_fpr,
)
from verasonus.features.cache import load_feature_table, load_spectrogram_cache
from verasonus.models.classical import train_classical

CLASSICAL = ["svm", "random_forest", "gradient_boosting"]
DEEP = ["cnn"]
TRAINABLE = [*CLASSICAL, *DEEP]
MODELS = [*TRAINABLE, "transformer", "probe_mert"]
DEFAULT_FEATURES = {**dict.fromkeys(CLASSICAL, "handcrafted"), "cnn": "mel"}

Scorer = Callable[[np.ndarray], np.ndarray]  # boolean window mask -> AI score per window


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """All binary metrics for one set of AI scores (labels: 1 = AI)."""
    eer, _ = equal_error_rate(y_true, y_score)
    return {
        **classification_metrics(y_true, y_score, threshold),
        "eer": eer,
        "ece": expected_calibration_error(y_true, y_score),
        "fpr_real": false_positive_rate(y_true, y_score, threshold),
        "n": int(len(y_true)),
    }


def clip_scores(windows: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Mean window score per clip, with its label (1 = AI) and generator."""
    scored = windows.assign(score=scores, y=(windows["label"] == "ai").astype(int))
    return scored.groupby("clip_id").agg(
        score=("score", "mean"), y=("y", "first"), generator=("generator", "first")
    )


def clip_threshold(windows: pd.DataFrame, scores: np.ndarray, max_fpr: float) -> float:
    """Threshold on val clip scores that flags at most ``max_fpr`` of the real val clips."""
    clips = clip_scores(windows, scores)
    return threshold_at_fpr(clips["y"].to_numpy(), clips["score"].to_numpy(), max_fpr)


def load_inputs(model_name: str, features: str, cfg: Config) -> tuple[pd.DataFrame, np.ndarray]:
    """The per-window table and the model's inputs: a feature matrix or spectrogram array."""
    folder = cfg.paths.processed / features
    if model_name in DEEP:
        windows, specs, _ = load_spectrogram_cache(folder, mmap=bool(cfg.raw["deep"]["mmap"]))
        return windows, specs
    windows, matrix, _ = load_feature_table(folder)
    if not np.isfinite(matrix).all():
        raise SystemExit("features contain NaN or infinity; check the feature functions")
    return windows, matrix


def fit(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    cfg: Config,
    epochs: int | None = None,
) -> tuple[Any, dict[str, Any], Scorer]:
    """Train on the "train" windows and tune or stop on "val".

    Returns the model, its val metrics and a function scoring any set of windows.
    """
    train, val = split == "train", split == "val"
    if model_name in CLASSICAL:
        model, val_metrics = train_classical(
            model_name, x[train], y[train], x[val], y[val], seed=cfg.seed
        )
        print(f"Chosen settings: {model[-1].get_params()}")
        return model, val_metrics, lambda mask: model.predict_proba(x[mask])[:, 1]

    from verasonus.models.deep import build_cnn, predict_deep, train_deep

    d = cfg.raw["deep"]
    model = build_cnn()
    history = train_deep(
        model,
        x,
        y,
        np.flatnonzero(train),
        np.flatnonzero(val),
        epochs=epochs or int(d["epochs"]),
        lr=float(d["lr"]),
        weight_decay=float(d["weight_decay"]),
        batch_size=int(d["batch_size"]),
        patience=int(d["patience"]),
        augment=bool(d["spec_augment"]),
        balance=bool(d["balance_classes"]),
        seed=cfg.seed,
    )

    def score(mask: np.ndarray) -> np.ndarray:
        return predict_deep(model, x, np.flatnonzero(mask))

    val_metrics = {**classification_metrics(y[val], score(val)), "history": history}
    return model, val_metrics, score


def save_model(model_name: str, model: Any, threshold: float, features: str, cfg: Config) -> None:
    """Save the model with its threshold (and spectrogram settings) under checkpoints/."""
    cfg.paths.checkpoints.mkdir(parents=True, exist_ok=True)
    if model_name in DEEP:
        import torch

        settings_path = cfg.paths.processed / features / "settings.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        torch.save(
            {"state_dict": model.state_dict(), "threshold": threshold, "spectrogram": settings},
            cfg.paths.checkpoints / f"{model_name}.pt",
        )
    else:
        columns_path = cfg.paths.processed / features / "columns.txt"
        columns = columns_path.read_text(encoding="utf-8").split()
        joblib.dump(
            {"model": model, "columns": columns, "threshold": threshold},
            cfg.paths.checkpoints / f"{model_name}.joblib",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VeraSonus model.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--epochs", type=int, help="deep models: overrides deep.epochs")
    parser.add_argument("--features", help="folder under processed/ (default: by model)")
    args = parser.parse_args()
    if args.model not in TRAINABLE:
        raise NotImplementedError("Phase 2: Transformer and probes (verasonus.models)")
    features = args.features or DEFAULT_FEATURES[args.model]

    cfg = load_config()
    windows, x = load_inputs(args.model, features, cfg)
    y = (windows["label"] == "ai").to_numpy(dtype=int)
    split = windows["split"].to_numpy()
    print(f"Inputs {x.shape[1:]}; windows per split: {dict(pd.Series(split).value_counts())}")

    model, val_metrics, score = fit(args.model, x, y, split, cfg, args.epochs)
    print(f"Val ROC-AUC {val_metrics['roc_auc']:.4f}")

    max_fpr = float(cfg.raw["evaluation"]["max_fpr_real"])
    val = split == "val"
    threshold = clip_threshold(windows[val], score(val), max_fpr)
    print(f"Clip threshold {threshold:.4f} (flags at most {max_fpr:.0%} of real val clips)")

    test = split == "test"
    window_scores = score(test)
    clips = clip_scores(windows[test], window_scores)
    y_clip, s_clip = clips["y"].to_numpy(), clips["score"].to_numpy()

    results = {
        "model": args.model,
        "features": features,
        "val_window": val_metrics,
        "threshold": {"max_fpr_real": max_fpr, "value": threshold},
        "test_window": evaluate(y[test], window_scores, threshold),
        "test_clip": evaluate(y_clip, s_clip, threshold),
        "test_clip_at_0.5": evaluate(y_clip, s_clip),
        # Share of each generator's clips scored as AI (for "human": the false-positive rate).
        "test_clip_flagged_ai_by_generator": (
            (clips["score"] >= threshold).groupby(clips["generator"]).mean().round(4).to_dict()
        ),
    }
    print(json.dumps({k: v for k, v in results.items() if k != "val_window"}, indent=2))

    save_model(args.model, model, threshold, features, cfg)
    out = cfg.paths.reports / "tables" / f"{args.model}_{features}_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved the model to {cfg.paths.checkpoints} and the metrics to {out}")


if __name__ == "__main__":
    main()
