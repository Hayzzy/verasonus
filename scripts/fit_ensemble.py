"""Fuse the CNN, gradient boosting and (optionally) SpecTTTra; calibrate; compare on test.

Needs both trained models (scripts/train.py --model cnn / gradient_boosting), both caches
(scripts/extract_features.py, with and without --features mel) and, for SpecTTTra, its clip
scores (scripts/eval_spectttra.py). Steps:
    1. score the val and test windows with both models; average each clip's windows
    2. pick the fusion weights on val clips (ensemble.fusion.tune_weights)
    3. fit a temperature on val so the fused score reads as a probability (ensemble.calibration)
    4. score test clips with each model and the ensemble, and save tables and figures
    5. save checkpoints/ensemble.joblib for the API: weights, temperature, verdict bands and
       the median real training window (the reference for explanations)

Usage:
    python scripts/fit_ensemble.py
    python scripts/fit_ensemble.py --handcrafted handcrafted_v2 --mel mel_v2 --spectttra
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve
from train import clip_scores, evaluate

from verasonus.config import load_config
from verasonus.ensemble.calibration import apply_temperature, fit_temperature, to_logits
from verasonus.ensemble.fusion import tune_weights, weighted_vote
from verasonus.evaluation.reports import (
    plot_confusion,
    plot_reliability,
    plot_roc,
    save_figure,
    save_table,
)
from verasonus.features.cache import load_feature_table, load_spectrogram_cache

# Each model keeps at least this weight. In the first version (FakeMusicCaps only) the CNN
# alone scored best on val (ROC-AUC 1.0) and tuning without a floor dropped gradient boosting
# entirely; we keep every model in for unseen material, and report the cost on test.
MIN_WEIGHT = 0.1
SPECTTTRA_ID = "awsaf49/sonics-spectttra-alpha-120s"

# Calibrated probability bands for the verdict shown in the app.
VERDICT = {"likely_ai": 0.7, "likely_human": 0.3}


def main() -> None:
    import torch

    from verasonus.models.deep import build_cnn, predict_deep

    parser = argparse.ArgumentParser(description="Fit the ensemble and compare models.")
    parser.add_argument("--handcrafted", default="handcrafted", help="folder under processed/")
    parser.add_argument("--mel", default="mel", help="folder under processed/")
    parser.add_argument("--spectttra", action="store_true", help="add pretrained SpecTTTra")
    args = parser.parse_args()

    cfg = load_config()
    windows, features, columns = load_feature_table(cfg.paths.processed / args.handcrafted)
    mel_windows, specs, _ = load_spectrogram_cache(cfg.paths.processed / args.mel, mmap=True)
    key = ["clip_id", "start_seconds"]
    if not windows[key].equals(mel_windows[key]):
        raise SystemExit("the handcrafted and mel caches hold different windows; rebuild one")

    gb_bundle = joblib.load(cfg.paths.checkpoints / "gradient_boosting.joblib")
    if gb_bundle["columns"] != columns:
        raise SystemExit("gradient boosting was trained on other feature columns; retrain it")
    gb = gb_bundle["model"]
    saved = torch.load(cfg.paths.checkpoints / "cnn.pt", map_location="cpu", weights_only=False)
    cnn = build_cnn()
    cnn.load_state_dict(saved["state_dict"])

    clips = {}
    for split in ("val", "test"):
        mask = (windows["split"] == split).to_numpy()
        rows = np.flatnonzero(mask)
        scores = {
            "cnn": predict_deep(cnn, specs, rows),
            "gradient_boosting": gb.predict_proba(features[rows])[:, 1],
        }
        per_model = {name: clip_scores(windows[mask], s) for name, s in scores.items()}
        table = per_model["cnn"][["y", "generator"]].copy()
        for name, c in per_model.items():
            table[name] = c["score"]
        clips[split] = table
        print(f"scored {len(rows)} {split} windows")

    val, test = clips["val"], clips["test"]
    models = ["cnn", "gradient_boosting"]
    if args.spectttra:
        name = SPECTTTRA_ID.split("/")[-1].removeprefix("sonics-")
        path = cfg.paths.processed / "spectttra" / f"{name}_repeat_clip_scores.csv"
        known = pd.read_csv(path).set_index("clip_id")["score"]
        for table in (val, test):
            table["spectttra"] = known.reindex(table.index).to_numpy()
            if table["spectttra"].isna().any():
                raise SystemExit(f"{path} lacks some clips: run scripts/eval_spectttra.py")
        models.append("spectttra")
    weights = tune_weights(
        {m: val[m].to_numpy() for m in models}, val["y"].to_numpy(), min_weight=MIN_WEIGHT
    )
    for table in (val, test):
        table["ensemble_raw"] = weighted_vote({m: table[m].to_numpy() for m in models}, weights)
    temperature = fit_temperature(to_logits(val["ensemble_raw"]), val["y"].to_numpy())
    for table in (val, test):
        table["ensemble"] = apply_temperature(to_logits(table["ensemble_raw"]), temperature)
    print(f"weights {weights}, temperature {temperature:.3f}")

    y = test["y"].to_numpy()
    results = {}
    for name in [*models, "ensemble"]:
        metrics = evaluate(y, test[name].to_numpy())
        flagged = (test[name] >= 0.5).groupby(test["generator"]).mean()
        results[name] = {**metrics, **{f"flagged_{g}": v for g, v in flagged.items()}}
    table = pd.DataFrame(results).T
    keep = ["roc_auc", "f1", "accuracy", "eer", "ece", "fpr_real"]
    print(table[keep + [c for c in table if c.startswith("flagged_")]].round(4).to_string())
    save_table(table, "test_clip_models_vs_ensemble")

    curves = {}
    for name in [*models, "ensemble"]:
        fpr, tpr, _ = roc_curve(y, test[name])
        curves[name] = (fpr, tpr)
    save_figure(plot_roc(curves), "roc_test_clips")
    matrix = confusion_matrix(y, test["ensemble"] >= 0.5)
    save_figure(plot_confusion(matrix, ["real", "AI"]), "confusion_ensemble_test_clips")
    save_figure(plot_reliability(y, test["ensemble"].to_numpy()), "reliability_ensemble_test_clips")

    train_real = ((windows["split"] == "train") & (windows["label"] == "real")).to_numpy()
    bundle = {
        "spectttra_id": SPECTTTRA_ID if args.spectttra else None,
        "weights": weights,
        "temperature": temperature,
        "verdict": VERDICT,
        "reference": np.median(features[train_real], axis=0),
        "columns": columns,
    }
    joblib.dump(bundle, cfg.paths.checkpoints / "ensemble.joblib")
    summary = {"weights": weights, "temperature": temperature, "verdict": VERDICT}
    out = cfg.paths.reports / "tables" / "ensemble_settings.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoints/ensemble.joblib and the tables and figures under {cfg.paths.reports}")


if __name__ == "__main__":
    main()
