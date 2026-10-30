"""Leave-one-generator-out test: how well does a model catch a generator it never saw?

For each AI generator in turn, every window of that generator is taken out of train, val and
test and becomes "test_unseen". A classical model is tuned and trained on the rest, then
scored on:
  * seen:   the usual test split (real clips + the other generators)
  * unseen: the held-out generator's clips + the same real test clips
The real clips keep their splits from metadata.csv, so every fold scores the same real music
and the numbers line up with scripts/train.py. As there, a clip counts as AI above a
threshold picked on the fold's val clips (evaluation.max_fpr_real).

Usage:
    python scripts/unseen_generators.py                   # gradient boosting, all generators
    python scripts/unseen_generators.py --model random_forest
"""

import argparse
import json

import numpy as np
import pandas as pd

# Run as a script, so scripts/ is on the import path and train.py's metric helper is reused.
from train import CLASSICAL, clip_scores, clip_threshold, evaluate

from verasonus.config import load_config
from verasonus.data.splits import UNSEEN
from verasonus.features.cache import load_feature_table
from verasonus.models.classical import train_classical


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-generator-out evaluation.")
    parser.add_argument("--model", choices=CLASSICAL, default="gradient_boosting")
    parser.add_argument("--features", default="handcrafted", help="folder under processed/")
    args = parser.parse_args()

    cfg = load_config()
    windows, features, _ = load_feature_table(cfg.paths.processed / args.features)
    y = (windows["label"] == "ai").to_numpy(dtype=int)
    max_fpr = float(cfg.raw["evaluation"]["max_fpr_real"])
    generators = sorted(windows.loc[windows["label"] == "ai", "generator"].unique())

    folds = {}
    for generator in generators:
        split = windows["split"].where(windows["generator"] != generator, UNSEEN).to_numpy()

        def part(name: str, split: np.ndarray = split) -> tuple[np.ndarray, np.ndarray]:
            mask = split == name
            return features[mask], y[mask]

        model, val_metrics = train_classical(
            args.model, *part("train"), *part("val"), seed=cfg.seed
        )
        val = split == "val"
        threshold = clip_threshold(model, windows[val], features[val], max_fpr)
        scored = (split == "test") | (split == UNSEEN)
        fold = windows[scored].assign(split=split[scored])
        clips = clip_scores(fold, model.predict_proba(features[scored])[:, 1])
        clip_split = fold.groupby("clip_id")["split"].first()
        seen = clips[clip_split == "test"]
        real = seen[seen["y"] == 0]
        unseen = pd.concat([clips[clip_split == UNSEEN], real])

        folds[generator] = {
            "val_roc_auc": val_metrics["roc_auc"],
            "threshold": threshold,
            "seen_clip": evaluate(seen["y"].to_numpy(), seen["score"].to_numpy(), threshold),
            "unseen_clip": evaluate(unseen["y"].to_numpy(), unseen["score"].to_numpy(), threshold),
            # Share of the held-out generator's clips scored as AI (its recall).
            "unseen_flagged_ai": round(
                float((unseen.loc[unseen["y"] == 1, "score"] >= threshold).mean()), 4
            ),
        }
        f = folds[generator]
        print(
            f"held out {generator:<18} seen AUC {f['seen_clip']['roc_auc']:.3f}  "
            f"unseen AUC {f['unseen_clip']['roc_auc']:.3f}  "
            f"caught {f['unseen_flagged_ai']:.1%}  real flagged {f['unseen_clip']['fpr_real']:.1%}"
        )

    out = cfg.paths.reports / "tables" / f"{args.model}_{args.features}_unseen_generators.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "features": args.features,
                "max_fpr_real": max_fpr,
                "folds": folds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
