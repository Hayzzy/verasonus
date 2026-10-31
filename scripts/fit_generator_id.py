"""Name the likely generator of an AI track, or say "unknown", and map how generators cluster.

Works on the cached handcrafted features of the AI windows only. Generators are named by
platform (Suno and Udio versions are merged into "suno" and "udio"). Steps:
    1. supervised: gradient boosting over the platforms, trained on train windows and
       scored on test clips (mean of each clip's window probabilities)
    2. "unknown" detector: an IsolationForest fitted on the known generators' train windows
       (models.unsupervised). Tested leave-one-generator-out: fitted without one generator,
       how many of that generator's test clips look unfamiliar?
    3. unsupervised: PCA map and k-means clusters of the AI test clips, compared with the
       true generators (adjusted Rand index; 0 = no match, 1 = perfect)
    4. feature importance for the real-vs-AI gradient boosting model (explain.importance)
The classifier and detector go to checkpoints/generator_id.joblib for the API.

Usage:
    python scripts/fit_generator_id.py
    python scripts/fit_generator_id.py --features handcrafted_v2
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix, f1_score

from verasonus.config import load_config
from verasonus.evaluation.reports import (
    PALETTE,
    new_figure,
    plot_confusion,
    save_figure,
    save_table,
)
from verasonus.explain.importance import permutation_importance_table
from verasonus.features.cache import load_feature_table
from verasonus.models.unsupervised import cluster, embedding_map, fit_unknown_detector, is_unknown


def clip_mean(windows: pd.DataFrame, values: np.ndarray) -> tuple[pd.Series, np.ndarray]:
    """Average rows of ``values`` per clip; return the clips' generators and the means."""
    frame = pd.DataFrame(values, index=windows["clip_id"].to_numpy())
    means = frame.groupby(level=0, sort=True).mean()
    generators = windows.groupby("clip_id", sort=True)["generator"].first()
    return generators.loc[means.index], means.to_numpy()


def platform(generator: str) -> str:
    """ "suno_v3.5" -> "suno"; research generators keep their own name."""
    return generator.split("_")[0] if generator.startswith(("suno_", "udio_")) else generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generator ID, unknown detector, clusters.")
    parser.add_argument("--features", default="handcrafted", help="folder under processed/")
    args = parser.parse_args()
    cfg = load_config()
    windows, features, columns = load_feature_table(cfg.paths.processed / args.features)
    windows["generator"] = windows["generator"].map(platform)
    ai = (windows["label"] == "ai").to_numpy()
    split = windows["split"].to_numpy()
    train, test = ai & (split == "train"), ai & (split == "test")
    tables = cfg.paths.reports / "tables"
    summary = {}

    # 1. Which generator?
    classifier = HistGradientBoostingClassifier(random_state=cfg.seed)
    classifier.fit(features[train], windows["generator"][train])
    classes = list(classifier.classes_)
    truth, probs = clip_mean(windows[test], classifier.predict_proba(features[test]))
    predicted = np.array(classes)[probs.argmax(axis=1)]
    summary["generator_id_test_clips"] = {
        "accuracy": accuracy_score(truth, predicted),
        "macro_f1": f1_score(truth, predicted, average="macro"),
        "n": len(truth),
    }
    matrix = confusion_matrix(truth, predicted, labels=classes)
    save_figure(plot_confusion(matrix, classes), "confusion_generator_id_test_clips")
    print("generator ID:", summary["generator_id_test_clips"])

    # 2. "Unknown" detector, leave one generator out.
    rows = []
    for held_out in classes:
        known = train & (windows["generator"] != held_out).to_numpy()
        detector = fit_unknown_detector(features[known], contamination=0.05)
        flags = is_unknown(detector, features[test])
        gens, clip_flags = clip_mean(windows[test], flags[:, None].astype(float))
        flagged = pd.Series(clip_flags[:, 0] > 0.5, index=gens.index).groupby(gens).mean()
        rows.append(
            {
                "held_out": held_out,
                "held_out_flagged_unknown": flagged[held_out],
                "known_flagged_unknown": flagged.drop(held_out).mean(),
            }
        )
    unknown = pd.DataFrame(rows).set_index("held_out")
    save_table(unknown, "unknown_detector_leave_one_out")
    print(unknown.round(3).to_string())
    detector = fit_unknown_detector(features[train], contamination=0.05)

    # 3. Unsupervised map of the AI test clips.
    gens, clip_features = clip_mean(windows[test], features[test])
    points = embedding_map(clip_features, method="pca", seed=cfg.seed)
    clusters = cluster(clip_features, n_clusters=len(classes), seed=cfg.seed)
    summary["kmeans_adjusted_rand_index"] = adjusted_rand_score(gens, clusters)
    print(f"k-means vs generators, adjusted Rand index {summary['kmeans_adjusted_rand_index']:.3f}")
    fig, ax = new_figure((5.5, 4.5))
    for name, colour in zip(classes, PALETTE, strict=False):
        mask = (gens == name).to_numpy()
        ax.scatter(*points[mask].T, s=8, alpha=0.6, color=colour, label=name)
    ax.set(xlabel="PC 1", ylabel="PC 2", title="AI test clips, handcrafted features (PCA)")
    ax.legend(markerscale=2, fontsize=8)
    save_figure(fig, "generator_map_pca")

    # 4. Which features drive the real-vs-AI gradient boosting model?
    gb = joblib.load(cfg.paths.checkpoints / "gradient_boosting.joblib")["model"]
    val = np.flatnonzero(split == "val")
    sample = np.random.default_rng(cfg.seed).choice(val, size=min(3000, len(val)), replace=False)
    y_val = (windows["label"].to_numpy()[sample] == "ai").astype(int)
    importance = permutation_importance_table(gb, features[sample], y_val, columns, n_repeats=5)
    save_table(importance, "gradient_boosting_permutation_importance")
    top = importance.head(15)[::-1]
    fig, ax = new_figure((6, 5))
    ax.barh(top.index, top["importance_mean"], xerr=top["importance_std"], color=PALETTE[0])
    ax.set(xlabel="Drop in val ROC-AUC when shuffled", title="Gradient boosting: top features")
    save_figure(fig, "gradient_boosting_permutation_importance")
    print(importance.head(10).round(4).to_string())

    joblib.dump(
        {"classifier": classifier, "classes": classes, "unknown_detector": detector},
        cfg.paths.checkpoints / "generator_id.joblib",
    )
    out = tables / "generator_id_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoints/generator_id.joblib and reports under {cfg.paths.reports}")


if __name__ == "__main__":
    main()
