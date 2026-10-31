"""Evaluate the full detector (checkpoints/) on edited audio and on separated stems.

Protocols:
    robustness  Test clips go through every edit in configs/augment.yaml (MP3, slowed and
                sped up, pitch, time stretch, noise, loudness) before analysis, the way
                re-uploaded music arrives. Reports ROC-AUC, the share of AI clips caught and
                of real clips flagged, per edit.
    stems       Vocals and instrumental are separated with Demucs and scored on their own.
                On val, each stem gets the threshold that flags at most
                evaluation.max_fpr_real of real clips; it is saved into
                checkpoints/ensemble.joblib for the API. On test, the thresholds are checked
                on real and AI clips and on hybrids: AI vocals over a real instrumental and
                real vocals over an AI instrumental (stems of different test songs).
                Both splits also report how often whole songs are flagged by how much
                singing they contain, to catch a "vocals = AI" shortcut.
For the held-out-tracks and unseen-generator protocols, see scripts/train.py,
scripts/fit_ensemble.py and scripts/unseen_generators.py.

Usage:
    python scripts/evaluate.py --protocol robustness --n 150
    python scripts/evaluate.py --protocol stems --split val --n 200
    python scripts/evaluate.py --protocol stems --split test --n 200
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from verasonus.api.inference import load_detector, score_audio
from verasonus.augmentation.transforms import apply_suite
from verasonus.config import REPO_ROOT, load_config, load_yaml
from verasonus.evaluation.metrics import threshold_at_fpr
from verasonus.evaluation.reports import save_table
from verasonus.preprocessing.audio_io import load_audio, resample


def sample_clips(meta: pd.DataFrame, split: str, n: int, seed: int) -> pd.DataFrame:
    """Up to ``n`` clips per source and label from one split (real FMA, AI per source)."""
    clips = meta[meta["split"] == split]
    return (
        clips.sample(frac=1, random_state=seed)
        .groupby(["source_dataset", "label"])
        .head(n)
        .reset_index(drop=True)
    )


def robustness(meta: pd.DataFrame, n: int) -> None:
    det = load_detector()
    cfg = det.cfg
    rate = cfg.audio.target_sample_rate
    suite = load_yaml(REPO_ROOT / "configs" / "augment.yaml")
    clips = sample_clips(meta, "test", n, cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for clip in clips.itertuples():
        # Edit at twice the model's rate, as a real upload is made from full-band audio:
        # a slowed 16 kHz file would lose its top band, which no real "slowed" upload does.
        audio, _ = load_audio(cfg.paths.data_root / clip.path, target_sr=2 * rate)
        edits = [("original", audio), *apply_suite(audio, 2 * rate, suite, rng)]
        for edit, edited in edits:
            scored = score_audio(det, resample(edited, 2 * rate, rate))
            if scored is not None:
                rows.append(
                    {
                        "edit": edit,
                        "y": int(clip.label == "ai"),
                        "generator": clip.generator,
                        "score": scored.track,
                    }
                )
    scores = pd.DataFrame(rows)
    table = scores.groupby("edit").apply(
        lambda g: pd.Series(
            {
                "roc_auc": roc_auc_score(g["y"], g["score"]) if g["y"].nunique() == 2 else np.nan,
                "ai_caught": (g.loc[g["y"] == 1, "score"] >= 0.5).mean(),
                "real_flagged": (g.loc[g["y"] == 0, "score"] >= 0.5).mean(),
                "n": len(g),
            }
        )
    )
    print(table.round(3).to_string())
    save_table(table, "robustness_by_edit")


def stem_scores(det, path, cfg) -> dict[str, np.ndarray]:
    """The separated stems of one file at the model's rate."""
    from verasonus.features.separation import vocals_and_instrumental

    original, original_rate = load_audio(path, mono=False)
    stems = vocals_and_instrumental(original, original_rate, device=det.device)
    return {k: resample(v, original_rate, cfg.audio.target_sample_rate) for k, v in stems.items()}


def vocal_share(stems: dict[str, np.ndarray]) -> float:
    """Share of the energy in the vocals stem: near 0 for instrumentals, higher for songs."""
    vocals = float(np.mean(stems["vocals"] ** 2))
    return vocals / (vocals + float(np.mean(stems["instrumental"] ** 2)) + 1e-12)


def false_alarms_by_vocals(mixes: pd.DataFrame, likely_ai: float, split: str) -> None:
    """Do songs with singing get flagged more? Share flagged per label and amount of vocals.

    Every SONICS AI song has vocals, so a model could learn "vocals = AI" and flag real
    singers. Real songs are compared by how much of their energy is vocals.
    """
    bins = pd.cut(
        mixes["vocal_share"],
        [0, 0.05, 0.2, 1],
        labels=["instrumental (<5%)", "some vocals (5-20%)", "vocal (>20%)"],
        include_lowest=True,
    )
    table = mixes.groupby(["kind", bins], observed=True)["score"].agg(
        n="size",
        flagged_at_0_5=lambda s: (s >= 0.5).mean(),
        flagged_likely_ai=lambda s: (s >= likely_ai).mean(),
        mean_score="mean",
    )
    print(table.round(3).to_string())
    save_table(table, f"false_alarms_by_vocals_{split}")


def stems_protocol(meta: pd.DataFrame, split: str, n: int) -> None:
    det = load_detector()
    cfg = det.cfg
    rate = cfg.audio.target_sample_rate
    clips = sample_clips(meta, split, n, cfg.seed)
    separated, rows, mixes = {}, [], []
    for clip in clips.itertuples():
        stems = stem_scores(det, cfg.paths.data_root / clip.path, cfg)
        separated[clip.clip_id] = (clip, stems)
        mix, _ = load_audio(cfg.paths.data_root / clip.path, target_sr=rate)
        scored = score_audio(det, mix)
        if scored is not None:
            mixes.append(
                {"kind": clip.label, "vocal_share": vocal_share(stems), "score": scored.track}
            )
        for name, audio in stems.items():
            scored = score_audio(det, audio, quiet_floor=0.1)
            if scored is not None:
                rows.append(
                    {
                        "clip_id": clip.clip_id,
                        "kind": clip.label,
                        "stem": name,
                        "generator": clip.generator,
                        "score": scored.track,
                    }
                )

    max_fpr = float(cfg.raw["evaluation"]["max_fpr_real"])
    bundle_path = cfg.paths.checkpoints / "ensemble.joblib"
    bundle = joblib.load(bundle_path)
    false_alarms_by_vocals(pd.DataFrame(mixes), bundle["verdict"]["likely_ai"], split)
    if split == "val":
        scores = pd.DataFrame(rows)
        thresholds = {}
        for stem, group in scores.groupby("stem"):
            y = (group["kind"] == "ai").astype(int).to_numpy()
            thresholds[stem] = max(0.5, threshold_at_fpr(y, group["score"].to_numpy(), max_fpr))
        bundle["stem_thresholds"] = thresholds
        joblib.dump(bundle, bundle_path)
        print(f"stem thresholds (flag at most {max_fpr:.0%} of real val clips): {thresholds}")
        return

    # Hybrids from different test songs: AI vocals + real instrumental and the reverse.
    rng = np.random.default_rng(cfg.seed)
    real = [c for c, _ in separated.values() if c.label == "real"]
    fake = [c for c, _ in separated.values() if c.label == "ai" and c.source_dataset == "sonics"]
    for kind, vocal_clips, backing_clips in (
        ("hybrid_ai_vocals", fake, real),
        ("hybrid_ai_instrumental", real, fake),
    ):
        order = rng.permutation(len(backing_clips))
        for vocal_clip, i in zip(vocal_clips, order, strict=False):
            backing_clip = backing_clips[i]
            vocals = separated[vocal_clip.clip_id][1]["vocals"]
            backing = separated[backing_clip.clip_id][1]["instrumental"]
            length = min(len(vocals), len(backing))
            mix = vocals[:length] + backing[:length]
            stems = {"vocals": vocals[:length], "instrumental": backing[:length], "mix": mix}
            for name, audio in stems.items():
                scored = score_audio(det, audio, quiet_floor=0.1 if name != "mix" else 0.0)
                if scored is not None:
                    rows.append(
                        {
                            "clip_id": f"{vocal_clip.clip_id}+{backing_clip.clip_id}",
                            "kind": kind,
                            "stem": name,
                            "generator": "",
                            "score": scored.track,
                        }
                    )

    scores = pd.DataFrame(rows)
    thresholds = bundle["stem_thresholds"]
    scores["flagged"] = [
        s >= thresholds.get(stem, bundle["verdict"]["likely_ai"])
        for s, stem in zip(scores["score"], scores["stem"], strict=True)
    ]
    table = scores.pivot_table(index="kind", columns="stem", values="flagged", aggfunc="mean")
    any_stem = scores[scores["stem"] != "mix"].groupby(["kind", "clip_id"])["flagged"].any()
    table["any_stem_flagged"] = any_stem.groupby("kind").mean()
    print(json.dumps(thresholds))
    print(table.round(3).to_string())
    save_table(table, "stems_flagged_test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the full VeraSonus detector.")
    parser.add_argument("--protocol", choices=["robustness", "stems"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--n", type=int, default=150, help="clips per source and label")
    args = parser.parse_args()
    cfg = load_config()
    meta = pd.read_csv(cfg.paths.metadata, keep_default_na=False)
    if args.protocol == "robustness":
        robustness(meta, args.n)
    else:
        stems_protocol(meta, args.split, args.n)


if __name__ == "__main__":
    main()
