"""Score val and test clips with the pretrained SONICS SpecTTTra model (no training on our data).

SpecTTTra was trained on SONICS's Suno v3.5 and Udio-120s training songs, so FakeMusicCaps'
five generators and Suno v2/v3 and Udio-30s are unseen by it, and FMA is real music it has
never heard. Clips shorter than its 120-second window are repeated to fill it
(``--fill pad`` pads with silence instead, as the SONICS code does, to compare the two).

Writes each clip's score to processed/spectttra/<model>_clip_scores.csv (scripts/
fit_ensemble.py reads it) and the test metrics to reports/tables/.

Usage:
    python scripts/eval_spectttra.py
    python scripts/eval_spectttra.py --fill pad --limit 300
"""

import argparse
import json

import numpy as np
import pandas as pd
from train import evaluate

from verasonus.config import load_config
from verasonus.evaluation.reports import save_table
from verasonus.models.deep import SPECTTTRA_RATE, load_spectttra, spectttra_scores
from verasonus.preprocessing.audio_io import load_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pretrained SpecTTTra.")
    parser.add_argument("--model-id", default="awsaf49/sonics-spectttra-alpha-120s")
    parser.add_argument("--fill", choices=["repeat", "pad"], default="repeat")
    parser.add_argument("--limit", type=int, help="test clips per label, for a quick check")
    args = parser.parse_args()
    cfg = load_config()
    meta = pd.read_csv(cfg.paths.metadata, keep_default_na=False)
    clips = meta[meta["split"].isin(["val", "test"])]
    if args.limit:
        clips = clips[clips["split"] == "test"].groupby("label").head(args.limit)
    model = load_spectttra(args.model_id)
    seconds = 120.0 if "120s" in args.model_id else 5.0
    name = args.model_id.split("/")[-1].removeprefix("sonics-")

    out_dir = cfg.paths.processed / "spectttra"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"{name}_{args.fill}_clip_scores.csv"
    done = pd.read_csv(cache_path) if cache_path.exists() and not args.limit else None
    known = dict(zip(done["clip_id"], done["score"], strict=True)) if done is not None else {}

    scores = []
    for i, (clip_id, path) in enumerate(zip(clips["clip_id"], clips["path"], strict=True)):
        if clip_id not in known:
            audio, _ = load_audio(cfg.paths.data_root / path, target_sr=SPECTTTRA_RATE)
            if args.fill == "pad" and len(audio) < seconds * SPECTTTRA_RATE:
                audio = np.pad(audio, (0, round(seconds * SPECTTTRA_RATE) - len(audio)))
            known[clip_id] = float(spectttra_scores(model, audio, seconds)[1].mean())
        scores.append(known[clip_id])
        if not args.limit and i % 2000 == 1999:  # save progress, so a stopped run can resume
            pd.DataFrame({"clip_id": list(known), "score": list(known.values())}).to_csv(
                cache_path, index=False
            )
            print(f"{i + 1}/{len(clips)} clips")
    clips = clips.assign(score=scores, y=(clips["label"] == "ai").astype(int))
    if not args.limit:
        clips[["clip_id", "score"]].to_csv(cache_path, index=False)

    test = clips[clips["split"] == "test"]
    results = {"all": evaluate(test["y"].to_numpy(), test["score"].to_numpy())}
    flagged = (test["score"] >= 0.5).groupby(test["generator"]).mean().round(4).to_dict()
    results["flagged_ai_by_generator"] = flagged
    print(json.dumps(results, indent=2))
    table = f"spectttra_{name}_{args.fill}"
    save_table(pd.DataFrame(results["all"], index=[table]), f"{table}_test_clips")
    save_table(pd.Series(flagged, name="flagged_ai").to_frame(), f"{table}_flagged_by_generator")


if __name__ == "__main__":
    main()
