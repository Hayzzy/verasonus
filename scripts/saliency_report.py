"""Where does the CNN look? Grad-CAM over test windows (explain.saliency).

Averages the Grad-CAM map of the "AI" class over random real and AI test windows, then
plots how much attention each mel band gets. A model that relies on one narrow band (or on
the very start or end of a window) is suspect of a shortcut. Also saves one example map.

Usage:
    python scripts/saliency_report.py --n 200
"""

import argparse

import librosa
import numpy as np
import pandas as pd

from verasonus.config import load_config
from verasonus.evaluation.reports import PALETTE, new_figure, save_figure, save_table
from verasonus.explain.saliency import saliency_map
from verasonus.features.cache import load_spectrogram_cache


def main() -> None:
    import torch

    from verasonus.models.deep import build_cnn

    parser = argparse.ArgumentParser(description="Grad-CAM summary for the CNN.")
    parser.add_argument("--n", type=int, default=200, help="test windows per class")
    args = parser.parse_args()
    cfg = load_config()
    windows, specs, settings = load_spectrogram_cache(cfg.paths.processed / "mel", mmap=True)
    saved = torch.load(cfg.paths.checkpoints / "cnn.pt", map_location="cpu", weights_only=False)
    model = build_cnn()
    model.load_state_dict(saved["state_dict"])

    rng = np.random.default_rng(cfg.seed)
    test = windows["split"] == "test"
    maps = {}
    for label in ("real", "ai"):
        rows = np.flatnonzero(test & (windows["label"] == label))
        rows = rng.choice(rows, size=min(args.n, len(rows)), replace=False)
        maps[label] = np.stack([saliency_map(model, specs[r].astype(np.float32)) for r in rows])
        print(f"{label}: {len(rows)} windows")

    rate = cfg.audio.target_sample_rate
    freqs = librosa.mel_frequencies(settings["n_mels"] + 2, fmax=settings["fmax"])[1:-1]
    by_band = pd.DataFrame({k: m.mean(axis=(0, 2)) for k, m in maps.items()}, index=freqs.round())
    by_band.index.name = "mel_band_centre_hz"
    save_table(by_band, "cnn_gradcam_by_mel_band")
    frames = np.arange(settings["n_frames"]) * settings["hop_length"] / rate
    by_time = pd.DataFrame({k: m.mean(axis=(0, 1)) for k, m in maps.items()}, index=frames)

    fig, _ = new_figure((9, 3.5))
    fig.clear()
    ax_f, ax_t = fig.subplots(1, 2)
    for label, colour in zip(("ai", "real"), PALETTE, strict=False):
        ax_f.plot(freqs, by_band[label], color=colour, label=label)
        ax_t.plot(frames, by_time[label], color=colour, label=label)
    ax_f.set(xlabel="Mel band centre (Hz)", ylabel="Mean Grad-CAM", title="Attention by frequency")
    ax_t.set(xlabel="Time in window (s)", title="Attention over time")
    ax_f.legend()
    save_figure(fig, "cnn_gradcam_profiles")

    fig, ax = new_figure((7, 3.5))
    example = np.flatnonzero(test & (windows["label"] == "ai"))[0]
    ax.imshow(specs[example].astype(np.float32), origin="lower", aspect="auto", cmap="gray")
    ax.imshow(
        saliency_map(model, specs[example].astype(np.float32)),
        origin="lower",
        aspect="auto",
        cmap="Oranges",
        alpha=0.5,
    )
    ax.set(xlabel="Frame (20 ms)", ylabel="Mel band", title="Grad-CAM on one AI test window")
    save_figure(fig, "cnn_gradcam_example")
    print(by_band.idxmax().to_dict(), "Hz carry the most attention")


if __name__ == "__main__":
    main()
