"""End-to-end analysis used by the API (Phase 3).

analyze_file:
    1. loads the audio; the mix is standardised exactly like the training clips (mono,
       16 kHz, 7 kHz low-pass, matched loudness, shared MP3 codec) and cut into windows
    2. scores every window with the CNN (log-mel) and gradient boosting (handcrafted
       features), and the whole track with the pretrained SONICS SpecTTTra model when the
       ensemble uses it
    3. fuses the scores with the weights tuned on val (ensemble.fusion), averages windows into
       a track score and timeline (ensemble.aggregation) and calibrates it
       (ensemble.calibration)
    4. separates vocals and instrumental with Demucs (features.separation) and scores each,
       so tracks where only one part is AI are caught ("partly AI")
    5. guesses the generator, or "unknown" when the outlier detector fires
       (models.unsupervised), and lists the handcrafted features that pushed the score
       (explain.importance)

The models come from checkpoints/: cnn.pt and gradient_boosting.joblib (scripts/train.py),
ensemble.joblib (scripts/fit_ensemble.py) and generator_id.joblib
(scripts/fit_generator_id.py). They are loaded once and kept in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from verasonus.api.schemas import (
    AnalysisResult,
    Explanation,
    FeatureContribution,
    GeneratorGuess,
    Segment,
    StemScore,
)
from verasonus.config import Config, load_config
from verasonus.ensemble.aggregation import aggregate
from verasonus.ensemble.calibration import apply_temperature, to_logits
from verasonus.ensemble.fusion import weighted_vote
from verasonus.explain.importance import explain_prediction
from verasonus.features.cache import split_windows
from verasonus.features.handcrafted import extract_handcrafted
from verasonus.features.spectrograms import window_mel
from verasonus.models.unsupervised import is_unknown
from verasonus.preprocessing.audio_io import load_audio, resample
from verasonus.preprocessing.standardise import standardise_audio

MODEL_VERSION = "v2"

# Readable names for the feature families in the explanation panel.
FEATURE_LABELS = {
    "mfcc_delta": "timbre change (MFCC delta)",
    "mfcc": "timbre (MFCC)",
    "centroid": "spectral brightness",
    "bandwidth": "spectral spread",
    "rolloff": "spectral roll-off",
    "flatness": "noisiness (spectral flatness)",
    "contrast": "peak-to-valley contrast",
    "zcr": "zero-crossing rate",
    "high_band_energy": "high-frequency energy",
    "artefact": "periodic decoder artefacts",
}


@dataclass
class Detector:
    cfg: Config
    cnn: Any
    spectrogram: dict[str, Any]
    gb: Any
    columns: list[str]
    ensemble: dict[str, Any]
    generator_id: dict[str, Any] | None
    spectttra: Any | None
    device: str


@dataclass
class Scores:
    """Window and track scores for one piece of audio (a mix or a stem)."""

    starts: np.ndarray
    parts: dict[str, np.ndarray]  # each model's AI score per window
    features: np.ndarray  # handcrafted features per window
    track: float  # calibrated track probability
    segments: list[tuple[float, float, float]]  # calibrated, per timeline segment


@cache
def load_detector() -> Detector:
    """Load every model once (the first request is slower)."""
    cfg = load_config()
    folder = cfg.paths.checkpoints
    missing = [
        name
        for name in ("cnn.pt", "gradient_boosting.joblib", "ensemble.joblib")
        if not (folder / name).exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing model files in {folder}: {', '.join(missing)}")

    import torch

    from verasonus.models.deep import build_cnn, default_device, load_spectttra

    saved = torch.load(folder / "cnn.pt", map_location="cpu", weights_only=False)
    cnn = build_cnn()
    cnn.load_state_dict(saved["state_dict"])
    cnn.eval()
    gb = joblib.load(folder / "gradient_boosting.joblib")
    ensemble = joblib.load(folder / "ensemble.joblib")
    generator_path = folder / "generator_id.joblib"
    use_spectttra = ensemble["weights"].get("spectttra", 0) > 0
    return Detector(
        cfg=cfg,
        cnn=cnn,
        spectrogram=saved["spectrogram"],
        gb=gb["model"],
        columns=gb["columns"],
        ensemble=ensemble,
        generator_id=joblib.load(generator_path) if generator_path.exists() else None,
        spectttra=load_spectttra(ensemble["spectttra_id"]) if use_spectttra else None,
        device=default_device(),
    )


def feature_label(name: str) -> str:
    """A readable label for a feature column such as ``mfcc_delta_3_std``."""
    for prefix, label in FEATURE_LABELS.items():
        if name.startswith(prefix):
            parts = name.removeprefix(prefix).strip("_").split("_")
            band = f" band {parts[0]}" if parts and parts[0].isdigit() else ""
            stat = ", variation" if name.endswith("_std") else ""
            return f"{label}{band}{stat}"
    return name


def calibrate(det: Detector, scores: np.ndarray | float) -> np.ndarray:
    return apply_temperature(to_logits(scores), det.ensemble["temperature"])


def model_scores(
    det: Detector, audio: np.ndarray, quiet_floor: float = 0.0
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray] | None:
    """Per-window scores of each model for 16 kHz mono audio (a mix or a stem).

    Returns ``(window starts, {model: scores}, handcrafted features)``, or None when no
    window is long or loud enough. ``quiet_floor`` drops windows quieter than that share of
    the loudest window, so a vocal stem is judged only where someone sings.
    """
    from verasonus.models.deep import predict_deep, spectttra_scores

    cfg = det.cfg
    rate = cfg.audio.target_sample_rate
    windows = split_windows(standardise_audio(audio, rate, cfg.audio), rate, cfg)
    if quiet_floor > 0 and windows:
        raw = split_windows(audio, rate, cfg)
        levels = np.array([np.sqrt(np.mean(np.square(w))) for _, w in raw])
        windows = [
            w
            for w, level in zip(windows, levels, strict=True)
            if level >= quiet_floor * levels.max()
        ]
    if not windows:
        return None
    starts = np.array([start for start, _ in windows])
    mels = np.stack([window_mel(data, rate, **det.spectrogram) for _, data in windows])
    feats = np.array(
        [[f[c] for c in det.columns] for f in (extract_handcrafted(d, rate) for _, d in windows)]
    )
    parts = {
        "cnn": predict_deep(det.cnn, mels, np.arange(len(mels)), device=det.device),
        "gradient_boosting": det.gb.predict_proba(feats)[:, 1],
    }
    if det.spectttra is not None:
        # SpecTTTra scores the unprocessed 16 kHz audio in 120 s windows; each of our windows
        # takes the score of the long window whose centre is nearest.
        long_starts, long_probs = spectttra_scores(det.spectttra, audio, device=det.device)
        centres = starts + cfg.windows.length_seconds / 2
        nearest = np.abs(centres[:, None] - (long_starts[None] + 60.0)).argmin(axis=1)
        parts["spectttra"] = long_probs[nearest]
    return starts, parts, feats


def score_audio(det: Detector, audio: np.ndarray, quiet_floor: float = 0.0) -> Scores | None:
    """Fused, calibrated track score and timeline for 16 kHz mono audio."""
    scored = model_scores(det, audio, quiet_floor)
    if scored is None:
        return None
    starts, parts, feats = scored
    cfg = det.cfg
    weights = det.ensemble["weights"]
    fused = weighted_vote({k: v for k, v in parts.items() if weights.get(k, 0) > 0}, weights)
    duration = len(audio) / cfg.audio.target_sample_rate
    track = aggregate(fused, starts, cfg.windows.length_seconds, duration)
    return Scores(
        starts=starts,
        parts=parts,
        features=feats,
        track=float(calibrate(det, track.ai_probability)),
        segments=[(s, e, float(calibrate(det, p))) for s, e, p in track.segments],
    )


def separated_scores(det: Detector, path: Path) -> dict[str, float]:
    """Calibrated AI probability of the separated vocals and of the instrumental."""
    from verasonus.features.separation import vocals_and_instrumental

    rate = det.cfg.audio.target_sample_rate
    original, original_rate = load_audio(path, mono=False)
    stems = vocals_and_instrumental(original, original_rate, device=det.device)
    out = {}
    for name, stem in stems.items():
        scored = score_audio(det, resample(stem, original_rate, rate), quiet_floor=0.1)
        if scored is not None:
            out[name] = scored.track
    return out


def analyze_file(path: Path, config: Config | None = None) -> AnalysisResult:
    """Run the full pipeline on one audio file.

    Raises ValueError when the audio is too short for one window.
    """
    det = load_detector()
    cfg = config or det.cfg
    rate = cfg.audio.target_sample_rate
    audio, _ = load_audio(path, target_sr=rate)
    duration = len(audio) / rate
    mix = score_audio(det, audio)
    if mix is None:
        need = cfg.windows.min_last_seconds
        raise ValueError(f"the clip is {duration:.1f} s long; at least {need:g} s are needed")

    thresholds = det.ensemble.get("stem_thresholds") or {}
    stems = separated_scores(det, path) if thresholds else {}
    stem_flags = [name for name, p in stems.items() if p >= thresholds[name]]
    bands = det.ensemble["verdict"]
    if mix.track >= bands["likely_ai"]:
        verdict = "likely_ai"
    elif stem_flags:
        verdict = "partly_ai"
    elif mix.track <= bands["likely_human"]:
        verdict = "likely_human"
    else:
        verdict = "uncertain"

    generator = None
    if det.generator_id is not None and verdict != "likely_human":
        gid = det.generator_id
        if is_unknown(gid["unknown_detector"], mix.features).mean() > 0.5:
            generator = GeneratorGuess(name="unknown", probability=1.0)
        else:
            probs = gid["classifier"].predict_proba(mix.features).mean(axis=0)
            best = int(np.argmax(probs))
            generator = GeneratorGuess(
                name=str(gid["classes"][best]), probability=float(probs[best])
            )

    top = explain_prediction(det.gb, mix.features, det.columns, det.ensemble["reference"], 5)
    contributions = [FeatureContribution(name=feature_label(n), contribution=c) for n, c in top]
    summary = {
        "likely_ai": "This track sounds AI-generated.",
        "partly_ai": "Parts of this track sound AI-generated.",
        "uncertain": "The models are not sure about this track.",
        "likely_human": "This track sounds human-made.",
    }[verdict]
    if stem_flags:
        summary += f" The separated {' and '.join(stem_flags)} score as AI."
    if verdict == "likely_human":
        toward = [c.name for c in contributions if c.contribution < 0][:2]
        if toward:
            summary += f" Features pointing to a human recording: {', '.join(toward)}."
    else:
        start, end, _ = max(mix.segments, key=lambda s: s[2])
        summary += f" The most AI-like part of the mix is {start:.0f}-{end:.0f} s."

    return AnalysisResult(
        ai_probability=mix.track,
        verdict=verdict,
        confidence=float(abs(mix.track - 0.5) * 2),
        duration_seconds=duration,
        segments=[
            Segment(start_seconds=s, end_seconds=e, ai_probability=p) for s, e, p in mix.segments
        ],
        stems=[StemScore(name=name, ai_probability=p) for name, p in stems.items()],
        generator=generator,
        explanation=Explanation(summary=summary, top_features=contributions),
        model_version=MODEL_VERSION,
    )
