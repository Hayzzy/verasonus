"""Handcrafted features for the classical baselines [aspects 2 and 4] (Phase 1).

One fixed-length vector per window, so SVM, random forest and gradient boosting train in
minutes and can be explained with feature importance.
"""

from __future__ import annotations

import librosa
import numpy as np

N_FFT = 2048
HOP = 512


def _stats(name: str, values: np.ndarray) -> dict[str, float]:
    """Mean and standard deviation of each row of ``values`` (one row per coefficient)."""
    values = np.atleast_2d(values)
    out = {}
    for i, row in enumerate(values):
        key = name if len(values) == 1 else f"{name}_{i}"
        out[f"{key}_mean"] = float(np.mean(row))
        out[f"{key}_std"] = float(np.std(row))
    return out


def mfcc_stats(audio: np.ndarray, sample_rate: int, n_mfcc: int = 20) -> dict[str, float]:
    """Mean and standard deviation of MFCCs and their deltas."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP)
    return {**_stats("mfcc", mfcc), **_stats("mfcc_delta", librosa.feature.delta(mfcc))}


def spectral_stats(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Spectral centroid, bandwidth, roll-off, flatness, contrast and zero-crossing rate."""
    spec = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP))
    feature = librosa.feature
    return {
        **_stats("centroid", feature.spectral_centroid(S=spec, sr=sample_rate)),
        **_stats("bandwidth", feature.spectral_bandwidth(S=spec, sr=sample_rate)),
        **_stats("rolloff", feature.spectral_rolloff(S=spec, sr=sample_rate)),
        **_stats("flatness", feature.spectral_flatness(S=spec)),
        **_stats("contrast", feature.spectral_contrast(S=spec, sr=sample_rate)),
        **_stats("zcr", feature.zero_crossing_rate(audio, frame_length=N_FFT, hop_length=HOP)),
    }


def high_band_energy(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 6000.0) -> float:
    """Share of energy above ``cutoff_hz``. Useful, but check it is not a bandwidth shortcut."""
    power = np.abs(np.fft.rfft(audio)) ** 2
    total = power.sum()
    if total == 0:
        return 0.0
    freqs = np.fft.rfftfreq(len(audio), 1 / sample_rate)
    return float(power[freqs >= cutoff_hz].sum() / total)


def decoder_artefact_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Periodic spectral peaks left by the upsampling layers of neural decoders.

    Afchar et al. (2024) show these artefacts make detection "easy" in the lab, and that
    simple edits can remove them, so report robustness for these features separately.

    The average log spectrum is detrended (its smooth shape removed), leaving narrow peaks.
    Upsampling artefacts repeat at a fixed frequency spacing, so the autocorrelation of the
    residual shows a strong peak at that spacing.
    """
    spec = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP))
    log_mean = np.log(spec.mean(axis=1) + 1e-10)
    residual = log_mean - np.convolve(log_mean, np.ones(31) / 31, mode="same")
    residual = residual[15:-15]  # drop the edges, where the smoothing is lopsided
    residual = residual - residual.mean()
    energy = float(np.dot(residual, residual))
    if energy == 0:
        return {"artefact_peakiness": 0.0, "artefact_periodicity": 0.0}
    auto = np.correlate(residual, residual, mode="full")[len(residual) :] / energy
    return {
        "artefact_peakiness": float(residual.max() / (residual.std() + 1e-10)),
        "artefact_periodicity": float(auto[2:200].max()),  # skip neighbouring bins
    }


def extract_handcrafted(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """All handcrafted features for one window, with a stable key order."""
    audio = np.asarray(audio, dtype=np.float32)
    return {
        **mfcc_stats(audio, sample_rate),
        **spectral_stats(audio, sample_rate),
        "high_band_energy": high_band_energy(audio, sample_rate),
        **decoder_artefact_features(audio, sample_rate),
    }
