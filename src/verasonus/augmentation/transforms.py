"""Signal edits for the robustness suite and training-time augmentation (Phase 1).

A published detector was fooled by a 2-semitone pitch shift, so every edit here is also
used to build per-edit test reports (verasonus.evaluation.protocols).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal

import numpy as np

from verasonus.augmentation.codecs import reencode


def change_speed(audio: np.ndarray, sample_rate: int, factor: float) -> np.ndarray:
    """Play ``audio`` ``factor`` times faster, so pitch and tempo change together.

    This is what "slowed" (factor < 1) and "sped up" / nightcore (factor > 1) edits of songs
    do. Implemented as resampling: the samples are read as if recorded at a different rate.
    """
    import soxr

    if factor <= 0:
        raise ValueError("factor must be positive")
    if factor == 1:
        return np.asarray(audio, dtype=np.float32)
    out = soxr.resample(np.asarray(audio, dtype=np.float32), sample_rate * factor, sample_rate)
    return out.astype(np.float32)


def pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    """Shift pitch by ``semitones`` without changing duration (librosa.effects)."""
    import librosa

    out = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)
    return out.astype(np.float32)


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Speed up (rate > 1) or slow down (rate < 1) without changing pitch."""
    import librosa

    return librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)


def add_noise(
    audio: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    colour: Literal["white", "pink"] = "white",
) -> np.ndarray:
    """Add noise at a given signal-to-noise ratio in dB."""
    noise = rng.standard_normal(len(audio))
    if colour == "pink":  # 1/f power: scale the spectrum by 1/sqrt(f)
        spectrum = np.fft.rfft(noise)
        spectrum[1:] /= np.sqrt(np.arange(1, len(spectrum)))
        noise = np.fft.irfft(spectrum, n=len(audio))
    signal_power = float(np.mean(np.square(audio))) or 1e-12
    noise *= np.sqrt(signal_power / (10 ** (snr_db / 10)) / np.mean(np.square(noise)))
    return (audio + noise).astype(np.float32)


def normalise_loudness(audio: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
    """Normalise loudness to about ``target_lufs``.

    Approximation: plain RMS in dBFS stands in for BS.1770 loudness (no K-weighting or
    gating), which is close for full-band music. Peaks are limited to stay below clipping.
    """
    from verasonus.preprocessing.standardise import match_loudness

    return match_loudness(audio, target_lufs)


def apply_suite(
    audio: np.ndarray,
    sample_rate: int,
    suite: Mapping[str, Any],
    rng: np.random.Generator,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(edit_name, edited_audio)`` for every edit in configs/augment.yaml.

    Edit names are stable, e.g. "mp3_128", "pitch_+2", "noise_20db", "speed_0.8". Codecs
    other than MP3 and phone playback are skipped until ffmpeg support exists.
    """
    for kbps in suite.get("codecs", {}).get("mp3", []):
        yield f"mp3_{kbps}", reencode(audio, sample_rate, "mp3", int(kbps))
    for semitones in suite.get("pitch_shift_semitones", []):
        yield f"pitch_{semitones:+g}", pitch_shift(audio, sample_rate, semitones)
    for rate in suite.get("time_stretch_rates", []):
        yield f"stretch_{rate:g}", time_stretch(audio, rate)
    for factor in suite.get("speed_factors", []):
        yield f"speed_{factor:g}", change_speed(audio, sample_rate, factor)
    for snr in suite.get("noise_snr_db", []):
        yield f"noise_{snr}db", add_noise(audio, snr, rng)
    for lufs in suite.get("loudness_lufs", []):
        yield f"loudness_{lufs}", normalise_loudness(audio, sample_rate, lufs)


def random_training_edit(
    audio: np.ndarray, sample_rate: int, rng: np.random.Generator
) -> tuple[str, np.ndarray]:
    """One random everyday edit for training: speed change, MP3, or both.

    Speed factors 0.8-1.25 cover "slowed" and "sped up" uploads; MP3 at 32-192 kbps covers
    compression. Real and AI clips get edits with the same odds, so an edit never hints at
    the label.
    """
    name, out = [], np.asarray(audio, dtype=np.float32)
    choice = rng.integers(3)  # 0 speed, 1 mp3, 2 both
    if choice in (0, 2):
        factor = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25))))
        out = change_speed(out, sample_rate, factor)
        name.append(f"speed_{factor:.2f}")
    if choice in (1, 2):
        kbps = int(rng.choice([32, 48, 64, 96, 128, 192]))
        out = reencode(out, sample_rate, "mp3", kbps)
        name.append(f"mp3_{kbps}")
    return "+".join(name), out
