"""Shared fixtures. All audio is synthesised in memory: tests never download anything."""

import numpy as np
import pytest


def make_tone(
    freq_hz: float = 440.0, sample_rate: int = 16_000, seconds: float = 1.0, amplitude: float = 0.5
) -> np.ndarray:
    """A float32 sine wave."""
    t = np.arange(round(sample_rate * seconds)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def dominant_frequency(audio: np.ndarray, sample_rate: int) -> float:
    """Frequency (Hz) of the strongest spectral peak."""
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return float(np.fft.rfftfreq(len(audio), 1 / sample_rate)[spectrum.argmax()])


@pytest.fixture
def tone():
    return make_tone


@pytest.fixture
def peak_hz():
    return dominant_frequency
