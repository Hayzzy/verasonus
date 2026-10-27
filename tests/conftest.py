"""Shared fixtures. All audio is synthesised in memory: tests never download anything."""

import numpy as np
import pandas as pd
import pytest

from verasonus.preprocessing.audio_io import save_audio


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


@pytest.fixture
def data_root(tmp_path):
    """A tiny data root laid out like the real FMA small and FakeMusicCaps downloads."""
    clip = make_tone(seconds=0.5)
    fma = tmp_path / "raw" / "fma"
    for name in ("000002", "000005", "000010", "098565"):  # 098565 is on the exclude list
        save_audio(fma / "fma_small" / name[:3] / f"{name}.wav", clip, 16_000)
    (fma / "fma_small" / "000" / "000011.wav").write_bytes(b"not audio")
    (fma / "fma_small" / "README.txt").write_text("not audio")
    tracks = pd.DataFrame(
        {
            ("set", "subset"): ["small"] * 5,
            ("artist", "id"): [1, 1, 6, 7, 8],
            ("track", "genre_top"): ["Hip-Hop", "Hip-Hop", "Pop", "Rock", "Folk"],
            ("track", "license"): ["CC BY-NC 3.0", "CC BY-NC 3.0", None, "CC BY 4.0", "CC BY 4.0"],
        },
        index=pd.Index([2, 5, 10, 11, 98565], name="track_id"),
    )
    (fma / "fma_metadata").mkdir()
    tracks.to_csv(fma / "fma_metadata" / "tracks.csv")  # same two-row header as FMA's file

    fakemusiccaps = tmp_path / "raw" / "fakemusiccaps"
    for folder, name in [("MusicGen_medium", "abc"), ("audioldm2", "abc"), ("audioldm2", "-dash")]:
        save_audio(fakemusiccaps / folder / f"{name}.wav", clip, 16_000)
    junk = fakemusiccaps / "__MACOSX" / "audioldm2" / "._abc.wav"  # macOS metadata, not audio
    junk.parent.mkdir(parents=True)
    junk.write_bytes(b"\x00\x05\x16\x07")
    return tmp_path
