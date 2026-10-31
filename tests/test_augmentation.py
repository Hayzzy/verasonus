import numpy as np
import pytest

from verasonus.augmentation.codecs import reencode
from verasonus.augmentation.transforms import (
    add_noise,
    apply_suite,
    change_speed,
    random_training_edit,
)

SR = 16_000


def test_mp3_round_trip_keeps_length_and_alignment(tone):
    audio = tone(440, SR, seconds=2.0)
    out = reencode(audio, SR, "mp3", 32)
    assert out.shape == audio.shape and out.dtype == np.float32
    assert np.corrcoef(audio[2000:-2000], out[2000:-2000])[0, 1] > 0.95
    with pytest.raises(NotImplementedError):
        reencode(audio, SR, "aac", 128)


def test_change_speed_moves_pitch_and_length(tone, peak_hz):
    audio = tone(1000, SR, seconds=2.0)
    slowed = change_speed(audio, SR, 0.8)
    assert len(slowed) == pytest.approx(len(audio) / 0.8, abs=2)
    assert peak_hz(slowed, SR) == pytest.approx(800, abs=10)


def test_noise_hits_the_requested_snr(tone):
    audio = tone(440, SR)
    noisy = add_noise(audio, 10, np.random.default_rng(0))
    snr = 10 * np.log10(np.mean(audio**2) / np.mean((noisy - audio) ** 2))
    assert snr == pytest.approx(10, abs=0.1)


def test_suite_names_and_training_edits(tone):
    audio = tone(440, SR)
    suite = {"codecs": {"mp3": [64], "aac": [128]}, "speed_factors": [0.8], "noise_snr_db": [20]}
    names = [name for name, _ in apply_suite(audio, SR, suite, np.random.default_rng(0))]
    assert names == ["mp3_64", "speed_0.8", "noise_20db"]
    rng = np.random.default_rng(1)
    edits = {random_training_edit(audio, SR, rng)[0].split("_")[0] for _ in range(20)}
    assert edits == {"speed", "mp3"}
