import librosa
import numpy as np
import pytest

from verasonus.config import load_config
from verasonus.features.spectrograms import mel_settings, mel_spectrogram, window_mel

SR = 16_000


def test_mel_spectrogram_shape_and_peak(tone):
    mel = mel_spectrogram(tone(1000, SR, seconds=2.0), SR, n_mels=64, n_fft=1024, hop_length=320)
    assert mel.shape == (64, 1 + (2 * SR - 1024) // 320)
    centres = librosa.mel_frequencies(64 + 2, fmax=SR / 2)[1:-1]
    expected = int(np.abs(centres - 1000).argmin())
    assert abs(int(mel.mean(axis=1).argmax()) - expected) <= 1


def test_settings_stop_at_the_lowpass_and_avoid_padding():
    settings = mel_settings(load_config())
    assert settings["fmax"] == 7000.0
    assert settings["n_frames"] == 1 + (8 * SR - 1024) // 320
    with pytest.raises(ValueError, match="min_last_seconds"):
        mel_settings(load_config(overrides={"spectrogram.seconds": 9.0}))


def test_window_mel_has_a_fixed_size(tone):
    settings = mel_settings(load_config())
    long = window_mel(tone(440, SR, seconds=10.0), SR, **settings)
    short = window_mel(tone(440, SR, seconds=8.0), SR, **settings)
    assert long.shape == short.shape == (settings["n_mels"], settings["n_frames"])
    assert long.dtype == np.float16
    with pytest.raises(ValueError, match="samples"):
        window_mel(tone(440, SR, seconds=7.0), SR, **settings)
