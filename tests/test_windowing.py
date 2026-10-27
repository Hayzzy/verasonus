import numpy as np
import pytest

from verasonus.preprocessing.windowing import frame_windows

SR = 100  # a low sample rate keeps the arrays tiny


def ramp(seconds: float) -> np.ndarray:
    """Audio whose sample values encode their own position, to check window contents."""
    return np.arange(round(seconds * SR), dtype=np.float32) + 1


def test_exact_fit_has_no_padding():
    w = frame_windows(ramp(30), SR, length_seconds=10, hop_seconds=5)
    np.testing.assert_allclose(w.starts_seconds, [0, 5, 10, 15, 20])
    np.testing.assert_allclose(w.ends_seconds, [10, 15, 20, 25, 30])
    assert w.data.shape == (5, 1000)
    assert w.n_padded_samples == 0


def test_windows_match_the_source_audio():
    audio = ramp(30)
    w = frame_windows(audio, SR, length_seconds=10, hop_seconds=5)
    for window, start in zip(w.data, w.starts_seconds, strict=True):
        s = round(start * SR)
        np.testing.assert_array_equal(window, audio[s : s + 1000])


def test_tail_is_zero_padded():
    audio = ramp(32)
    w = frame_windows(audio, SR, length_seconds=10, hop_seconds=5)
    assert len(w) == 6
    assert w.starts_seconds[-1] == 25
    assert w.n_padded_samples == 300
    np.testing.assert_array_equal(w.data[-1, :700], audio[2500:])
    assert not w.data[-1, 700:].any()


def test_tail_dropped_without_padding():
    w = frame_windows(ramp(32), SR, length_seconds=10, hop_seconds=5, pad_last=False)
    assert len(w) == 5
    assert w.n_padded_samples == 0


def test_short_clip_becomes_one_padded_window_or_none():
    w = frame_windows(ramp(3), SR, length_seconds=10, hop_seconds=5, min_last_seconds=2)
    assert w.data.shape == (1, 1000)
    assert w.n_padded_samples == 700

    w = frame_windows(ramp(3), SR, length_seconds=10, hop_seconds=5, min_last_seconds=4)
    assert w.data.shape == (0, 1000)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"audio": np.zeros((2, 100))},
        {"length_seconds": 0},
        {"hop_seconds": -1},
        {"min_last_seconds": 20},
    ],
)
def test_invalid_arguments_raise(kwargs):
    args = {
        "audio": ramp(5),
        "sample_rate": SR,
        "length_seconds": 1.0,
        "hop_seconds": 0.5,
        **kwargs,
    }
    with pytest.raises(ValueError):
        frame_windows(**args)
