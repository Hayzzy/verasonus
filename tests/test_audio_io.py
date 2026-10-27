import numpy as np
import pytest
import soundfile as sf

from verasonus.preprocessing.audio_io import audio_info, load_audio, resample, save_audio, to_mono


def test_load_downmixes_and_resamples(tmp_path, tone, peak_hz):
    left = tone(440, sample_rate=44_100, seconds=2.0)
    right = tone(440, sample_rate=44_100, seconds=2.0, amplitude=0.25)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([left, right], axis=1), 44_100)

    audio, sr = load_audio(path, target_sr=16_000)

    assert sr == 16_000
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert abs(len(audio) - 32_000) <= 1
    assert peak_hz(audio, sr) == pytest.approx(440, abs=2)


def test_load_keeps_channels_when_not_mono(tmp_path, tone):
    stereo = np.stack([tone(220, 22_050), tone(330, 22_050)], axis=1)
    path = tmp_path / "stereo.flac"
    sf.write(path, stereo, 22_050)

    audio, sr = load_audio(path, mono=False)

    assert sr == 22_050
    assert audio.shape == (2, 22_050)


def test_to_mono_averages_channels():
    stereo = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(to_mono(stereo), [0.5, 0.5])
    with pytest.raises(ValueError):
        to_mono(np.zeros((1, 2, 3)))


def test_resample_preserves_duration_and_pitch(tone, peak_hz):
    x = tone(1000, sample_rate=48_000, seconds=1.0)
    down = resample(x, 48_000, 16_000)
    up = resample(down, 16_000, 48_000)

    assert len(down) == 16_000
    assert abs(len(up) - 48_000) <= 1
    assert peak_hz(down, 16_000) == pytest.approx(1000, abs=2)


def test_resample_multichannel_and_noop(tone):
    stereo = np.stack([tone(440, 44_100), tone(880, 44_100)])
    assert resample(stereo, 44_100, 16_000).shape == (2, 16_000)
    mono = tone(440, 16_000)
    assert resample(mono, 16_000, 16_000) is not None
    with pytest.raises(ValueError):
        resample(mono, 0, 16_000)


def test_save_and_info_round_trip(tmp_path, tone):
    x = tone(440, sample_rate=16_000, seconds=1.5)
    path = save_audio(tmp_path / "nested" / "clip.flac", x, 16_000, subtype="PCM_16")

    info = audio_info(path)
    assert (info.sample_rate, info.channels, info.frames) == (16_000, 1, 24_000)
    assert info.duration_seconds == pytest.approx(1.5)
    assert (info.format, info.subtype) == ("FLAC", "PCM_16")

    loaded, _ = load_audio(path)
    np.testing.assert_allclose(loaded, x, atol=1e-3)


def test_save_clips_out_of_range_samples(tmp_path):
    path = save_audio(tmp_path / "loud.wav", np.array([2.0, -2.0, 0.5], dtype=np.float32), 8_000)
    loaded, _ = load_audio(path)
    assert loaded.max() <= 1.0 and loaded.min() >= -1.0


def test_mp3_decoding(tmp_path, tone, peak_hz):
    if "MP3" not in sf.available_formats():
        pytest.skip("this libsndfile build has no MP3 support")
    path = tmp_path / "clip.mp3"
    sf.write(path, tone(440, sample_rate=44_100, seconds=2.0), 44_100, format="MP3")

    audio, sr = load_audio(path, target_sr=16_000)

    assert sr == 16_000
    assert peak_hz(audio, sr) == pytest.approx(440, abs=3)
