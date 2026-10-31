import numpy as np
import pytest
import soundfile as sf

from verasonus.config import AudioConfig, load_config
from verasonus.features.handcrafted import high_band_energy
from verasonus.preprocessing.standardise import (
    effective_bandwidth_hz,
    interim_path,
    lowpass,
    match_loudness,
    standardise_file,
)


def test_interim_path_mirrors_raw():
    cfg = AudioConfig()
    assert (
        interim_path("raw/fma/fma_small/000/000002.mp3", cfg)
        == "interim/fma/fma_small/000/000002.flac"
    )
    with pytest.raises(ValueError):
        interim_path("interim/fma/000002.flac", cfg)
    with pytest.raises(ValueError):
        interim_path("raw/fma/000002.mp3", AudioConfig(storage_format="OGG"))


def test_standardise_writes_mono_flac_at_target_rate(tmp_path, tone, peak_hz):
    stereo = np.stack([tone(440, 44_100, seconds=2.0), tone(440, 44_100, seconds=2.0)], axis=1)
    src = tmp_path / "in.wav"
    sf.write(src, stereo, 44_100, subtype="FLOAT")
    dst = tmp_path / "out" / "in.flac"

    info = standardise_file(src, dst, AudioConfig())

    assert (info.sample_rate, info.channels, info.format, info.subtype) == (
        16_000,
        1,
        "FLAC",
        "PCM_16",
    )
    assert info.duration_seconds == pytest.approx(2.0, abs=1e-3)
    audio, sr = sf.read(dst)
    assert peak_hz(audio, sr) == pytest.approx(440, abs=2)


def test_shared_codec_goes_through_reencode(tmp_path, tone):
    src = tmp_path / "in.wav"
    sf.write(src, tone(440, 16_000), 16_000)
    cfg = AudioConfig(shared_codec={"codec": "mp3", "bitrate_kbps": 32})
    info = standardise_file(src, tmp_path / "out.flac", cfg)
    assert info.frames == 16_000  # same length after the MP3 round trip


def test_long_clips_are_cut_to_one_excerpt(tmp_path, tone):
    src = tmp_path / "song.wav"
    sf.write(src, tone(440, 16_000, seconds=90.0), 16_000)
    cfg = AudioConfig(long_clip_seconds=60, excerpt_seconds=20)
    assert standardise_file(src, tmp_path / "out.flac", cfg).frames == 20 * 16_000
    short = tmp_path / "clip.wav"
    sf.write(short, tone(440, 16_000, seconds=30.0), 16_000)
    assert standardise_file(short, tmp_path / "out2.flac", cfg).frames == 30 * 16_000


def test_effective_bandwidth_finds_the_highest_tone(tmp_path, tone):
    path = tmp_path / "two_tones.wav"
    sf.write(path, tone(500, 32_000) + tone(6_000, 32_000, amplitude=0.1), 32_000)
    assert effective_bandwidth_hz(path) == pytest.approx(6_000, abs=50)

    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(16_000, dtype=np.float32), 16_000)
    assert effective_bandwidth_hz(silent) == 0.0


def test_lowpass_removes_the_top_band(tone, peak_hz):
    mix = tone(1_000, 16_000) + tone(7_600, 16_000)
    out = lowpass(mix, 16_000, 7_000)
    assert high_band_energy(out, 16_000, cutoff_hz=7_300) < 1e-4
    assert peak_hz(out, 16_000) == pytest.approx(1_000, abs=2)


def test_match_loudness_sets_rms_without_clipping(tone):
    quiet = tone(440, 16_000, amplitude=0.01)
    out = match_loudness(quiet, -20.0)
    assert 20 * np.log10(np.sqrt(np.mean(out**2))) == pytest.approx(-20.0, abs=0.01)
    spiky = np.zeros(16_000, dtype=np.float32)
    spiky[0] = 0.5  # one loud click: the -20 dB target would need a peak far above 1
    assert np.abs(match_loudness(spiky, -20.0)).max() == pytest.approx(0.99)
    assert not match_loudness(np.zeros(10, dtype=np.float32), -20.0).any()


def test_lowpass_setting_is_validated():
    with pytest.raises(ValueError):
        load_config(overrides={"audio.lowpass_hz": 9_000})
