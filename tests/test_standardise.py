import numpy as np
import pytest
import soundfile as sf

from verasonus.config import AudioConfig
from verasonus.preprocessing.standardise import (
    effective_bandwidth_hz,
    interim_path,
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
    cfg = AudioConfig(shared_codec={"codec": "mp3", "bitrate_kbps": 192})
    with pytest.raises(NotImplementedError):  # reencode is still a Phase 1 stub
        standardise_file(src, tmp_path / "out.flac", cfg)


def test_effective_bandwidth_finds_the_highest_tone(tmp_path, tone):
    path = tmp_path / "two_tones.wav"
    sf.write(path, tone(500, 32_000) + tone(6_000, 32_000, amplitude=0.1), 32_000)
    assert effective_bandwidth_hz(path) == pytest.approx(6_000, abs=50)

    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(16_000, dtype=np.float32), 16_000)
    assert effective_bandwidth_hz(silent) == 0.0
