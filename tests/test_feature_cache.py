import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from verasonus.config import load_config
from verasonus.features.cache import (
    build_feature_table,
    clip_windows,
    load_feature_table,
    save_feature_table,
)

SR = 16_000


def stand_in_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """A stand-in for the real feature functions: length and trailing silence."""
    nonzero = np.flatnonzero(audio)
    return {"seconds": len(audio) / sample_rate, "silent_tail": len(audio) - 1 - nonzero[-1]}


@pytest.fixture
def setup(tmp_path, tone):
    cfg = load_config(overrides={"paths.data_root": str(tmp_path)})
    clips = {"real/a": ("raw/fma/a.mp3", 29.98, "real"), "ai/b": ("raw/fmc/b.wav", 10.2, "ai")}
    rows = []
    for clip_id, (path, seconds, label) in clips.items():
        audio = tone(440, SR, seconds=seconds) + 0.1  # offset so no sample is exactly zero
        dst = tmp_path / path.replace("raw/", "interim/").rsplit(".", 1)[0]
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(f"{dst}.flac", audio, SR, subtype="PCM_16")
        rows.append(
            {
                "clip_id": clip_id,
                "path": path,
                "label": label,
                "split": "train",
                "generator": "x",
                "group_id": clip_id,
                "source_dataset": "s",
            }
        )
    return cfg, pd.DataFrame(rows)


def test_padded_tail_is_trimmed(setup):
    cfg, df = setup
    windows = clip_windows(cfg.paths.data_root / "interim/fma/a.flac", cfg)
    assert [start for start, _ in windows] == [0, 5, 10, 15, 20]
    assert len(windows[-1][1]) == pytest.approx(9.98 * SR, abs=1)
    assert all(len(audio) == 10 * SR for _, audio in windows[:-1])


def test_feature_table_has_one_row_per_window(setup):
    cfg, df = setup
    windows, features, columns = build_feature_table(df, cfg, stand_in_features)

    assert columns == ["seconds", "silent_tail"]
    assert features.dtype == np.float32 and features.shape == (6, 2)
    assert windows["clip_id"].tolist() == ["real/a"] * 5 + ["ai/b"]
    assert windows["window"].tolist() == [0, 1, 2, 3, 4, 0]
    assert set(windows["label"]) == {"real", "ai"}
    assert (features[:, 1] == 0).all()  # no window ends in padding
    assert windows["real_seconds"].iloc[4] == pytest.approx(9.98, abs=1e-3)


def test_save_and_load_round_trip(setup, tmp_path):
    cfg, df = setup
    table = build_feature_table(df, cfg, stand_in_features)
    save_feature_table(tmp_path / "out", *table)
    windows, features, columns = load_feature_table(tmp_path / "out")
    pd.testing.assert_frame_equal(windows, table[0])
    np.testing.assert_array_equal(features, table[1])
    assert columns == table[2]


def test_feature_names_must_match(setup):
    cfg, df = setup
    calls = iter(range(100))

    def unstable(audio, sample_rate):
        return {f"f{next(calls)}": 0.0}

    with pytest.raises(ValueError, match="differ"):
        build_feature_table(df, cfg, unstable)


def test_handcrafted_features_fill_the_table(setup):
    from verasonus.features.handcrafted import extract_handcrafted

    cfg, df = setup
    windows, features, columns = build_feature_table(df, cfg, extract_handcrafted)
    assert features.shape == (6, len(columns))
    assert np.isfinite(features).all()
