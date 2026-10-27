import pandas as pd
import pytest

from verasonus.data.splits import (
    UNSEEN,
    generator_holdout_split,
    group_split,
    leave_one_generator_out,
)

GENERATORS = [("human", "real"), ("suno", "ai"), ("udio", "ai"), ("musicgen", "ai")]


def make_table(tracks_per_generator: int = 20, windows_per_track: int = 4) -> pd.DataFrame:
    rows = []
    for generator, label in GENERATORS:
        for t in range(tracks_per_generator):
            track = f"{generator}-{t:03d}"
            for w in range(windows_per_track):
                rows.append(
                    {
                        "clip_id": f"{track}-w{w}",
                        "track_id": track,
                        "label": label,
                        "generator": generator,
                    }
                )
    return pd.DataFrame(rows)


def tracks_per_split(df: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    return df.assign(split=split).drop_duplicates("track_id").groupby(["label", "split"]).size()


def test_windows_of_a_track_share_one_split():
    df = make_table()
    split = group_split(df)
    assert (df.assign(split=split).groupby("track_id")["split"].nunique() == 1).all()


def test_ratios_respected_within_each_label():
    df = make_table()
    counts = tracks_per_split(df, group_split(df))
    assert counts[("real", "train")] == 16 and counts[("real", "test")] == 2
    assert counts[("ai", "train")] == 48 and counts[("ai", "val")] == 6


def test_same_seed_same_split():
    df = make_table()
    pd.testing.assert_series_equal(group_split(df, seed=1), group_split(df, seed=1))
    assert not group_split(df, seed=1).equals(group_split(df, seed=2))


def test_generator_holdout_goes_only_to_test_unseen():
    df = make_table()
    split = generator_holdout_split(df, ["udio"])
    assert (split[df["generator"] == "udio"] == UNSEEN).all()
    assert UNSEEN not in set(split[df["generator"] != "udio"])
    assert {"train", "val", "test"} <= set(split[df["label"] == "real"])


def test_leave_one_generator_out_yields_one_fold_per_ai_generator():
    df = make_table()
    folds = dict(leave_one_generator_out(df))
    assert set(folds) == {"suno", "udio", "musicgen"}
    for generator, split in folds.items():
        assert (split[df["generator"] == generator] == UNSEEN).all()


def test_errors_are_explicit():
    df = make_table()
    with pytest.raises(ValueError, match="sum to 1"):
        group_split(df, {"train": 0.5, "test": 0.4})
    with pytest.raises(KeyError):
        group_split(df.drop(columns="label"))
    with pytest.raises(ValueError, match="not in the table"):
        generator_holdout_split(df, ["riffusion"])

    mixed = df.copy()
    mixed.loc[0, "generator"] = "udio"  # one window of a suno track now claims udio
    with pytest.raises(ValueError, match="mix"):
        generator_holdout_split(mixed, ["udio"])
