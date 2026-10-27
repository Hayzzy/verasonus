import pandas as pd
import pytest

from verasonus.data.metadata import METADATA_COLUMNS, build_metadata, validate_metadata


@pytest.fixture
def table(data_root):
    """Metadata of the tiny data root: three FMA clips, then three FakeMusicCaps clips."""
    with pytest.warns(UserWarning):  # one FMA file is unreadable on purpose
        fma = build_metadata("fma", data_root / "raw" / "fma")
    fakemusiccaps = build_metadata("fakemusiccaps", data_root / "raw" / "fakemusiccaps")
    return pd.concat([fma, fakemusiccaps], ignore_index=True)


def test_build_metadata_returns_a_valid_table(table, data_root):
    assert tuple(table.columns) == METADATA_COLUMNS
    assert table["label"].value_counts().to_dict() == {"real": 3, "ai": 3}
    validate_metadata(table, data_root=data_root)


def test_build_metadata_rejects_an_empty_source(tmp_path):
    root = tmp_path / "raw" / "fakemusiccaps"
    root.mkdir(parents=True)
    with pytest.raises(ValueError, match="no audio"):
        build_metadata("fakemusiccaps", root)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("label", "fake", "label outside"),
        ("generator", "musicgen", 'label "real"'),
        ("group_id", "", "no group_id"),
        ("duration_seconds", 0.0, "duration_seconds"),
        ("path", "D:/verasonus-data/raw/fma/x.mp3", "absolute path"),
    ],
)
def test_validate_metadata_reports_bad_values(table, column, value, message):
    table.loc[0, column] = value  # row 0 is a real FMA clip
    with pytest.raises(ValueError, match=message):
        validate_metadata(table)


def test_validate_metadata_reports_duplicates_and_inconsistent_tracks(table):
    with pytest.raises(ValueError, match="share a clip_id"):
        validate_metadata(pd.concat([table, table.head(1)], ignore_index=True))
    table.loc[2, "track_id"] = table.loc[0, "track_id"]  # one track, two artists
    with pytest.raises(ValueError, match="more than one group_id"):
        validate_metadata(table)


def test_validate_metadata_checks_columns_and_files(table, data_root):
    with pytest.raises(ValueError, match="missing columns"):
        validate_metadata(table.drop(columns="group_id"))
    (data_root / table.loc[0, "path"]).unlink()
    with pytest.raises(ValueError, match="missing under"):
        validate_metadata(table, data_root=data_root)
