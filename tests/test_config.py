import pytest

from verasonus.config import DATA_ROOT_ENV, REPO_ROOT, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.audio.target_sample_rate == 16_000
    assert cfg.windows.length_seconds == 10.0
    assert cfg.splits.group_column == "track_id"
    assert cfg.splits.stratify_columns == ("label",)
    assert sum(cfg.splits.ratios.values()) == pytest.approx(1.0)


def test_paths_are_absolute_and_derived_from_data_root():
    cfg = load_config()
    assert cfg.paths.data_root.is_absolute()
    assert cfg.paths.raw == cfg.paths.data_root / "raw"
    assert cfg.paths.metadata == cfg.paths.data_root / "metadata.csv"
    assert cfg.paths.reports == REPO_ROOT / "reports"


def test_dotted_overrides():
    cfg = load_config(overrides={"audio.target_sample_rate": 32_000, "windows.hop_seconds": 2.5})
    assert cfg.audio.target_sample_rate == 32_000
    assert cfg.windows.hop_seconds == 2.5


def test_data_root_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    assert load_config().paths.data_root == tmp_path


def test_invalid_values_raise():
    with pytest.raises(ValueError, match="ratios"):
        load_config(overrides={"splits.ratios": {"train": 0.9, "test": 0.3}})
    with pytest.raises(ValueError, match="positive"):
        load_config(overrides={"windows.hop_seconds": 0})


def test_unknown_key_raises():
    with pytest.raises(TypeError):
        load_config(overrides={"audio.sample_rte": 16_000})
