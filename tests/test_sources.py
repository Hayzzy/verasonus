import numpy as np
import pytest

from verasonus.data.sources import load_registry, scan
from verasonus.preprocessing.audio_io import save_audio


def test_registry_lists_every_source_with_a_licence():
    registry = load_registry()
    assert set(registry) == {
        "sonics",
        "fakemusiccaps",
        "melody_or_machine",
        "fma",
        "mtg_jamendo",
        "musdb18hq",
        "own_set",
    }
    for source in registry.values():
        assert source.licence and source.role and source.access
        assert set(source.labels) <= {"real", "ai", "partly_ai"}
    assert "098565" in registry["fma"].exclude


def test_scan_fma_reads_tracks_csv_and_skips_broken_files(data_root):
    with pytest.warns(UserWarning, match="000011"):
        records = list(scan("fma", data_root / "raw" / "fma"))
    assert [r.clip_id for r in records] == ["fma/000002", "fma/000005", "fma/000010"]
    first, second, third = records
    assert first.track_id == "fma/000002"
    assert first.group_id == second.group_id == "fma/artist/1"  # same artist, same group
    assert third.group_id == "fma/artist/6"
    assert (first.label, first.generator, first.genre) == ("real", "human", "Hip-Hop")
    assert first.path == "raw/fma/fma_small/000/000002.wav"
    assert (first.original_sample_rate, first.codec) == (16_000, "PCM_16")
    assert first.duration_seconds == pytest.approx(0.5)
    assert (first.licence, third.licence) == ("CC BY-NC 3.0", "unknown")


def test_scan_fma_needs_tracks_csv(data_root):
    (data_root / "raw" / "fma" / "fma_metadata" / "tracks.csv").unlink()
    with pytest.raises(FileNotFoundError, match="fma_metadata"):
        list(scan("fma", data_root / "raw" / "fma"))


def test_scan_fakemusiccaps_maps_folders_to_generators(data_root):
    records = list(scan("fakemusiccaps", data_root / "raw" / "fakemusiccaps"))
    assert [r.clip_id for r in records] == [  # the __MACOSX file is skipped
        "fakemusiccaps/musicgen/abc",
        "fakemusiccaps/audioldm2/-dash",
        "fakemusiccaps/audioldm2/abc",
    ]
    for r in records:
        assert r.track_id == r.group_id == r.clip_id
        assert (r.label, r.licence, r.genre) == ("ai", "CC BY-NC 4.0", "")


def test_scan_fakemusiccaps_rejects_unknown_folders(data_root):
    root = data_root / "raw" / "fakemusiccaps"
    save_audio(root / "new_model" / "x.wav", np.zeros(1600, np.float32), 16_000)
    with pytest.raises(ValueError, match="new_model"):
        list(scan("fakemusiccaps", root))


def test_scan_rejects_unknown_and_unimplemented_sources(tmp_path):
    with pytest.raises(KeyError):
        scan("nope", tmp_path)
    with pytest.raises(NotImplementedError):
        scan("sonics", tmp_path)
