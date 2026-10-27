from verasonus.data.sources import load_registry


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
