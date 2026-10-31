import numpy as np
import pytest

from verasonus.ensemble.aggregation import aggregate
from verasonus.ensemble.calibration import apply_temperature, fit_temperature, to_logits
from verasonus.ensemble.fusion import fit_stacker, stack, tune_weights, weighted_vote


def test_aggregate_averages_overlapping_windows():
    # Windows of 10 s every 5 s over a 20 s track: 0-10, 5-15, 10-20.
    track = aggregate(np.array([0.0, 0.5, 1.0]), np.array([0.0, 5.0, 10.0]), 10.0, 20.0)
    assert track.ai_probability == pytest.approx(0.5)
    assert track.segments == [
        (0.0, 5.0, 0.0),
        (5.0, 10.0, 0.25),
        (10.0, 15.0, 0.75),
        (15.0, 20.0, 1.0),
    ]
    top = aggregate(
        np.array([0.1, 0.9, 0.8]), np.array([0.0, 5.0, 10.0]), 10.0, 20.0, "top_k_mean", k=2
    )
    assert top.ai_probability == pytest.approx(0.85)


def test_aggregate_clips_the_last_window_to_the_track():
    track = aggregate(np.array([0.2]), np.array([0.0]), 10.0, 8.5)
    assert track.segments == [(0.0, 8.5, 0.2)]


def test_weighted_vote_and_weight_floor():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)]
    scores = {"good": y + rng.normal(0, 0.5, 400), "noise": rng.random(400)}
    assert weighted_vote(scores, {"good": 3, "noise": 1}) == pytest.approx(
        0.75 * scores["good"] + 0.25 * scores["noise"]
    )
    assert tune_weights(scores, y)["good"] >= 0.8
    floored = tune_weights(scores, y, min_weight=0.2)
    assert min(floored.values()) >= 0.2 and sum(floored.values()) == pytest.approx(1.0)


def test_temperature_softens_overconfident_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    true_logits = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 1, 2000)
    overconfident = 4 * true_logits
    temperature = fit_temperature(overconfident, y)
    assert 1.5 < temperature < 2.5  # the ideal temperature here is 2
    probs = apply_temperature(overconfident, temperature)
    assert ((probs > 0) & (probs < 1)).all()
    assert to_logits(np.array([0.5]))[0] == pytest.approx(0.0)


def test_stacker_learns_from_the_better_model():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(100), np.ones(100)]
    scores = {"a": y + rng.normal(0, 0.1, 200), "b": rng.random(200)}
    stacker = fit_stacker(scores, y)
    assert ((stack(stacker, scores) > 0.5) == y).mean() > 0.95
