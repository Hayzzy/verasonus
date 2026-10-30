import numpy as np
import pytest

from verasonus.evaluation.metrics import (
    classification_metrics,
    confusion,
    equal_error_rate,
    expected_calibration_error,
    false_positive_rate,
)
from verasonus.features.handcrafted import (
    decoder_artefact_features,
    extract_handcrafted,
    high_band_energy,
)
from verasonus.models.classical import build_classical, train_classical

SR = 16_000


def test_handcrafted_keys_are_stable_and_finite(tone):
    a = extract_handcrafted(tone(440, SR, seconds=2.0), SR)
    b = extract_handcrafted(np.random.default_rng(0).normal(0, 0.1, SR * 3), SR)
    assert list(a) == list(b)
    assert len(a) == 107
    assert np.isfinite(list(a.values())).all()


def test_high_band_energy(tone):
    assert high_band_energy(tone(1_000, SR), SR) == pytest.approx(0.0, abs=1e-3)
    assert high_band_energy(tone(7_000, SR), SR) == pytest.approx(1.0, abs=1e-3)
    assert high_band_energy(np.zeros(SR, dtype=np.float32), SR) == 0.0


def test_artefacts_stand_out_for_evenly_spaced_peaks(tone):
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.05, SR * 2).astype(np.float32)
    combs = sum(tone(f, SR, seconds=2.0, amplitude=0.05) for f in range(500, 8000, 500))
    plain = decoder_artefact_features(noise, SR)
    comb = decoder_artefact_features(noise + combs, SR)
    assert comb["artefact_periodicity"] > plain["artefact_periodicity"] + 0.2
    assert comb["artefact_peakiness"] > plain["artefact_peakiness"]


def test_classification_metrics():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, np.array([0.1, 0.6, 0.4, 0.9]))
    assert m == pytest.approx(
        {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.75}
    )
    assert np.isnan(classification_metrics(np.ones(3), np.ones(3))["roc_auc"])


def test_eer_ece_and_fpr():
    y = np.array([0, 0, 0, 1, 1, 1])
    eer, _ = equal_error_rate(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]))
    assert eer == 0.0
    assert expected_calibration_error(y, y.astype(float)) == 0.0
    assert expected_calibration_error(np.array([0, 1]), np.array([0.9, 0.1])) == pytest.approx(0.9)
    assert false_positive_rate(y, np.array([0.6, 0.2, 0.1, 0.9, 0.9, 0.9])) == pytest.approx(1 / 3)
    np.testing.assert_array_equal(
        confusion(["a", "b", "b"], ["a", "a", "b"], ["a", "b"]), [[1, 0], [1, 1]]
    )


@pytest.mark.parametrize("name", ["svm", "random_forest", "gradient_boosting"])
def test_classical_models_learn_a_separable_problem(name):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    model, metrics = train_classical(name, x[:300], y[:300], x[300:], y[300:])
    assert metrics["roc_auc"] > 0.9
    assert model.predict_proba(x[:2]).shape == (2, 2)


def test_unknown_model():
    with pytest.raises(ValueError):
        build_classical("knn")
