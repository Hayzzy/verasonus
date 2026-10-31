"""CNN tests. Skipped where torch is not installed (CI installs no deep learning stack)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verasonus.explain.saliency import saliency_map  # noqa: E402
from verasonus.models.deep import build_cnn, predict_deep, spec_augment, train_deep  # noqa: E402


def test_cnn_takes_any_spectrogram_size():
    model = build_cnn().eval()
    for shape in [(2, 1, 128, 397), (3, 1, 64, 100)]:
        assert model(torch.zeros(shape)).shape == (shape[0], 2)
    assert sum(p.numel() for p in model.parameters()) < 2_000_000


def test_spec_augment_hides_bands_with_the_example_mean():
    torch.manual_seed(0)
    x = torch.randn(8, 1, 32, 50)
    out = spec_augment(x, max_freq=8, max_time=10)
    assert out.shape == x.shape
    changed = out != x
    assert changed.any()
    means = x.mean(dim=(2, 3), keepdim=True).expand_as(x)
    assert torch.equal(out[changed], means[changed])


def test_cnn_learns_a_separable_problem():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(160, 16, 24)).astype(np.float16)
    y = np.r_[np.zeros(80, int), np.ones(80, int)]
    x[y == 1, 4:6, :] += 3  # "AI" examples carry a bright band
    rows = rng.permutation(160)
    train, val = rows[:120], rows[120:]

    model = build_cnn()
    history = train_deep(
        model,
        x,
        y,
        train,
        val,
        epochs=6,
        lr=3e-3,
        batch_size=16,
        augment=False,
        device="cpu",
        log=lambda _: None,
    )
    assert len(history["val_roc_auc"]) <= 6
    scores = predict_deep(model, x, val, device="cpu")
    assert scores.shape == (40,) and ((scores >= 0) & (scores <= 1)).all()
    assert max(history["val_roc_auc"]) > 0.95


def test_gradcam_matches_the_input_and_is_scaled():
    torch.manual_seed(0)
    spec = np.random.default_rng(0).normal(size=(32, 48)).astype(np.float32)
    cam = saliency_map(build_cnn(), spec)
    assert cam.shape == spec.shape
    assert cam.min() >= 0 and cam.max() <= 1
