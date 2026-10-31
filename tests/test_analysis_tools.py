"""Clustering, the "unknown" detector, explanations and report outputs."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score

from verasonus.evaluation.reports import plot_confusion, plot_roc, save_figure, save_table
from verasonus.explain.importance import explain_prediction
from verasonus.models.unsupervised import cluster, embedding_map, fit_unknown_detector, is_unknown


def blobs(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0, 0], [8, 0, 0], [0, 8, 0]])
    labels = np.repeat(np.arange(3), 50)
    return centres[labels] + rng.normal(0, 0.5, (150, 3)), labels


def test_clusters_and_map_recover_blobs():
    x, labels = blobs()
    assert adjusted_rand_score(labels, cluster(x, 3)) > 0.95
    assert embedding_map(x).shape == (150, 2)


def test_unknown_detector_flags_far_points_only():
    x, _ = blobs()
    detector = fit_unknown_detector(x, contamination=0.01)
    assert is_unknown(detector, x).mean() < 0.05
    assert is_unknown(detector, np.full((5, 3), 40.0)).all()


def test_explanation_finds_the_feature_that_matters():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 4))
    y = (x[:, 2] > 0).astype(int)
    model = LogisticRegression().fit(x, y)
    track = np.array([[0.0, 0.0, 3.0, 0.0]])
    top = explain_prediction(model, track, ["a", "b", "c", "d"], np.zeros(4), top_k=2)
    assert top[0][0] == "c" and top[0][1] > 0.3


def test_reports_save_tables_and_figures(tmp_path):
    table = save_table(pd.DataFrame({"auc": [0.9]}, index=["svm"]), "scores", tmp_path)
    assert table.read_text().startswith(",auc")
    fig = plot_roc({"model": (np.array([0, 0.1, 1]), np.array([0, 0.8, 1]))})
    assert all(p.exists() for p in save_figure(fig, "roc", tmp_path))
    assert save_figure(plot_confusion(np.array([[5, 1], [0, 4]]), ["real", "AI"]), "cm", tmp_path)
