"""Evaluation protocols (Phase 1-2).

1. Seen generators: held-out tracks from generators that appear in training.
2. Leave-one-generator-out: verasonus.data.splits.leave_one_generator_out; score the
   "test_unseen" rows together with the real rows of "test".
3. Robustness: re-run the test set through every edit in configs/augment.yaml.

Every report breaks results down per generator and per edit, and lists the false-positive
rate on real music separately. If accuracy looks perfect, assume a shortcut and run
shortcut_checks before celebrating.

The ``scores`` tables have one row per track with columns
track_id, label, generator, split, edit and ai_probability.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


def evaluate_seen(scores: pd.DataFrame) -> pd.DataFrame:
    """Metrics on the "test" split, overall and per generator."""
    raise NotImplementedError("Phase 1")


def evaluate_unseen(folds: Iterable[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """One row of metrics per held-out generator, plus the gap to the seen-generator score."""
    raise NotImplementedError("Phase 2")


def evaluate_robustness(scores_by_edit: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Metrics per edit (e.g. "mp3_64", "pitch_+2"), with the drop versus unedited audio."""
    raise NotImplementedError("Phase 2")


def shortcut_checks(metadata: pd.DataFrame) -> pd.DataFrame:
    """ROC-AUC of trivial predictors (duration, original sample rate, codec, bandwidth,
    loudness, leading silence) for real vs AI. A high value means a shortcut to remove."""
    raise NotImplementedError("Phase 1")
