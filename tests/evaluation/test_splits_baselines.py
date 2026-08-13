"""Validation-machinery invariants — walk-forward splits and baselines.

The splits are the leakage guard for the whole project: if train and test overlap,
every backtest number is inflated. These tests pin the purge behaviour and the
baseline comparisons the model must beat.
"""
import numpy as np
import pandas as pd
import pytest

from src.forecast_engine.evaluation.splits import walk_forward_splits
from src.forecast_engine.evaluation.baselines import (
    beats_baseline,
    persistence_volatility_baseline,
    majority_class_baseline,
)


@pytest.fixture
def dated_frame():
    dates = pd.bdate_range("2020-01-01", periods=200)
    rows = []
    for sym in ("A", "B"):
        for d in dates:
            rows.append({"symbol": sym, "date": d, "adj_close": 100.0})
    return pd.DataFrame(rows)


def test_splits_are_non_overlapping_and_ordered(dated_frame):
    """Train indices must all precede test indices — no leakage across the fold."""
    for train_idx, test_idx in walk_forward_splits(dated_frame, purge_days=5, n_splits=4):
        train_max = dated_frame.loc[train_idx, "date"].max()
        test_min = dated_frame.loc[test_idx, "date"].min()
        assert train_max < test_min, "train must end before test begins"


def test_purge_gap_enforced(dated_frame):
    """The gap between train end and test start must be at least purge_days."""
    purge = 10
    for train_idx, test_idx in walk_forward_splits(dated_frame, purge_days=purge, n_splits=4):
        train_max = dated_frame.loc[train_idx, "date"].max()
        test_min = dated_frame.loc[test_idx, "date"].min()
        gap_days = (test_min - train_max).days
        assert gap_days >= purge, f"purge gap {gap_days} < required {purge}"


def test_negative_purge_rejected(dated_frame):
    with pytest.raises(ValueError):
        list(walk_forward_splits(dated_frame, purge_days=-1))


def test_missing_date_column_rejected():
    with pytest.raises(ValueError):
        list(walk_forward_splits(pd.DataFrame({"x": [1, 2, 3]}), purge_days=5))


def test_too_few_dates_rejected():
    tiny = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=3)})
    with pytest.raises(ValueError):
        list(walk_forward_splits(tiny, purge_days=1, n_splits=10))


def test_beats_baseline_direction():
    # error metric: lower is better
    assert beats_baseline(0.02, 0.03, lower_is_better=True)
    assert not beats_baseline(0.03, 0.02, lower_is_better=True)
    # skill metric: higher is better
    assert beats_baseline(0.10, 0.05, lower_is_better=False)
    assert not beats_baseline(0.05, 0.10, lower_is_better=False)


def test_persistence_baseline_is_trailing_std(dated_frame):
    """Persistence baseline is a trailing rolling std — non-negative, NaN in warmup."""
    # give it real price variation
    rng = np.random.default_rng(0)
    dated_frame = dated_frame.copy()
    dated_frame["adj_close"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dated_frame))))
    base = persistence_volatility_baseline(dated_frame, window=5)
    valid = base.dropna()
    assert (valid >= 0).all(), "volatility must be non-negative"
    assert len(valid) > 0


def test_majority_class_baseline():
    up_heavy = pd.Series([1, 1, 1, 0])
    assert majority_class_baseline(up_heavy, 3).tolist() == [1, 1, 1]
    down_heavy = pd.Series([0, 0, 0, 1])
    assert majority_class_baseline(down_heavy, 2).tolist() == [0, 0]
