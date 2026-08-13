"""Feature-parity invariants — the train/serve divergence guard.

This is the guard for the silent bug class the project actually hit: a feature that
computes differently at train vs serve time (cumulative-from-inception features)
breaks parity invisibly. These tests confirm the detector flags a real mismatch and
passes identical frames.
"""
import numpy as np
import pandas as pd
import pytest

from src.forecast_engine.features.feature_parity import (
    parity_mismatches,
    assert_feature_parity,
)


def _frame(values):
    return pd.DataFrame({
        "symbol": ["A"] * len(values),
        "date": pd.bdate_range("2020-01-01", periods=len(values)),
        "feat": values,
    })


def test_identical_frames_have_no_mismatches():
    f = _frame([1.0, 2.0, 3.0, 4.0])
    out = parity_mismatches(f, f.copy(), feature_columns=["feat"])
    assert len(out) == 0, "identical frames must show zero mismatches"


def test_assert_parity_passes_on_identical():
    f = _frame([1.0, 2.0, 3.0])
    # should not raise
    assert_feature_parity(f, f.copy(), feature_columns=["feat"])


def test_value_divergence_is_flagged():
    train = _frame([1.0, 2.0, 3.0, 4.0])
    serve = _frame([1.0, 2.0, 3.5, 4.0])  # one differing value
    out = parity_mismatches(train, serve, feature_columns=["feat"])
    assert len(out) >= 1, "a differing value must be flagged"


def test_assert_parity_raises_on_divergence():
    train = _frame([1.0, 2.0, 3.0])
    serve = _frame([1.0, 9.9, 3.0])
    with pytest.raises(AssertionError):
        assert_feature_parity(train, serve, feature_columns=["feat"])


def test_value_vs_nan_is_a_mismatch():
    train = _frame([1.0, 2.0, 3.0])
    serve = _frame([1.0, np.nan, 3.0])
    out = parity_mismatches(train, serve, feature_columns=["feat"])
    assert len(out) >= 1, "a value vs NaN in the same slot must be a mismatch"


def test_nan_in_same_place_is_agreement():
    train = _frame([1.0, np.nan, 3.0])
    serve = _frame([1.0, np.nan, 3.0])
    out = parity_mismatches(train, serve, feature_columns=["feat"])
    assert len(out) == 0, "NaN in the same slot is agreement, not a mismatch"


def test_no_overlap_raises():
    train = _frame([1.0, 2.0])
    serve = train.copy()
    serve["date"] = pd.bdate_range("2021-01-01", periods=2)  # disjoint dates
    with pytest.raises(ValueError):
        parity_mismatches(train, serve, feature_columns=["feat"])
