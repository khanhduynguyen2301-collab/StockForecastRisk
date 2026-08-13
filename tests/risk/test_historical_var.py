"""Unconditional historical VaR — still used for the backtest comparison.

Superseded for serving by filtered_historical_var, but retained and exercised by
backtest.py, so its core functions must stay correct. These pin horizon-return
construction and VaR/CVaR coherence.
"""
import numpy as np
import pytest

from src.forecast_engine.risk.historical_var import (
    horizon_returns,
    historical_var,
    historical_cvar,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(7)
    return rng.normal(0.0003, 0.015, 3000)


def test_horizon_returns_length():
    """Overlapping windows of length h over n points give n-h+1 returns."""
    daily = np.full(100, 0.001)
    hr = horizon_returns(daily, horizon=5)
    assert len(hr) == 100 - 5 + 1


def test_horizon_returns_compounding():
    """A constant daily log return r over h days compounds to exp(h*r)-1."""
    r = 0.01
    daily = np.full(50, r)
    hr = horizon_returns(daily, horizon=10)
    expected = np.exp(10 * r) - 1.0
    assert np.allclose(hr, expected)


def test_horizon_returns_insufficient_history():
    assert len(horizon_returns(np.array([0.01, 0.02]), horizon=10)) == 0


def test_var_positive_loss(returns):
    v = historical_var(returns, horizon=21, confidence=0.95)
    assert v is not None and v > 0, "VaR is a positive loss magnitude"


def test_cvar_at_least_var(returns):
    v = historical_var(returns, horizon=21, confidence=0.95)
    cv = historical_cvar(returns, horizon=21, confidence=0.95)
    assert cv >= v, "CVaR (expected shortfall) must be >= VaR"


def test_higher_confidence_larger_var(returns):
    """99% VaR should be at least as large as 95% VaR (further into the tail)."""
    v95 = historical_var(returns, horizon=21, confidence=0.95)
    v99 = historical_var(returns, horizon=21, confidence=0.99)
    assert v99 >= v95


def test_var_none_on_short_history():
    assert historical_var(np.array([0.01, -0.01, 0.0]), horizon=21) is None
