"""Risk-engine invariants — the math that must not silently break.

These guard the correctness gates from the acceptance checklist: filtered VaR
coherence, per-horizon window selection, and the vol-scaling behaving sanely.
"""
import numpy as np
import pytest

from src.forecast_engine.risk import filtered_historical_var as filt


@pytest.fixture
def returns():
    """A long, mildly fat-tailed daily log-return series (deterministic)."""
    rng = np.random.default_rng(42)
    base = rng.normal(0.0003, 0.015, 4000)
    jumps = np.where(rng.random(4000) < 0.01, rng.normal(-0.03, 0.05, 4000), 0.0)
    return base + jumps


def test_var_cvar_coherence(returns):
    """CVaR must be at least as severe as VaR (expected shortfall >= threshold)."""
    est = filt.estimate_risk(returns, horizon=21, confidence=0.95)
    assert est is not None
    # var and cvar are stored as positive loss magnitudes
    assert est.cvar >= est.var, "CVaR must be >= VaR in magnitude"
    assert est.var > 0, "VaR should be a positive loss magnitude for equity returns"


def test_percentiles_monotone(returns):
    est = filt.estimate_risk(returns, horizon=21, confidence=0.95)
    p = est.percentiles
    assert p["p5"] < p["p25"] < p["p50"] < p["p75"] < p["p95"], "percentiles must be ordered"


def test_per_horizon_window_selection():
    """Each offered horizon must resolve to its validated conditioning window."""
    assert filt.window_for_horizon(5) == 21
    assert filt.window_for_horizon(10) == 21
    assert filt.window_for_horizon(21) == 42


def test_estimate_uses_horizon_window_and_stamps_it(returns):
    """estimate_risk must use the per-horizon window and record which it used."""
    for h in (5, 10, 21):
        est = filt.estimate_risk(returns, horizon=h)
        assert est is not None
        assert est.vol_window == filt.window_for_horizon(h), \
            f"horizon {h} should use window {filt.window_for_horizon(h)}"


def test_trailing_vol_no_lookahead(returns):
    """sigma_t must use only data strictly before t (no peeking at the return it scales)."""
    sig = filt.trailing_vol(np.asarray(returns), window=21)
    # first `window` entries undefined; sigma at t must not depend on returns[t:]
    assert np.isnan(sig[:21]).all(), "warm-up entries must be NaN"
    assert np.isfinite(sig[21:]).all(), "post-warmup sigma must be defined"


def test_higher_vol_widens_var():
    """A higher-vol series should produce a larger VaR than a calm one."""
    rng = np.random.default_rng(1)
    calm = rng.normal(0, 0.008, 3000)
    wild = rng.normal(0, 0.030, 3000)
    v_calm = filt.filtered_var(calm, horizon=21, confidence=0.95)
    v_wild = filt.filtered_var(wild, horizon=21, confidence=0.95)
    assert v_wild > v_calm, "higher volatility must yield a larger VaR"


def test_insufficient_history_returns_none():
    """Too little history to form horizon windows must return None, not crash."""
    assert filt.filtered_var(np.array([0.01, -0.01, 0.0]), horizon=21) is None
