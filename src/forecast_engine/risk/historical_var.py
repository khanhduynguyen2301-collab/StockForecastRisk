"""Historical-simulation Value at Risk on horizon returns.

SUPERSEDED FOR SERVING. The engine's risk numbers now come from
`filtered_historical_var.py` (vol-scaled / filtered historical simulation), not from
this unconditional method. This module is kept for reference, for the backtest
comparison, and because the filtered version reuses its horizon-return machinery.

Why it was replaced -- the honest history:

The project originally computed VaR from a Merton jump-diffusion Monte Carlo
(notebooks 09-10), which compounded independent daily steps and overstated the tail.
Notebook 11, on the OLD data cut (~2021-2026), showed this UNCONDITIONAL historical
quantile calibrated cleanly (~4.6% breach vs 5%), so it replaced the Monte Carlo.

That ~4.6% did NOT survive the full 2010-2026 panel. The full-panel calibration
backtest showed 6.40% breach (Kupiec p=0.0000 -- understates risk), and a per-year
breakdown showed why: the breaches cluster in high-vol regimes (2020: 13.8%, 2022:
13.4%, 2011: 9.4%) while calm years pool near 5% and some over-cover (2013/2017:
~2.6%). This is the textbook failure of UNCONDITIONAL historical VaR: a fixed
through-the-cycle band is too tight in vol spikes and too loose in calm periods. The
old ~4.6% was an artifact of a benign sample window, not a calibrated method.

The fix (see filtered_historical_var.py) standardizes returns by the prevailing
volatility, takes the quantile of the standardized series, and rescales by current
vol -- so the band breathes with the regime. On the full panel this flattens the
per-year profile (RMSE-to-5% 0.038 -> 0.020) and cuts crisis-year breaches roughly
in half. It is calibrated in normal conditions and modestly conservative-leaning
overall, with a residual understatement in severe stress that the engine discloses.

The core functions here are pure (returns in, numbers out) so they remain useful for
the backtest comparison; a thin point-in-time backtest wraps them for validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_HORIZON = 21
DEFAULT_CONFIDENCE = 0.95
MIN_HISTORY = 252            # at least a year of daily returns before a forecast
MIN_HORIZON_SAMPLES = 60     # minimum overlapping horizon returns to trust a quantile


def horizon_returns(daily_log_returns: Iterable[float], horizon: int) -> np.ndarray:
    """Overlapping horizon-length simple returns from a daily log-return series.

    A horizon return is exp(sum of `horizon` consecutive daily log returns) - 1.
    Overlapping windows are used deliberately: they maximise the sample available
    for the empirical quantile. (Overlap inflates the *effective* sample for
    significance testing, which matters in the backtest below, but not for the
    point estimate of the quantile itself.)
    """
    values = np.asarray(daily_log_returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < horizon + 1:
        return np.array([])
    windows = np.lib.stride_tricks.sliding_window_view(values, horizon)
    return np.exp(windows.sum(axis=1)) - 1.0


def historical_var(
    daily_log_returns: Iterable[float],
    horizon: int = DEFAULT_HORIZON,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float | None:
    """VaR as a positive loss fraction, from the empirical horizon-return quantile.

    Returns the loss at the (1 - confidence) quantile of horizon returns, as a
    positive number (a VaR of 0.038 means "a 5% chance of losing more than 3.8%
    over the horizon"). Returns None when there is too little history to estimate
    the quantile reliably, so callers can distinguish "no estimate" from "zero
    risk".
    """
    returns = horizon_returns(daily_log_returns, horizon)
    if len(returns) < MIN_HORIZON_SAMPLES:
        return None
    return float(-np.quantile(returns, 1.0 - confidence))


def historical_cvar(
    daily_log_returns: Iterable[float],
    horizon: int = DEFAULT_HORIZON,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float | None:
    """Conditional VaR (expected shortfall): mean loss beyond the VaR threshold.

    CVaR answers "if the loss exceeds VaR, how bad is it on average?" -- always at
    least as large as VaR, and more sensitive to the shape of the tail.
    """
    returns = horizon_returns(daily_log_returns, horizon)
    if len(returns) < MIN_HORIZON_SAMPLES:
        return None
    threshold = np.quantile(returns, 1.0 - confidence)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return None
    return float(-tail.mean())


def risk_percentiles(
    daily_log_returns: Iterable[float],
    horizon: int = DEFAULT_HORIZON,
    percentiles: Iterable[float] = (5, 25, 50, 75, 95),
) -> dict[str, float] | None:
    """Empirical horizon-return distribution at the given percentiles.

    Produces the `percentiles` block of the engine's risk response directly from
    the realised horizon-return distribution -- the same distribution the VaR is
    read from, so the numbers are mutually consistent.
    """
    returns = horizon_returns(daily_log_returns, horizon)
    if len(returns) < MIN_HORIZON_SAMPLES:
        return None
    cut = np.percentile(returns, list(percentiles))
    return {f"p{int(p)}": float(v) for p, v in zip(percentiles, cut)}


@dataclass
class RiskEstimate:
    """The risk half of the engine's response for one ticker."""
    horizon_days: int
    confidence: float
    var: float
    cvar: float
    percentiles: dict[str, float]
    n_samples: int


def estimate_risk(
    daily_log_returns: Iterable[float],
    horizon: int = DEFAULT_HORIZON,
    confidence: float = DEFAULT_CONFIDENCE,
) -> RiskEstimate | None:
    """Assemble the full risk estimate for one ticker from its return history.

    This is the function the service layer calls. Everything is read from the one
    empirical horizon-return distribution, so VaR, CVaR, and the percentiles agree
    with each other by construction.
    """
    returns = horizon_returns(daily_log_returns, horizon)
    if len(returns) < MIN_HORIZON_SAMPLES:
        return None

    threshold = np.quantile(returns, 1.0 - confidence)
    tail = returns[returns <= threshold]
    percentile_cuts = np.percentile(returns, [5, 25, 50, 75, 95])

    return RiskEstimate(
        horizon_days=horizon,
        confidence=confidence,
        var=float(-threshold),
        cvar=float(-tail.mean()) if len(tail) else float(-threshold),
        percentiles={f"p{p}": float(v) for p, v in zip((5, 25, 50, 75, 95), percentile_cuts)},
        n_samples=len(returns),
    )


def backtest_breach_rate(
    daily_log_returns: Iterable[float],
    horizon: int = DEFAULT_HORIZON,
    confidence: float = DEFAULT_CONFIDENCE,
    min_history: int = MIN_HISTORY,
    step: int | None = None,
) -> dict[str, float] | None:
    """Point-in-time breach rate for one ticker's return series.

    At each step the VaR is estimated from returns strictly *before* the point,
    then compared against the realised forward horizon return. A well-calibrated
    VaR at `confidence` should be breached about (1 - confidence) of the time.

    `step` defaults to `horizon` (non-overlapping test windows). Non-overlapping
    windows are required for honest calibration statistics: overlapping forward
    windows share data and would understate the variance of the breach rate. This
    is the same discipline notebook 11 established after overlapping windows there
    produced a spurious clustering result.
    """
    values = np.asarray(daily_log_returns, dtype=float)
    values = values[np.isfinite(values)]
    step = horizon if step is None else step

    breaches = 0
    windows = 0
    for position in range(min_history, len(values) - horizon, step):
        var = historical_var(values[:position], horizon, confidence)
        if var is None:
            continue
        realised = np.exp(values[position:position + horizon].sum()) - 1.0
        breaches += int(realised < -var)
        windows += 1

    if windows == 0:
        return None
    return {
        "breach_rate": breaches / windows,
        "expected_rate": 1.0 - confidence,
        "n_windows": windows,
        "n_breaches": breaches,
    }