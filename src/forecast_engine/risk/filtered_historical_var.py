"""Filtered historical simulation (vol-scaled VaR).

WHY THIS EXISTS — the tail-clustering fix.

The unconditional historical VaR (`historical_var.py`) computes one quantile over
all history. The full-panel calibration backtest showed it breaches 6.40% against a
5% target (Kupiec p=0.0000), and the per-year breakdown showed why: the excess is
concentrated in high-vol periods (2020: 13.8%, 2022: 13.5%, 2011: 9.4%) while calm
years pool to 5.29% — and some calm years OVER-cover (2013: 2.6%, 2017: 2.6%). The
band is fixed at a through-the-cycle width, so it is too tight in vol spikes and too
loose in calm periods: the textbook failure of unconditional historical VaR.

THE FIX — standardize by the volatility that prevailed, then rescale by current vol:

    1. z_t = r_t / sigma_t        (standardize each daily return by trailing vol)
    2. quantile over the z's       (a regime-NEUTRAL "shape of the tail")
    3. VaR_today = quantile(z-based horizon returns) rescaled by CURRENT vol

When current vol is high (2020-like) the band widens; when calm (2013-like) it
tightens. This is Filtered Historical Simulation (Barone-Adesi et al.), the standard
answer to exactly this miscalibration. It connects the two halves of the engine: the
scaling input is the same volatility signal the forecast model is validated on.

The success criterion is NOT just a pooled 5% — it is the PER-YEAR breach rates
flattening toward 5% (run diagnose_var_by_year against this method to confirm).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

TRADING_DAYS = 252
DEFAULT_HORIZON = 21
DEFAULT_CONFIDENCE = 0.95
MIN_HISTORY = 252
MIN_HORIZON_SAMPLES = 60
VOL_WINDOW = 21             # trailing window for conditioning vol sigma_t. w21 gives the
                            # flattest per-year breach profile across the offered horizons
                            # (5d RMSE 0.012, 10d 0.015, 21d ~0.022 — best or tied at each).
                            # 60d does NOT calibrate at any window and is not offered.
                            # See training/validate_filtered_var.py for the per-horizon sweep.
VOL_FLOOR = 1e-6            # guard against divide-by-zero on flat stretches

# Per-horizon conditioning window — the SINGLE SOURCE OF TRUTH for which trailing
# vol window scales the VaR at each offered horizon. The window scales with the
# forecast horizon: short horizons use a short, fast-reacting window; the 21-day
# horizon uses a longer, smoother one. Chosen from the per-horizon validation sweep
# (training/validate_filtered_var.py) as the best-calibrated window at each horizon:
#   5d, 10d -> w21   (flattest per-year AND best pooled)
#   21d     -> w42   (best pooled 5.80%; per-year still good)
# 60d is absent: it does not calibrate at any window and is not offered.
# window_for_horizon() is the ONLY way callers should pick a window, so the engine
# and the disclosed figure can never silently diverge.
HORIZON_VOL_WINDOW = {5: 21, 10: 21, 21: 42}
DEFAULT_HORIZON_WINDOW = 21  # fallback for any horizon not in the map


def window_for_horizon(horizon: int) -> int:
    """The validated conditioning window for a given horizon (single source of truth)."""
    return HORIZON_VOL_WINDOW.get(horizon, DEFAULT_HORIZON_WINDOW)


def trailing_vol(daily_log_returns: np.ndarray, window: int = VOL_WINDOW) -> np.ndarray:
    """Rolling std of daily log returns, aligned so sigma_t uses only past data.

    Returns an array the same length as the input; the first `window` entries are
    NaN (insufficient history). sigma_t is the volatility KNOWN AT t-1 going into t,
    i.e. shifted by one so it never peeks at the return it standardizes.

    Vectorized: a trailing rolling std over `window` observations ENDING at t-1
    (shift(1)), which is identically what the former per-index loop computed
    (values[i-window:i].std) but in C rather than a Python loop.
    """
    import pandas as pd
    values = np.asarray(daily_log_returns, dtype=float)
    s = pd.Series(values)
    out = s.shift(1).rolling(window=window, min_periods=window).std(ddof=1)
    return out.to_numpy()


def standardized_returns(
    daily_log_returns: np.ndarray, window: int = VOL_WINDOW
) -> tuple[np.ndarray, np.ndarray]:
    """Return (z, sigma) where z_t = r_t / sigma_t, dropping the warm-up NaNs.

    z is the volatility-standardized daily return series — regime-neutral, so its
    quantiles describe the shape of the tail independent of the vol level.
    """
    values = np.asarray(daily_log_returns, dtype=float)
    sigma = trailing_vol(values, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = values / np.maximum(sigma, VOL_FLOOR)
    mask = np.isfinite(z) & np.isfinite(sigma)
    return z[mask], sigma[mask]


def _current_vol(daily_log_returns: np.ndarray, window: int = VOL_WINDOW) -> float | None:
    """The most recent trailing vol — the sigma to rescale the forecast VaR by."""
    values = np.asarray(daily_log_returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < window:
        return None
    return float(values[-window:].std(ddof=1))


def filtered_horizon_returns(
    daily_log_returns: Iterable[float],
    horizon: int,
    window: int = VOL_WINDOW,
) -> np.ndarray:
    """Vol-scaled horizon returns: standardize, window to horizon, rescale by current vol.

    Each standardized daily return is scaled UP to today's vol regime, then summed
    over the horizon and exponentiated — so the resulting distribution reflects the
    tail shape of all history but the volatility LEVEL of now.
    """
    values = np.asarray(daily_log_returns, dtype=float)
    values = values[np.isfinite(values)]
    z, _ = standardized_returns(values, window)
    sigma_now = _current_vol(values, window)
    if sigma_now is None or len(z) < horizon + 1:
        return np.array([])
    # Rescale standardized daily returns to today's vol, then build horizon windows.
    scaled_daily = z * sigma_now
    windows = np.lib.stride_tricks.sliding_window_view(scaled_daily, horizon)
    return np.exp(windows.sum(axis=1)) - 1.0


def filtered_var(
    daily_log_returns: Iterable[float],
    horizon: int = DEFAULT_HORIZON,
    confidence: float = DEFAULT_CONFIDENCE,
    window: int = VOL_WINDOW,
) -> float | None:
    """Vol-scaled VaR (positive loss fraction). Drop-in replacement for historical_var."""
    returns = filtered_horizon_returns(daily_log_returns, horizon, window)
    if len(returns) < MIN_HORIZON_SAMPLES:
        return None
    return float(-np.quantile(returns, 1.0 - confidence))


def filtered_cvar(
    daily_log_returns: Iterable[float],
    horizon: int = DEFAULT_HORIZON,
    confidence: float = DEFAULT_CONFIDENCE,
    window: int = VOL_WINDOW,
) -> float | None:
    """Vol-scaled CVaR (expected shortfall)."""
    returns = filtered_horizon_returns(daily_log_returns, horizon, window)
    if len(returns) < MIN_HORIZON_SAMPLES:
        return None
    threshold = np.quantile(returns, 1.0 - confidence)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return None
    return float(-tail.mean())


def filtered_percentiles(
    daily_log_returns: Iterable[float],
    horizon: int = DEFAULT_HORIZON,
    percentiles: Iterable[float] = (5, 25, 50, 75, 95),
    window: int = VOL_WINDOW,
) -> dict[str, float] | None:
    returns = filtered_horizon_returns(daily_log_returns, horizon, window)
    if len(returns) < MIN_HORIZON_SAMPLES:
        return None
    cut = np.percentile(returns, list(percentiles))
    return {f"p{int(p)}": float(v) for p, v in zip(percentiles, cut)}


@dataclass
class RiskEstimate:
    horizon_days: int
    confidence: float
    var: float
    cvar: float
    percentiles: dict[str, float]
    n_samples: int
    method: str = "filtered_historical_simulation"
    vol_window: int = VOL_WINDOW    # which conditioning window produced this estimate


def estimate_risk(
    daily_log_returns: Iterable[float],
    horizon: int = DEFAULT_HORIZON,
    confidence: float = DEFAULT_CONFIDENCE,
    window: int | None = None,
) -> RiskEstimate | None:
    """Assemble the full vol-scaled risk estimate — same shape as historical_var's.

    Drop-in for historical_var.estimate_risk. When `window` is None (the normal
    case), the per-horizon validated window is used (window_for_horizon), so callers
    get the correctly-calibrated window for the horizon automatically. Pass an
    explicit `window` only to override (e.g. in the validation sweep).
    """
    if window is None:
        window = window_for_horizon(horizon)
    returns = filtered_horizon_returns(daily_log_returns, horizon, window)
    if len(returns) < MIN_HORIZON_SAMPLES:
        return None
    threshold = np.quantile(returns, 1.0 - confidence)
    tail = returns[returns <= threshold]
    cuts = np.percentile(returns, [5, 25, 50, 75, 95])
    return RiskEstimate(
        horizon_days=horizon,
        confidence=confidence,
        var=float(-threshold),
        cvar=float(-tail.mean()) if len(tail) else float(-threshold),
        percentiles={f"p{p}": float(v) for p, v in zip((5, 25, 50, 75, 95), cuts)},
        n_samples=len(returns),
        vol_window=window,
    )


def backtest_breach_rate(
    daily_log_returns: Iterable[float],
    horizon: int = DEFAULT_HORIZON,
    confidence: float = DEFAULT_CONFIDENCE,
    min_history: int = MIN_HISTORY,
    step: int | None = None,
    window: int = VOL_WINDOW,
) -> dict[str, float] | None:
    """Point-in-time breach test for the vol-scaled VaR — same protocol as the base.

    At each step, sigma_now and the standardized quantile are computed from returns
    strictly before the point, so the vol scaling is itself point-in-time (no peeking
    at the vol regime it is being tested in).
    """
    values = np.asarray(daily_log_returns, dtype=float)
    values = values[np.isfinite(values)]
    step = horizon if step is None else step

    breaches = 0
    windows = 0
    for position in range(min_history, len(values) - horizon, step):
        var = filtered_var(values[:position], horizon, confidence, window)
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