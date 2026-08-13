"""Merton jump-diffusion parameter estimation.

Read this before wiring the output anywhere near a risk number: **these
parameters feed fan-chart visualisation (`risk/monte_carlo.py`), not the VaR the
engine reports.** The engine's VaR/CVaR come from `risk/historical_var.py`, which
calibrated at ~4.6% breach rate; the Monte Carlo path engine these parameters
drive overstates the tail because it compounds daily steps (notebook 11). This
module is kept because the fitted distribution is validated for what it claims --
reproducing the *daily* return kurtosis (notebook 09) -- and the fan chart is a
useful visual. It is not a risk oracle.

The estimation reflects two findings from notebook 09:

1. **Volatility is contaminated by jumps.** A single standard deviation over all
   returns is inflated by crash days. An iterative threshold estimator flags
   returns beyond `k` sigma as jumps, refits sigma on the remainder, and repeats.
   Iterating matters: the first pass uses an inflated sigma and misses the
   smaller jumps.

2. **Jumps are too rare to fit per ticker.** ~26 flagged jumps per ticker is far
   too few to estimate a jump-size distribution. So sigma and the jump intensity
   lambda are fitted per ticker (a count estimates a rate acceptably), but
   jump_mean and jump_std are **pooled across all tickers**. A bootstrap in
   notebook 09 showed the apparent per-ticker spread in jump size is sampling
   noise. This pooling is specified in the project blueprint; do not revert it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
JUMP_THRESHOLD_K = 3.0      # returns beyond k sigma are treated as jumps
MIN_OBSERVATIONS = 60
MAX_ITERATIONS = 10


@dataclass
class DiffusionFit:
    """Per-ticker diffusion estimate plus the flagged jump returns for pooling."""
    mu_daily: float
    sigma_daily: float
    jump_returns: np.ndarray
    n_jumps: int
    n_obs: int


def fit_diffusion_and_flag_jumps(
    log_returns, k: float = JUMP_THRESHOLD_K
) -> DiffusionFit | None:
    """Iteratively separate the diffusion core from jump returns for one ticker.

    Returns None if there is too little history. The returned `jump_returns` are
    collected across tickers and pooled to estimate the shared jump-size
    distribution -- they are not used per ticker.
    """
    values = np.asarray(log_returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < MIN_OBSERVATIONS:
        return None

    mean, sigma = values.mean(), values.std(ddof=1)
    for _ in range(MAX_ITERATIONS):
        is_jump = np.abs(values - mean) > k * sigma
        core = values[~is_jump]
        if len(core) < 30:
            break
        new_mean, new_sigma = core.mean(), core.std(ddof=1)
        converged = abs(new_sigma - sigma) < 1e-10
        mean, sigma = new_mean, new_sigma
        if converged:
            break

    is_jump = np.abs(values - mean) > k * sigma
    return DiffusionFit(
        mu_daily=float(mean),
        sigma_daily=float(sigma),
        jump_returns=values[is_jump],
        n_jumps=int(is_jump.sum()),
        n_obs=len(values),
    )


def fit_jump_diffusion_parameters(
    returns: pd.DataFrame,
    symbol_column: str = "symbol",
    return_column: str = "log_return",
    k: float = JUMP_THRESHOLD_K,
) -> pd.DataFrame:
    """Fit Merton parameters for every ticker: sigma/lambda per name, jumps pooled.

    Returns a DataFrame indexed by symbol with columns:
    mu_annual, sigma_annual, mu_daily, sigma_daily, lambda_per_year,
    jump_mean, jump_std, n_jumps, n_obs.

    jump_mean and jump_std are identical across every row -- they are the pooled
    estimate, deliberately shared (see module docstring).
    """
    per_ticker: dict[str, DiffusionFit] = {}
    jump_pool: list[np.ndarray] = []

    for symbol, group in returns.groupby(symbol_column):
        fit = fit_diffusion_and_flag_jumps(group[return_column].values, k=k)
        if fit is None:
            continue
        per_ticker[symbol] = fit
        jump_pool.append(fit.jump_returns)

    if not per_ticker:
        raise ValueError("No tickers had enough history to fit.")

    pooled = np.concatenate(jump_pool) if jump_pool else np.array([])
    pooled_mean = float(pooled.mean()) if len(pooled) else 0.0
    pooled_std = float(pooled.std(ddof=1)) if len(pooled) > 1 else 0.0

    records = {
        symbol: {
            "mu_annual": fit.mu_daily * TRADING_DAYS,
            "sigma_annual": fit.sigma_daily * np.sqrt(TRADING_DAYS),
            "mu_daily": fit.mu_daily,
            "sigma_daily": fit.sigma_daily,
            "lambda_per_year": fit.n_jumps / fit.n_obs * TRADING_DAYS,
            "jump_mean": pooled_mean,   # pooled, identical across tickers
            "jump_std": pooled_std,     # pooled, identical across tickers
            "n_jumps": fit.n_jumps,
            "n_obs": fit.n_obs,
        }
        for symbol, fit in per_ticker.items()
    }
    frame = pd.DataFrame(records).T
    frame.index.name = symbol_column
    return frame


def empirical_kurtosis(log_returns) -> float:
    """Excess kurtosis of daily returns -- the quantity the jump model must match.

    Notebook 09's validation: a Gaussian diffusion cannot reproduce the empirical
    daily kurtosis (~5.5 on the real data), while the jump-diffusion can. This is
    what the module is validated *for* -- daily distributional shape, not
    multi-day VaR.
    """
    values = np.asarray(log_returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return float("nan")
    return float(pd.Series(values).kurtosis())   # Fisher (excess) by default
