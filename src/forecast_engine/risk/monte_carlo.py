"""Merton jump-diffusion path simulation — for VISUALISATION ONLY.

Read this before using anything here to produce a risk number: **do not.**

This module simulates price paths for fan charts and illustrative percentile
cones. It must NOT be the source of the VaR/CVaR the engine reports. That number
comes from `risk/historical_var.py`.

The reason is a hard-won result from notebook 11. This engine fits parameters on
*daily* returns and compounds simulated daily steps to the horizon. Compounding
independent daily draws assumes cross-day independence; real returns mean-revert
mildly at multi-day horizons, so the compounded tail is too wide. Backtesting
showed Monte Carlo VaR breached only ~3.5% of the time against a 5% target —
systematically overstating risk — while a historical quantile of horizon returns
calibrated at ~4.6%. So:

    risk numbers  -> historical_var.py   (calibrated)
    fan charts    -> this module          (illustrative only)

Two further notes on what this module deliberately does NOT do:

- **No steered drift.** An earlier design (blueprint Phase 4b) planned to warp the
  drift with an ML return-direction signal. Notebook 08 showed absolute return
  direction is not predictable from the available features, so there is no signal
  to steer with. Drift here is a plain, configurable input (default: risk-free).
- **No claim of calibration.** The percentiles this produces are a model's view of
  the distribution, not a validated risk statement. The engine surfaces them as a
  visual, alongside the calibrated historical figures.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass
class JumpDiffusionParams:
    """Per-ticker Merton parameters (from training/fit_jump_diffusion_params).

    sigma and lambda are per-ticker; jump_mean and jump_std are pooled across
    tickers, because ~26 jumps per ticker is far too few to fit a jump-size
    distribution reliably (notebook 09's bootstrap justifies the pooling).
    """
    mu_annual: float
    sigma_annual: float
    lambda_per_year: float
    jump_mean: float
    jump_std: float


def simulate_paths(
    initial_price: float,
    params: JumpDiffusionParams,
    horizon_days: int,
    n_paths: int = 10_000,
    seed: int = 42,
    trading_days: int = TRADING_DAYS,
) -> np.ndarray:
    """Simulate Merton jump-diffusion price paths.

    Returns an array of shape (n_paths, horizon_days) of simulated prices. This
    is a faithful simulation; it is the *use as a VaR source* that is disallowed,
    not the simulation itself.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / trading_days

    # kappa = E[e^Y - 1]: expected proportional price change per jump.
    kappa = np.exp(params.jump_mean + 0.5 * params.jump_std ** 2) - 1.0

    # Drift with Ito correction and the jump compensator (-lambda*kappa). The
    # compensator keeps the mean unbiased under asymmetric jumps; removing it
    # biases terminal prices low by ~9% for equity-like negative jump means.
    drift = (params.mu_annual - 0.5 * params.sigma_annual ** 2
             - params.lambda_per_year * kappa) * dt

    diffusion = params.sigma_annual * np.sqrt(dt) * rng.standard_normal((n_paths, horizon_days))

    counts = rng.poisson(params.lambda_per_year * dt, (n_paths, horizon_days))
    jumps = np.where(
        counts > 0,
        rng.normal(params.jump_mean * counts,
                   max(params.jump_std, 1e-12) * np.sqrt(np.maximum(counts, 1))),
        0.0,
    )

    log_increments = drift + diffusion + jumps
    return initial_price * np.exp(np.cumsum(log_increments, axis=1))


def fan_chart_percentiles(
    paths: np.ndarray,
    percentiles: tuple[int, ...] = (5, 25, 50, 75, 95),
) -> dict[str, np.ndarray]:
    """Percentile bands over time, for drawing a fan chart.

    Returns {"p5": array over horizon, ...}. These are illustrative bands, not a
    risk statement — see the module docstring.
    """
    cuts = np.percentile(paths, list(percentiles), axis=0)
    return {f"p{p}": cuts[i] for i, p in enumerate(percentiles)}


def terminal_distribution(paths: np.ndarray, initial_price: float) -> np.ndarray:
    """Terminal simple returns across paths — for a histogram, not for VaR."""
    return paths[:, -1] / initial_price - 1.0


# A deliberately loud name so a future caller cannot quietly mistake this for the
# risk API. If you find yourself importing this to fill var_95, stop and use
# risk/historical_var.py instead.
def NOT_FOR_VAR_illustrative_terminal_quantile(paths: np.ndarray, initial_price: float,
                                               confidence: float = 0.95) -> float:
    """Illustrative only. The engine's VaR is historical_var.estimate_risk().

    Provided so notebooks can show the *difference* between the (miscalibrated)
    simulation quantile and the calibrated historical figure. Never wire this
    into the service response.
    """
    terminal = terminal_distribution(paths, initial_price)
    return float(-np.quantile(terminal, 1.0 - confidence))
