"""Offline VaR calibration backtest -- the regression guard for the risk engine.

This is an offline job, not a per-request path. Run it after any data refresh or
change to the risk method. It answers the one question that decides whether the
engine's risk numbers can be trusted: **over history, was the VaR breached about
as often as promised?**

It validates the *calibrated* method -- `historical_var`, the empirical horizon
quantile -- pooled across the universe, and applies the two standard tests:

- **Kupiec (unconditional coverage):** is the breach *rate* consistent with the
  claimed confidence? With enough windows this detects a rate that is too high
  (understates risk -- dangerous) or too low (overstates it -- overly conservative).
- **Pooling for power:** a single ticker gives ~2-7 expected breaches, far too few
  to detect miscalibration. Pooling across the universe is where the statistical
  power comes from -- notebook 11 showed an apparent per-ticker pass (p=0.85 on 8
  names) became a decisive p=0.0002 across 50.

Two disciplines are inherited from notebook 11 and must not be relaxed:

1. **Non-overlapping windows** (step == horizon). Overlapping windows share data
   and understate the variance of the breach rate, invalidating the tests.
2. **Point-in-time fitting** -- each VaR uses only returns strictly before the
   forecast date. `historical_var.backtest_breach_rate` already enforces this.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Ensure the project root is on the import path so `src` can be imported.
PROJECT_ROOT = Path.cwd().resolve()
for parent in (PROJECT_ROOT, *PROJECT_ROOT.parents):
    if (parent / 'src').is_dir():
        PROJECT_ROOT = parent
        break
else:
    raise RuntimeError('Run this notebook from inside the StockForecastRisk repository.')

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecast_engine.risk import historical_var  # adjust to real package path


@dataclass
class CalibrationResult:
    horizon_days: int
    confidence: float
    n_windows: int
    n_breaches: int
    breach_rate: float
    expected_rate: float
    kupiec_statistic: float
    kupiec_p_value: float
    verdict: str


def kupiec_pof_test(n_windows: int, n_breaches: int, expected_rate: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures likelihood-ratio test.

    Returns (statistic, p_value). A p-value above 0.05 means the observed breach
    rate is consistent with `expected_rate`; below means it is not.
    """
    if n_windows == 0 or n_breaches == 0:
        return float("nan"), float("nan")
    observed = n_breaches / n_windows
    if observed in (0.0, 1.0):
        return float("nan"), float("nan")

    log_l_null = (
        (n_windows - n_breaches) * np.log(1 - expected_rate)
        + n_breaches * np.log(expected_rate)
    )
    log_l_alt = (
        (n_windows - n_breaches) * np.log(1 - observed)
        + n_breaches * np.log(observed)
    )
    statistic = -2 * (log_l_null - log_l_alt)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    return float(statistic), float(p_value)


def _verdict(breach_rate: float, expected_rate: float, p_value: float) -> str:
    if np.isnan(p_value):
        return "insufficient breaches to judge"
    if p_value > 0.05:
        return "calibrated: breach rate consistent with the claimed confidence"
    if breach_rate > expected_rate:
        return "UNDERSTATES RISK: more breaches than promised (dangerous)"
    return "OVERSTATES RISK: fewer breaches than promised (overly conservative)"


def backtest_calibration(
    returns_panel: pd.DataFrame,
    horizon_days: int = 21,
    confidence: float = 0.95,
    symbol_column: str = "symbol",
    return_column: str = "log_return",
    min_history: int = 252,
) -> CalibrationResult:
    """Pool the point-in-time breach test across the universe and judge it.

    `returns_panel` has one row per (symbol, date) with a daily log-return column.
    Each ticker is backtested with non-overlapping windows via
    `historical_var.backtest_breach_rate`, then the breach counts are pooled for
    the Kupiec test -- pooling is what gives the test power.
    """
    total_windows = 0
    total_breaches = 0

    for _, group in returns_panel.groupby(symbol_column):
        series = group.sort_values("date")[return_column].to_numpy()
        result = historical_var.backtest_breach_rate(
            series, horizon=horizon_days, confidence=confidence, min_history=min_history
        )
        if result is None:
            continue
        total_windows += result["n_windows"]
        total_breaches += result["n_breaches"]

    if total_windows == 0:
        raise ValueError("No ticker had enough history to backtest.")

    expected_rate = 1.0 - confidence
    breach_rate = total_breaches / total_windows
    statistic, p_value = kupiec_pof_test(total_windows, total_breaches, expected_rate)

    return CalibrationResult(
        horizon_days=horizon_days,
        confidence=confidence,
        n_windows=total_windows,
        n_breaches=total_breaches,
        breach_rate=breach_rate,
        expected_rate=expected_rate,
        kupiec_statistic=statistic,
        kupiec_p_value=p_value,
        verdict=_verdict(breach_rate, expected_rate, p_value),
    )


def run(returns_panel: pd.DataFrame, **kwargs) -> dict:
    """Entry point for the offline job: run, print, and return the result dict."""
    result = backtest_calibration(returns_panel, **kwargs)
    print(f"VaR calibration backtest ({result.horizon_days}-day, "
          f"{result.confidence:.0%} confidence)")
    print(f"  windows      : {result.n_windows:,}")
    print(f"  breaches     : {result.n_breaches:,}")
    print(f"  breach rate  : {result.breach_rate:.3%}  (expected {result.expected_rate:.1%})")
    print(f"  Kupiec p     : {result.kupiec_p_value:.4f}")
    print(f"  verdict      : {result.verdict}")
    return asdict(result)
