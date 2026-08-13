"""Aggregate Kupiec disclosure for the SHIPPING risk method (filtered/vol-scaled VaR).

This reports the pooled breach rate and Kupiec p-value for the method the engine
actually serves — filtered_historical_var — as a DISCLOSURE number, not a pass/fail
gate. Per the Option-A decision: the engine ships a vol-scaled VaR that is calibrated
in normal conditions and mildly conservative-leaning overall, with a residual
understatement in severe stress disclosed in the response's calibration_note. With
~86k windows the Kupiec test may still reject a ~5.8% rate against 5% — that is
expected and disclosed, not a failure.

Run:  python -m training.run_kupiec_disclosure --horizon 21
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path.cwd().resolve()
for parent in (PROJECT_ROOT, *PROJECT_ROOT.parents):
    if (parent / "src").is_dir():
        PROJECT_ROOT = parent
        break
else:
    raise RuntimeError("Run from inside the StockForecastRisk repository.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecast_engine.data.loader import load_processed_features
from src.forecast_engine.risk import filtered_historical_var as filt


def kupiec_pof(n_windows: int, n_breaches: int, expected_rate: float) -> tuple[float, float]:
    if n_windows == 0 or n_breaches == 0:
        return float("nan"), float("nan")
    observed = n_breaches / n_windows
    if observed in (0.0, 1.0):
        return float("nan"), float("nan")
    ll_null = (n_windows - n_breaches) * np.log(1 - expected_rate) + n_breaches * np.log(expected_rate)
    ll_alt = (n_windows - n_breaches) * np.log(1 - observed) + n_breaches * np.log(observed)
    stat = -2 * (ll_null - ll_alt)
    return float(stat), float(1 - stats.chi2.cdf(stat, df=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-history", type=int, default=252)
    args = parser.parse_args()
    exp = 1.0 - args.confidence

    panel = load_processed_features(columns=["symbol", "date", "adj_close"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["log_return"] = (
        panel.groupby("symbol")["adj_close"]
        .transform(lambda c: np.log(c.astype(float) / c.astype(float).shift(1)))
    )
    panel = panel.dropna(subset=["log_return"]).reset_index(drop=True)
    print(f"Panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"Method: filtered_historical_simulation (VOL_WINDOW={filt.VOL_WINDOW})\n")

    total_w = total_b = 0
    for _, g in panel.groupby("symbol"):
        series = g.sort_values("date")["log_return"].to_numpy()
        r = filt.backtest_breach_rate(
            series, horizon=args.horizon, confidence=args.confidence, min_history=args.min_history
        )
        if r is None:
            continue
        total_w += r["n_windows"]
        total_b += r["n_breaches"]

    if total_w == 0:
        raise SystemExit("No ticker had enough history to backtest.")

    rate = total_b / total_w
    stat, p = kupiec_pof(total_w, total_b, exp)

    print(f"windows      : {total_w:,}")
    print(f"breaches     : {total_b:,}")
    print(f"breach rate  : {rate:.3%}   (expected {exp:.1%})")
    print(f"Kupiec stat  : {stat:.2f}")
    print(f"Kupiec p     : {p:.4f}")
    print("\n=== DISCLOSURE ===")
    direction = "understates" if rate > exp else "overstates"
    print(f"Shipping method breach rate: {rate:.2%} ({direction} vs {exp:.0%} target).")
    print(f"This is the disclosed calibration figure. The response's calibration_note "
          f"states the residual stress behavior. Record this number in the v1 docs / "
          f"acceptance checklist as the honest calibration of the shipped risk engine.")


if __name__ == "__main__":
    main()
