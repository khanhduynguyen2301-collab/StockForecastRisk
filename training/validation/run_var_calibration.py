"""Run the VaR calibration backtest on the real panel — the risk-half gate.

The smoke test proved the risk block *assembles* correctly. This proves the risk
numbers are *calibrated*: that a claimed 95% VaR is actually breached about 5% of
the time, pooled across the universe with non-overlapping, point-in-time windows.

This is the re-confirmation of notebook 11 (~4.6% breach) on the current
2010–2026 panel, at the horizon the engine actually reports VaR (21d).

Reads: data/processed/sp500_features.parquet (needs symbol, date, adj_close).
Derives daily log returns per ticker (the panel has no log_return column), then
calls backtest.run.

Run:  python -m training.run_var_calibration            # horizon 21, 95%
      python -m training.run_var_calibration --horizon 21 --confidence 0.95
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

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
# backtest.py lives under training/ in the repo; import by file location.
try:
    from training.validation.backtest import run as run_backtest
except ModuleNotFoundError:
    # fallback: some layouts keep it in src/forecast_engine/validation
    from src.forecast_engine.validation.backtest import run as run_backtest


def add_log_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add a per-ticker daily log-return column derived from adj_close.

    Grouped by symbol so returns never cross ticker boundaries; the first row of
    each ticker is NaN (no prior close) and is dropped.
    """
    panel = panel.sort_values(["symbol", "date"]).copy()
    panel["log_return"] = (
        panel.groupby("symbol")["adj_close"]
        .transform(lambda c: np.log(c.astype(float) / c.astype(float).shift(1)))
    )
    return panel.dropna(subset=["log_return"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=21,
                        help="VaR horizon in trading days (match what the engine reports).")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-history", type=int, default=252)
    args = parser.parse_args()

    panel = load_processed_features(columns=["symbol", "date", "adj_close"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = add_log_returns(panel)
    print(f"Panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}\n")

    result = run_backtest(
        panel,
        horizon_days=args.horizon,
        confidence=args.confidence,
        min_history=args.min_history,
    )

    # Interpret the verdict for the acceptance checklist.
    print("\n=== READ ===")
    br, exp = result["breach_rate"], result["expected_rate"]
    p = result["kupiec_p_value"]
    if p == p and p > 0.05:
        print(f"CALIBRATED: breach {br:.2%} vs {exp:.0%} expected, Kupiec p={p:.3f} "
              f"(> 0.05). The risk numbers can be trusted. v1 risk gate PASSES.")
    elif p == p and br > exp:
        print(f"UNDERSTATES RISK: breach {br:.2%} > {exp:.0%} expected, p={p:.3f}. "
              f"Do NOT ship the VaR as-is — it is too optimistic.")
    elif p == p:
        print(f"OVERSTATES RISK: breach {br:.2%} < {exp:.0%} expected, p={p:.3f}. "
              f"Conservative — safe to ship, but the bands are wider than necessary.")
    else:
        print("Insufficient breaches to judge — check panel size / horizon.")


if __name__ == "__main__":
    main()
