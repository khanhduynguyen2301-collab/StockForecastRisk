"""Diagnose the VaR miscalibration: is the 6.4% breach rate crisis-clustered or uniform?

The aggregate backtest said the 21-day 95% VaR breaches 6.40% (p=0.0000, understates
risk). This tells you WHICH fix you need:

  - If the excess breaches concentrate in crisis years (2020, 2022) and other years
    sit near 5%, the cause is TAIL-CLUSTERING. An unconditional historical quantile
    prices the average tail and under-covers during vol spikes. Fix: regime-conditional
    VaR (scale the quantile by forecasted vol) — reuses your validated vol model.

  - If breaches are ~uniformly above 5% across all years, the method is miscalibrated
    at this horizon generally. Fix: recalibrate the quantile / rework the method.

It reuses the exact same point-in-time, non-overlapping logic as
historical_var.backtest_breach_rate (same historical_var() call, same step==horizon),
but tags each test window by the DATE of its forward window so breaches can be bucketed
by year. The pooled totals here should reconcile with the aggregate run.

Run:  python -m training.diagnose_var_by_year --horizon 21
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
from src.forecast_engine.risk import historical_var as hv

# Known high-volatility regimes in the 2010–2026 span, for the crisis-vs-calm split.
CRISIS_YEARS = {2020, 2022}


def per_window_breaches(
    dates: np.ndarray,
    values: np.ndarray,
    horizon: int,
    confidence: float,
    min_history: int,
) -> list[tuple[pd.Timestamp, int]]:
    """Replicate backtest_breach_rate's loop, but return (window_date, breached) per window.

    window_date is the date at the START of the forward window (the 'as-of' date the
    VaR was made for), so a breach is attributed to when the forecast was live.
    """
    step = horizon
    out = []
    for position in range(min_history, len(values) - horizon, step):
        var = hv.historical_var(values[:position], horizon, confidence)
        if var is None:
            continue
        realised = np.exp(values[position:position + horizon].sum()) - 1.0
        breached = int(realised < -var)
        out.append((pd.Timestamp(dates[position]), breached))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-history", type=int, default=252)
    args = parser.parse_args()

    panel = load_processed_features(columns=["symbol", "date", "adj_close"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["log_return"] = (
        panel.groupby("symbol")["adj_close"]
        .transform(lambda c: np.log(c.astype(float) / c.astype(float).shift(1)))
    )
    panel = panel.dropna(subset=["log_return"]).reset_index(drop=True)

    expected = 1.0 - args.confidence
    records: list[tuple[pd.Timestamp, int]] = []
    for _, g in panel.groupby("symbol"):
        g = g.sort_values("date")
        recs = per_window_breaches(
            g["date"].to_numpy(), g["log_return"].to_numpy(),
            args.horizon, args.confidence, args.min_history,
        )
        records.extend(recs)

    df = pd.DataFrame(records, columns=["date", "breached"])
    df["year"] = df["date"].dt.year

    # pooled reconciliation
    total_w, total_b = len(df), int(df["breached"].sum())
    print(f"Pooled: {total_b:,}/{total_w:,} = {total_b/total_w:.2%} breach "
          f"(expected {expected:.0%})  [should match the aggregate run]\n")

    # by year
    by_year = df.groupby("year")["breached"].agg(["sum", "count"])
    by_year["breach_rate"] = by_year["sum"] / by_year["count"]
    by_year["excess"] = by_year["breach_rate"] - expected
    by_year["flag"] = np.where(by_year["breach_rate"] > expected * 1.5, "  <-- HIGH", "")
    print("Breach rate by year (window attributed to its as-of date):")
    print(by_year.assign(
        breach_rate=lambda d: (d["breach_rate"] * 100).round(2).astype(str) + "%",
        excess=lambda d: (d["excess"] * 100).round(2).astype(str) + "pp",
    )[["sum", "count", "breach_rate", "excess", "flag"]].to_string())

    # crisis vs calm split
    crisis = df[df["year"].isin(CRISIS_YEARS)]
    calm = df[~df["year"].isin(CRISIS_YEARS)]
    cr = crisis["breached"].mean() if len(crisis) else float("nan")
    cl = calm["breached"].mean() if len(calm) else float("nan")
    print(f"\nCrisis years {sorted(CRISIS_YEARS)}: {cr:.2%} breach ({len(crisis):,} windows)")
    print(f"All other years         : {cl:.2%} breach ({len(calm):,} windows)")

    print("\n=== DIAGNOSIS ===")
    if cl <= expected * 1.2 and cr > expected * 1.5:
        print(f"TAIL-CLUSTERING. Calm years calibrate ({cl:.2%} ~ {expected:.0%}); the excess "
              f"lives in crises ({cr:.2%}). The unconditional VaR is fine on average but "
              f"under-covers vol spikes. FIX: regime-conditional VaR — scale the historical "
              f"quantile by your forecasted volatility. Reuses the validated vol model and "
              f"connects the two halves of the engine.")
    elif cl > expected * 1.2:
        print(f"UNIFORM MISCALIBRATION. Even calm years breach above 5% ({cl:.2%}). This is not "
              f"just clustering — the method is miscalibrated at this horizon. FIX: recalibrate "
              f"the quantile (widen) and/or reconsider the horizon; regime-scaling alone won't "
              f"fully fix it.")
    else:
        print(f"MIXED. Calm {cl:.2%}, crisis {cr:.2%}. Inspect the by-year table above; the fix "
              f"is likely regime-conditional but confirm calm years truly sit near {expected:.0%}.")

    print("\nAlso reconcile: notebook 11 reported ~4.6% on the OLD cut. Confirm its horizon and "
          "date range — this run supersedes that claim on the current 2010–2026 panel.")


if __name__ == "__main__":
    main()
