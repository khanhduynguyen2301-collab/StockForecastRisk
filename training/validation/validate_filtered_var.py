"""Validate the vol-scaled VaR fix on the real panel — the risk-half acceptance gate.

Runs the per-year breach-rate diagnostic for BOTH methods side by side:
  - unconditional  (historical_var — the current, miscalibrated method)
  - vol-scaled      (filtered_historical_var — the fix)

Success is NOT a pooled 5%. Success is the vol-scaled method's PER-YEAR rates
flattening toward 5% — especially the crisis years (2020, 2022, 2011) dropping from
~13% toward 5% — without the calm years blowing past 5% the other way.

Also sweeps the conditioning window (21/42/63) so you can pick the one that gives the
flattest per-year profile: shorter windows react faster but are noisier.

Run:  python -m training.validate_filtered_var --horizon 21
      python -m training.validate_filtered_var --horizon 21 --windows 21 42 63
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
from src.forecast_engine.risk import historical_var as base
from src.forecast_engine.risk import filtered_historical_var as filt

CRISIS_YEARS = {2011, 2018, 2020, 2022}


def load_returns() -> pd.DataFrame:
    panel = load_processed_features(columns=["symbol", "date", "adj_close"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["log_return"] = (
        panel.groupby("symbol")["adj_close"]
        .transform(lambda c: np.log(c.astype(float) / c.astype(float).shift(1)))
    )
    return panel.dropna(subset=["log_return"]).reset_index(drop=True)


def breaches_by_year(panel, var_fn, horizon, confidence, min_history, window=None):
    """Pooled per-year breach series, PIT non-overlapping, attributed to as-of date."""
    recs = []
    for _, g in panel.groupby("symbol"):
        g = g.sort_values("date")
        vals = g["log_return"].to_numpy()
        dates = g["date"].to_numpy()
        for pos in range(min_history, len(vals) - horizon, horizon):
            var = (var_fn(vals[:pos], horizon, confidence, window) if window is not None
                   else var_fn(vals[:pos], horizon, confidence))
            if var is None:
                continue
            realised = np.exp(vals[pos:pos + horizon].sum()) - 1.0
            recs.append((pd.Timestamp(dates[pos]).year, int(realised < -var)))
    df = pd.DataFrame(recs, columns=["year", "br"])
    return df.groupby("year")["br"].agg(["mean", "count"]), df["br"].mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-history", type=int, default=252)
    parser.add_argument("--windows", type=int, nargs="+", default=[21, 42, 63])
    args = parser.parse_args()
    exp = 1.0 - args.confidence

    panel = load_returns()
    print(f"Panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}\n")

    # baseline (unconditional)
    base_year, base_pooled = breaches_by_year(
        panel, base.historical_var, args.horizon, args.confidence, args.min_history
    )
    result = pd.DataFrame({"unconditional": base_year["mean"]})

    # vol-scaled across windows
    pooled = {"unconditional": base_pooled}
    for w in args.windows:
        vy, vp = breaches_by_year(
            panel, filt.filtered_var, args.horizon, args.confidence, args.min_history, window=w
        )
        result[f"vol_scaled_w{w}"] = vy["mean"]
        pooled[f"vol_scaled_w{w}"] = vp

    result["expected"] = exp
    print("Per-year breach rate (fraction; expected = %.2f):" % exp)
    print((result * 100).round(1).to_string())

    print("\nPooled breach rate:")
    for k, v in pooled.items():
        print(f"  {k:<18}: {v:.2%}")

    print("\nFlatness (std of per-year rates around expected; lower = better):")
    for col in result.columns:
        if col == "expected":
            continue
        yrs = result[col].dropna()
        rmse = float(np.sqrt(((yrs - exp) ** 2).mean()))
        crisis = yrs[yrs.index.isin(CRISIS_YEARS)].mean()
        print(f"  {col:<18}: RMSE-to-5% {rmse:.3f}   crisis-year mean {crisis:.2%}")

    # pick the best vol-scaled window by RMSE-to-target
    vs_cols = [c for c in result.columns if c.startswith("vol_scaled")]
    rmses = {c: float(np.sqrt(((result[c].dropna() - exp) ** 2).mean())) for c in vs_cols}
    best = min(rmses, key=rmses.get)
    print(f"\n=== READ ===")
    print(f"Flattest vol-scaled method: {best} (RMSE-to-5% {rmses[best]:.3f})")
    base_rmse = float(np.sqrt(((base_year['mean'] - exp) ** 2).mean()))
    print(f"vs unconditional RMSE-to-5%: {base_rmse:.3f}")
    if rmses[best] < base_rmse * 0.7:
        print(f"PASS: {best} substantially flattens the per-year profile. Adopt it "
              f"(set VOL_WINDOW accordingly), swap the orchestrator import, re-run the "
              f"aggregate Kupiec to confirm pooled ~5%.")
    else:
        print(f"PARTIAL: vol-scaling helps but the profile is still uneven. Consider a "
              f"longer conditioning window, or check whether specific years resist scaling "
              f"(idiosyncratic, not vol-driven).")


if __name__ == "__main__":
    main()
