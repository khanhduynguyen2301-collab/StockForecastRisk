"""Per-horizon Kupiec disclosure sweep for the offered horizons (5/10/21).

Produces the disclosure breach rate for each horizon the app offers, using the
shipping method (filtered_historical_var at its module VOL_WINDOW). These are the
numbers the app's horizon-aware calibration note must show — the disclosed figure
has to match the horizon being displayed, or the honesty note is wrong.

60-day is intentionally excluded: it does not calibrate at any conditioning window
(crisis-year breach ~10%+, vol-scaling does not rescue it), so it is not offered.

Run:  python -m training.sweep_horizon_disclosure
"""
from __future__ import annotations

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

OFFERED_HORIZONS = [5, 10, 21]


def kupiec_p(n_windows: int, n_breaches: int, expected_rate: float) -> float:
    if n_windows == 0 or n_breaches == 0:
        return float("nan")
    observed = n_breaches / n_windows
    if observed in (0.0, 1.0):
        return float("nan")
    ll_null = (n_windows - n_breaches) * np.log(1 - expected_rate) + n_breaches * np.log(expected_rate)
    ll_alt = (n_windows - n_breaches) * np.log(1 - observed) + n_breaches * np.log(observed)
    stat = -2 * (ll_null - ll_alt)
    return float(1 - stats.chi2.cdf(stat, df=1))


def main() -> None:
    confidence = 0.95
    exp = 1.0 - confidence

    panel = load_processed_features(columns=["symbol", "date", "adj_close"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["log_return"] = (
        panel.groupby("symbol")["adj_close"]
        .transform(lambda c: np.log(c.astype(float) / c.astype(float).shift(1)))
    )
    panel = panel.dropna(subset=["log_return"]).reset_index(drop=True)
    print(f"Panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols")
    print(f"Method: filtered_historical_simulation (VOL_WINDOW={filt.VOL_WINDOW})\n")

    grouped = [g.sort_values("date")["log_return"].to_numpy() for _, g in panel.groupby("symbol")]

    rows = []
    for h in OFFERED_HORIZONS:
        tw = tb = 0
        for series in grouped:
            r = filt.backtest_breach_rate(series, horizon=h, confidence=confidence)
            if r is None:
                continue
            tw += r["n_windows"]
            tb += r["n_breaches"]
        rate = tb / tw if tw else float("nan")
        rows.append({"horizon": h, "windows": tw, "breaches": tb,
                     "breach_rate": rate, "kupiec_p": kupiec_p(tw, tb, exp)})

    out = pd.DataFrame(rows).set_index("horizon")
    print("Per-horizon disclosure (breach rate vs 5% target):")
    print(out.assign(
        breach_rate=lambda d: (d["breach_rate"] * 100).round(2).astype(str) + "%",
        kupiec_p=lambda d: d["kupiec_p"].round(4),
    ).to_string())

    print("\n=== FOR THE APP'S CALIBRATION NOTE ===")
    for h, r in out.iterrows():
        print(f"  {h:>2}-day: breach {r['breach_rate']:.2%}  "
              f"(disclose this next to a {h}-day VaR)")
    print("\nWire these into the app's horizon-aware note so the disclosed figure "
          "always matches the horizon on screen.")


if __name__ == "__main__":
    main()
