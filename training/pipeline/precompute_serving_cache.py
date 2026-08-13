"""Precompute the serving cache — the offline half of the offline/online split.

The deployed app must not load the 1.9M-row panel (too big for a free cloud tier,
and unnecessary: a request for one ticker needs only that ticker's answer). This job
runs the full engine OFFLINE for every (ticker, horizon) and writes a compact cache
the app just reads.

For each ticker x horizon it stores:
  - the calibrated risk block (VaR/CVaR/percentiles/vol_window) from build_response
  - predicted_volatility
  - the 4 jump-diffusion params + last_price the fan chart needs (the fan is
    re-simulated in-app from these tiny inputs, so no 8000-path arrays are stored)

Output: models/serving_cache/serving_cache.parquet — small (a few thousand rows,
one per ticker x horizon), safe to commit to the repo and deploy on Streamlit Cloud.

Re-run this after any data refresh or model retrain — it is the "slow path" that the
fast serving path reads (blueprint Section 0.2).

Run:  python -m training.precompute_serving_cache
"""
from __future__ import annotations

import json
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
from src.forecast_engine.forecasting import registry
from src.forecast_engine.orchestrator import build_response

OFFERED_HORIZONS = [5, 10, 21]
MODEL_DIR = PROJECT_ROOT / "models" / "forecast" / "volatility_latest"
JUMP_PARAMS = PROJECT_ROOT / "models" / "risk_params" / "jump_diffusion.parquet"
OUT = PROJECT_ROOT / "models" / "serving_cache" / "serving_cache.parquet"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Precompute the serving cache.")
    parser.add_argument(
        "--model-version", default=registry.LATEST_POINTER,
        help="Model version dir under models/forecast/ to serve "
             "(default: volatility_latest). Use registry.list_versions() to see options.",
    )
    args = parser.parse_args()

    if not JUMP_PARAMS.exists():
        raise SystemExit(f"No jump params at {JUMP_PARAMS} — run fit_jump_diffusion_params first.")

    # Load the requested model VERSION via the registry (not a hardcoded path), so the
    # cache is built from a known, named model version and that version is recorded.
    try:
        model = registry.load_model(args.model_version)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    model_version = model.version
    print(f"Model version: {model_version}")
    jump = pd.read_parquet(JUMP_PARAMS).set_index("symbol")

    # Real parameter-store version, stamped by fit_jump_diffusion_params (not a
    # hardcoded constant). Falls back to "unversioned" if an old params file predates
    # versioning, so the provenance in every response is honest.
    if "risk_params_version" in jump.columns:
        risk_params_version = str(jump["risk_params_version"].iloc[0])
    else:
        risk_params_version = "unversioned"
    print(f"Risk params version: {risk_params_version}")

    panel = load_processed_features()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    symbols = sorted(panel["symbol"].unique())
    print(f"Precomputing cache for {len(symbols)} tickers x {len(OFFERED_HORIZONS)} horizons...")

    rows = []
    skipped = 0
    for i, sym in enumerate(symbols, 1):
        feats = panel[panel["symbol"] == sym]
        c = feats.sort_values("date")["adj_close"].astype(float)
        returns = np.log(c / c.shift(1)).dropna().to_numpy()
        last_price = float(c.iloc[-1])
        as_of = feats["date"].max().date().isoformat()

        # fan-chart params (may be missing for a few short-history names)
        jrow = jump.loc[sym] if sym in jump.index else None

        for h in OFFERED_HORIZONS:
            resp = build_response(
                ticker=sym, as_of=as_of, volatility_model=model,
                ticker_features=feats, daily_log_returns=returns,
                horizon_days=h, risk_params_version=risk_params_version,
            )
            risk = resp["risk"]
            if risk is None:
                skipped += 1
                continue
            rec = {
                "symbol": sym,
                "horizon": h,
                "as_of": as_of,
                "model_version": model_version,
                "risk_params_version": risk_params_version,
                "predicted_volatility": resp["forecast"].get("predicted_volatility"),
                "var_95": risk["var_95"],
                "cvar_95": risk["cvar_95"],
                "vol_window": risk["vol_window"],
                "n_samples": risk["n_samples"],
                "percentiles": json.dumps(risk["percentiles"]),
                "last_price": last_price,
                # fan inputs (None-safe)
                "sigma_annual": float(jrow["sigma_annual"]) if jrow is not None else None,
                "lambda_per_year": float(jrow["lambda_per_year"]) if jrow is not None else None,
                "jump_mean": float(jrow["jump_mean"]) if jrow is not None else None,
                "jump_std": float(jrow["jump_std"]) if jrow is not None else None,
            }
            rows.append(rec)

        if i % 50 == 0:
            print(f"  {i}/{len(symbols)} tickers done")

    cache = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(OUT, index=False)
    size_kb = OUT.stat().st_size / 1024
    print(f"\nSaved cache: {len(cache):,} rows ({cache['symbol'].nunique()} tickers), "
          f"{size_kb:.0f} KB -> {OUT}")
    if skipped:
        print(f"  ({skipped} ticker-horizons skipped for insufficient history)")
    print("This file is small enough to commit and deploy. The app reads it instead "
          "of the full panel.")


if __name__ == "__main__":
    main()
