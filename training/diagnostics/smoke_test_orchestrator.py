"""Orchestrator end-to-end smoke test.

Proves the engine emits its honest JSON contract on real data, and — the point
of running it — exercises the two seams we could only reason about otherwise:

  1. Feature selection: does the loaded 34-feature volatility model line up with
     the columns in the processed panel? (predict path guards on feature_names.)
  2. Risk assembly: does historical_var.estimate_risk produce a coherent block
     from a real ticker's return history, now that the import is fixed?

It does NOT retrain — it loads the saved model from training. Run training first
(train_forecast_models.py) so models/forecast/volatility_latest exists.

Run:  python -m training.smoke_test_orchestrator  --ticker AAPL
"""
from __future__ import annotations

import argparse
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
from src.forecast_engine.features.schema import VOLATILITY_FEATURE_NAMES
from src.forecast_engine.forecasting.gbm_boost import VolatilityModel
from src.forecast_engine.orchestrator import build_response  # adjust path if needed


def daily_log_returns_for(frame: pd.DataFrame) -> np.ndarray:
    """Derive the daily log-return series the risk block needs from adj_close.

    The processed panel carries adj_close but not a log_return column, so we
    compute it here (single ticker, already date-sorted).
    """
    close = frame.sort_values("date")["adj_close"].astype(float)
    return np.log(close / close.shift(1)).dropna().to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--model-dir", type=Path,
                        default=PROJECT_ROOT / "models" / "forecast" / "volatility_latest")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Risk horizon; defaults to historical_var.DEFAULT_HORIZON.")
    args = parser.parse_args()

    # --- load the saved model (not retrain) -------------------------------
    if not args.model_dir.exists():
        raise SystemExit(f"No saved model at {args.model_dir}. Run training first.")
    model = VolatilityModel.load(args.model_dir)
    print(f"Loaded model     : {model.version}")
    print(f"Model features   : {len(model.feature_names)}")

    # --- seam check 1: model features vs. canonical vs. panel columns -----
    canonical = set(VOLATILITY_FEATURE_NAMES)
    model_feats = set(model.feature_names)
    if model_feats != canonical:
        print("WARNING: loaded model's feature set != VOLATILITY_FEATURE_NAMES")
        print(f"  only in model    : {sorted(model_feats - canonical)}")
        print(f"  only in canonical: {sorted(canonical - model_feats)}")
    else:
        print("Feature parity   : model matches VOLATILITY_FEATURE_NAMES exactly")

    # --- load one ticker's features ---------------------------------------
    features = load_processed_features(symbols=args.ticker)
    if features.empty:
        raise SystemExit(f"No rows for ticker {args.ticker!r} in the processed panel.")
    features["date"] = pd.to_datetime(features["date"])
    features = features.sort_values("date").reset_index(drop=True)
    print(f"Ticker rows      : {len(features):,} "
          f"({features['date'].min().date()} -> {features['date'].max().date()})")

    missing = [c for c in model.feature_names if c not in features.columns]
    if missing:
        raise SystemExit(f"Panel is missing model feature columns: {missing}")
    print("Column check     : all model features present in the panel")

    # --- risk input -------------------------------------------------------
    returns = daily_log_returns_for(features)
    print(f"Return history   : {len(returns):,} daily log returns")

    # --- the actual end-to-end call ---------------------------------------
    kwargs = {}
    if args.horizon is not None:
        kwargs["horizon_days"] = args.horizon
    response = build_response(
        ticker=args.ticker,
        as_of=features["date"].max().date(),
        volatility_model=model,
        ticker_features=features,
        daily_log_returns=returns,
        risk_params_version="smoke-test",
        **kwargs,
    )

    print("\n=== ENGINE RESPONSE ===")
    print(json.dumps(response, indent=2, default=str))

    # --- honesty-contract assertions --------------------------------------
    print("\n=== CONTRACT CHECKS ===")
    fc = response["forecast"]
    checks = {
        "no predicted_return field": "predicted_return" not in fc,
        "no predicted_direction field": "predicted_direction" not in fc,
        "return_predictability == none": fc.get("return_predictability") == "none",
        "predicted_volatility present": "predicted_volatility" in fc,
        "risk block present": response["risk"] is not None,
        "model_versions present": "model_versions" in response,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if response["risk"] is not None:
        r = response["risk"]
        print(f"\n  VaR95 {r.get('var_95')}  CVaR95 {r.get('cvar_95')}  "
              f"horizon {r.get('horizon_days')}d  n={r.get('n_samples')}")

    if not all(checks.values()):
        sys.exit(1)
    print("\nAll contract checks passed.")


if __name__ == "__main__":
    main()
