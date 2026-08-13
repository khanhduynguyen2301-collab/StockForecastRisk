"""Fit Merton jump-diffusion parameters on the real panel — for the fan-chart visual.

IMPORTANT — these parameters drive the Monte Carlo fan chart ONLY, never the VaR.
The engine's risk numbers come from filtered_historical_var.py (vol-scaled historical
simulation). The Monte Carlo path engine overstates the compounded tail, so it is used
for the illustrative fan/histogram in the dashboard, not for any risk statement. This
job produces the params that visual consumes.

Fitting follows the module's validated design (notebook 09):
  - sigma and lambda are per-ticker (a per-name count estimates a rate acceptably),
  - jump_mean and jump_std are POOLED across all tickers (jumps are too rare per name).

Output: a parquet indexed by symbol with mu_annual, sigma_annual, mu_daily, sigma_daily,
lambda_per_year, jump_mean, jump_std, n_jumps, n_obs — the columns the JumpDiffusionParams
dataclass and simulate_paths consume.

Run:  python -m training.fit_jump_diffusion_params
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
from src.forecast_engine.risk.jump_diffusion import (
    fit_jump_diffusion_parameters,
    JUMP_THRESHOLD_K,
)
from src.forecast_engine.features.schema import SCHEMA_VERSION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "models" / "risk_params" / "jump_diffusion.parquet")
    args = parser.parse_args()

    panel = load_processed_features(columns=["symbol", "date", "adj_close"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["log_return"] = (
        panel.groupby("symbol")["adj_close"]
        .transform(lambda c: np.log(c.astype(float) / c.astype(float).shift(1)))
    )
    panel = panel.dropna(subset=["log_return"]).reset_index(drop=True)
    print(f"Panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols")

    params = fit_jump_diffusion_parameters(
        panel, symbol_column="symbol", return_column="log_return"
    )
    print(f"Fitted params for {len(params)} tickers")
    # jump_mean / jump_std are pooled -> identical across rows; report once.
    print(f"Pooled jump_mean : {params['jump_mean'].iloc[0]:.5f}")
    print(f"Pooled jump_std  : {params['jump_std'].iloc[0]:.5f}")
    print(f"sigma_annual     : median {params['sigma_annual'].median():.3f}, "
          f"range [{params['sigma_annual'].min():.3f}, {params['sigma_annual'].max():.3f}]")
    print(f"lambda_per_year  : median {params['lambda_per_year'].median():.2f} jumps/yr")

    # --- versioned parameter store -----------------------------------------
    # The blueprint requires a VERSIONED parameter store, not a bare overwrite. We
    # stamp the fit date and a version string into every row AND write a manifest,
    # so a served response can be traced to exactly which fitted params produced it
    # and a retrain is distinguishable from what it replaced.
    from datetime import date, datetime, timezone
    import json as _json

    fit_date = date.today().isoformat()
    version = f"jumpdiff_{fit_date}"
    out = params.reset_index()
    out["risk_params_version"] = version
    out["fit_date"] = fit_date
    out["schema_version"] = SCHEMA_VERSION
    out["jump_threshold_k"] = JUMP_THRESHOLD_K

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    manifest = {
        "risk_params_version": version,
        "fit_date": fit_date,
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "jump_threshold_k": JUMP_THRESHOLD_K,
        "n_tickers": int(len(params)),
        "pooled_jump_mean": float(params["jump_mean"].iloc[0]),
        "pooled_jump_std": float(params["jump_std"].iloc[0]),
        "artifact": str(args.output.name),
    }
    manifest_path = args.output.parent / "jump_diffusion_manifest.json"
    manifest_path.write_text(_json.dumps(manifest, indent=2))

    print(f"Saved -> {args.output}")
    print(f"Version -> {version}  (manifest: {manifest_path})")


if __name__ == "__main__":
    main()
