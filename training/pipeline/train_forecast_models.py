"""Train the pooled volatility model — the v1 shipping model.

This is the real training job (replaces the earlier placeholder). It:

  1. Loads the processed feature panel.
  2. Uses the CANONICAL volatility feature list from schema
     (`VOLATILITY_FEATURE_NAMES`) — base technical only. Fundamentals, short
     interest, and macro are excluded there by construction, so the model's
     NaN-drop in fit() cannot silently truncate the training panel.
  3. Validates out-of-sample with purged walk-forward, and — the gate that
     decides whether the model ships — checks it beats the persistence
     volatility baseline in every fold (per baselines.py / notebook 08b).
  4. Only if it passes, fits the final model on all history and saves it,
     version-stamped.

The horizon is explicit (`--vol-horizon`, default from config) and is used for
BOTH the target's forward window and the walk-forward purge gap — they must
match, or training targets leak into the test fold (splits.py enforces the purge
but cannot know your horizon).

Run:  python -m training.train_forecast_models
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure the project root is on the import path so `src` can be imported.
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
from src.forecast_engine.features.schema import (
    SCHEMA_VERSION,
    VOLATILITY_FEATURE_NAMES,
)
from src.forecast_engine.forecasting.gbm_boost import VolatilityModel, TARGET_COLUMN
from src.forecast_engine.evaluation.baselines import (
    persistence_volatility_baseline,
    beats_baseline,
)
from src.forecast_engine.evaluation.splits import walk_forward_splits

try:
    from src.forecast_engine.config import Settings
    _DEFAULT_HORIZON = Settings().horizon
except Exception:
    _DEFAULT_HORIZON = 5  # matches indicators.DEFAULT_TARGET_HORIZON


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def validate_walk_forward(
    data: pd.DataFrame,
    feature_names: list[str],
    vol_horizon: int,
    schema_version: str,
) -> pd.DataFrame:
    """Purged walk-forward validation of the vol model vs. the persistence baseline.

    Returns a per-fold frame with model MAE, baseline MAE, and whether the model
    beat the baseline (lower MAE). The purge gap equals the vol horizon so a
    training row's forward-vol window cannot overlap the test fold.
    """
    # Persistence baseline is the target's backward-looking twin, computed the
    # same way (trailing realised vol) — see baselines.py. Compute once, up front.
    baseline_full = persistence_volatility_baseline(
        data, window=vol_horizon, symbol_column="symbol", price_column="adj_close"
    )
    data = data.assign(_persistence_vol=baseline_full.to_numpy())

    usable = data.dropna(subset=feature_names + [TARGET_COLUMN, "_persistence_vol"]).reset_index(drop=True)
    if usable.empty:
        raise ValueError("No rows left after dropping NaNs in features/target/baseline.")

    rows = []
    fold = 0
    for train_idx, test_idx in walk_forward_splits(usable, purge_days=vol_horizon):
        fold += 1
        train = usable.loc[train_idx]
        test = usable.loc[test_idx]

        model = VolatilityModel(feature_names, schema_version=schema_version)
        model.fit(train, target=TARGET_COLUMN)
        pred = model.predict(test)

        y = test[TARGET_COLUMN].to_numpy()
        model_mae = _mae(y, pred)
        baseline_mae = _mae(y, test["_persistence_vol"].to_numpy())

        rows.append({
            "fold": fold,
            "train_rows": len(train),
            "test_rows": len(test),
            "model_mae": round(model_mae, 6),
            "baseline_mae": round(baseline_mae, 6),
            "beats_baseline": beats_baseline(model_mae, baseline_mae, lower_is_better=True),
        })

    if not rows:
        raise ValueError("Walk-forward produced no usable folds.")
    return pd.DataFrame(rows).set_index("fold")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pooled volatility model.")
    parser.add_argument("--vol-horizon", type=int, default=_DEFAULT_HORIZON,
                        help="Forward-vol horizon; also the walk-forward purge gap.")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "models" / "forecast" / "volatility_latest",
                        help="Directory to save the fitted model.")
    parser.add_argument("--skip-gate", action="store_true",
                        help="Fit and save even if the model does not beat the baseline "
                             "in every fold (NOT for shipping — diagnostics only).")
    args = parser.parse_args()

    print(f"Schema version         : {SCHEMA_VERSION}")
    print(f"Vol feature count      : {len(VOLATILITY_FEATURE_NAMES)}")
    print(f"Vol horizon / purge    : {args.vol_horizon}")

    data = load_processed_features()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"Loaded panel           : {len(data):,} rows, {data['symbol'].nunique()} symbols, "
          f"{data['date'].min().date()} -> {data['date'].max().date()}")

    # --- validation gate --------------------------------------------------
    folds = validate_walk_forward(
        data, VOLATILITY_FEATURE_NAMES, args.vol_horizon, SCHEMA_VERSION
    )
    print("\nWalk-forward validation (vs. persistence baseline):")
    print(folds.to_string())

    all_beat = bool(folds["beats_baseline"].all())
    mean_model = folds["model_mae"].mean()
    mean_base = folds["baseline_mae"].mean()
    print(f"\n  mean model MAE   : {mean_model:.6f}")
    print(f"  mean baseline MAE: {mean_base:.6f}")
    print(f"  beats baseline   : {folds['beats_baseline'].sum()}/{len(folds)} folds"
          f"  -> {'PASS' if all_beat else 'FAIL'}")

    if not all_beat and not args.skip_gate:
        print("\nGATE FAILED: model does not beat persistence in every fold. "
              "Not saving. Investigate before shipping (use --skip-gate to override "
              "for diagnostics only).")
        sys.exit(1)

    # --- fit final model on all history and save --------------------------
    print("\nFitting final model on all available history...")
    final = VolatilityModel(VOLATILITY_FEATURE_NAMES, schema_version=SCHEMA_VERSION)
    final.fit(data, target=TARGET_COLUMN)
    saved_to = final.save(args.output)
    print(f"Saved model            : {saved_to}")
    print(f"Model version          : {final.version}")
    print(f"Trained on rows        : {final.metadata.n_train_rows:,}")

    # A row-count sanity line: if this is far below the panel size, a feature is
    # dragging NaN-drops through training (the bug this feature list prevents).
    frac = final.metadata.n_train_rows / len(data)
    print(f"Training-row retention : {frac:.1%} of panel"
          f"  {'(healthy)' if frac > 0.6 else '(LOW - investigate NaN drops)'}")


if __name__ == "__main__":
    main()
