"""Epistemic (model) uncertainty for the volatility forecast.

The point volatility model (`gbm_boost.VolatilityModel`) answers "what is the
volatility?" but not "how sure is the model about *this* prediction?" Those are
different questions:

- **Aleatoric / market uncertainty** -- irreducible randomness in price movement.
  Already captured by the risk model's distribution.
- **Epistemic / model uncertainty** -- how confident the forecasting model is in
  this specific prediction. Larger where the features are unusual or the model
  has seen little similar data. Nothing in the point model captures this.

This module adds epistemic uncertainty the cheapest defensible way: **quantile
regression**. Instead of one point, fit the p10/p50/p90 of volatility (XGBoost's
`reg:quantileerror`). The p90-p10 spread is a per-prediction uncertainty band,
and it feeds the response's `confidence` field with something real behind it.

CRITICAL -- the honesty gate. A wide band is only meaningful if predictions with
wide bands are *actually* less accurate. Before trusting this anywhere (a
`confidence` field, a dashboard, gating logic), call `validate_uncertainty` and
confirm error rises with band width. If it does not, the band is noise; do not
wire it to anything. This mirrors the project's rule everywhere else: a signal
counts only after it beats a check.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

TARGET_COLUMN = "future_realized_vol"
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)

BASE_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)


@dataclass
class UncertaintyPrediction:
    p10: float
    p50: float
    p90: float

    @property
    def band_width(self) -> float:
        """Epistemic uncertainty: wider = the model is less sure here."""
        return self.p90 - self.p10

    @property
    def confidence(self) -> float:
        """A bounded 0-1 confidence, higher when the band is tighter relative to
        the central estimate. This is a monotone transform of the band, not a
        probability -- it exists to give the API a calibrated-in-spirit field
        rather than a bare number. Only meaningful once validate_uncertainty
        passes."""
        if self.p50 <= 0:
            return 0.0
        relative_band = self.band_width / abs(self.p50)
        return float(1.0 / (1.0 + relative_band))


class QuantileVolatilityModel:
    """XGBoost quantile-regression volatility model (p10/p50/p90).

    Same features and target as the point model, three heads instead of one.
    Fits on all history (validation lives in the notebooks / backtest).
    """

    def __init__(
        self,
        feature_names: Iterable[str],
        schema_version: str,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        params: dict | None = None,
    ):
        self.feature_names = list(feature_names)
        self.schema_version = schema_version
        self.quantiles = tuple(quantiles)
        self.params = params or dict(BASE_PARAMS)
        self._models: dict[float, XGBRegressor] = {}
        self._trained_on: str | None = None

    def fit(self, frame: pd.DataFrame, target: str = TARGET_COLUMN) -> "QuantileVolatilityModel":
        missing = [c for c in self.feature_names if c not in frame.columns]
        if missing:
            raise ValueError(f"Training frame missing features: {missing}")
        usable = frame.dropna(subset=self.feature_names + [target])
        if usable.empty:
            raise ValueError("No rows after dropping NaNs.")

        X, y = usable[self.feature_names], usable[target]
        for q in self.quantiles:
            model = XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=q, **self.params
            )
            model.fit(X, y)
            self._models[q] = model
        self._trained_on = date.today().isoformat()
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict each quantile; returns a frame with columns p10/p50/p90.

        Quantile crossing (p10 > p50 occasionally, an artefact of independent
        heads) is repaired by sorting each row's quantiles, so the band is always
        well-ordered.
        """
        if not self._models:
            raise RuntimeError("Model not fitted or loaded.")
        preds = {q: self._models[q].predict(frame[self.feature_names]) for q in self.quantiles}
        stacked = np.sort(np.column_stack([preds[q] for q in self.quantiles]), axis=1)
        return pd.DataFrame(
            stacked, columns=[f"p{int(q * 100)}" for q in self.quantiles], index=frame.index
        )

    def predict_one(self, feature_row: pd.DataFrame) -> UncertaintyPrediction:
        """Uncertainty prediction for a single row -- the serving path."""
        row = self.predict(feature_row).iloc[0]
        return UncertaintyPrediction(p10=float(row["p10"]), p50=float(row["p50"]), p90=float(row["p90"]))

    def save(self, directory: str | Path) -> Path:
        if not self._models:
            raise RuntimeError("Nothing to save.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for q, model in self._models.items():
            model.save_model(directory / f"q{int(q * 100)}.json")
        meta = dict(
            feature_names=self.feature_names, schema_version=self.schema_version,
            quantiles=list(self.quantiles), params=self.params, trained_on=self._trained_on,
        )
        (directory / "metadata.json").write_text(json.dumps(meta, indent=2))
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "QuantileVolatilityModel":
        directory = Path(directory)
        meta = json.loads((directory / "metadata.json").read_text())
        instance = cls(meta["feature_names"], meta["schema_version"],
                       tuple(meta["quantiles"]), meta["params"])
        for q in instance.quantiles:
            model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **meta["params"])
            model.load_model(directory / f"q{int(q * 100)}.json")
            instance._models[q] = model
        instance._trained_on = meta.get("trained_on")
        return instance

    @property
    def version(self) -> str:
        return f"volatility_quantile_{self._trained_on}" if self._trained_on else "unfitted"


def validate_uncertainty(
    predictions: pd.DataFrame,
    actuals: pd.Series,
    n_buckets: int = 4,
) -> dict:
    """The honesty gate: does prediction error actually rise with the band width?

    `predictions` must have p10/p50/p90 columns aligned with `actuals`. Buckets
    the rows by band width and reports mean absolute error per bucket plus the
    correlation between band width and error. `is_valid` is True only if error
    increases with uncertainty -- the condition for trusting the band.
    """
    band_width = (predictions["p90"] - predictions["p10"]).to_numpy()
    abs_error = np.abs(actuals.to_numpy() - predictions["p50"].to_numpy())

    frame = pd.DataFrame({"band_width": band_width, "abs_error": abs_error})
    frame["bucket"] = pd.qcut(frame["band_width"], n_buckets, labels=False, duplicates="drop")
    by_bucket = frame.groupby("bucket")["abs_error"].mean()

    correlation = float(np.corrcoef(band_width, abs_error)[0, 1])
    monotone_increasing = bool(by_bucket.is_monotonic_increasing)
    error_ratio = float(by_bucket.iloc[-1] / by_bucket.iloc[0]) if by_bucket.iloc[0] > 0 else float("inf")

    return {
        "correlation_width_error": correlation,
        "error_by_bucket": by_bucket.round(5).to_dict(),
        "widest_vs_narrowest_error_ratio": error_ratio,
        "monotone_increasing": monotone_increasing,
        # Valid if error clearly rises with the band: positive correlation and the
        # widest bucket is materially worse than the narrowest.
        "is_valid": correlation > 0.1 and error_ratio > 1.2,
    }