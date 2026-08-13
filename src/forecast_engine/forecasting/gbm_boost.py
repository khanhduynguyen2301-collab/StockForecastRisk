"""Pooled volatility forecasting model.

This is the *forecast* half of the engine, and it forecasts **volatility**, not
returns. That distinction is the central finding of the project:

- Notebook 08 showed absolute return direction is not predictable from these
  features (naive baseline beats the models; directional accuracy on the
  coin-flip line). So the engine does not predict returns.
- Notebook 08b showed realised volatility *is* predictable (XGBoost beats a
  persistence baseline in every walk-forward fold, R^2 ~ 0.25-0.34, no leakage).
  So the engine predicts volatility, and that forecast feeds the risk envelope.

The model is pooled across all tickers, with ticker/sector encoded as features,
because per-ticker models would have far too little data. It uses the base
technical-feature set only: the macro (FRED) A/B test in 08b showed market-wide
macro factors *reduce* out-of-sample R^2 (Ridge -0.007, XGBoost -0.061), so they
are deliberately excluded here.

Why a class rather than functions: a trained model carries state (the fitted
booster, the feature list it was trained on, a version stamp) and is swappable
behind the registry, so it fits the project's "stateful/swappable -> class"
convention. The pure transforms live in `features/indicators.py`.
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

# Matches the walk-forward configuration validated in notebook 08b.
DEFAULT_PARAMS = dict(
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
class ModelMetadata:
    """Everything needed to reproduce and version a trained model."""
    schema_version: str
    feature_names: list[str]
    target: str
    trained_on: str          # ISO date the model was fitted
    n_train_rows: int
    params: dict


class VolatilityModel:
    """Pooled XGBoost forecaster for realised volatility.

    Usage:
        model = VolatilityModel(feature_names, schema_version="2.2")
        model.fit(train_df)
        preds = model.predict(serve_df)          # np.ndarray, one per row
        model.save("models/volatility_v1")
        model = VolatilityModel.load("models/volatility_v1")
    """

    def __init__(
        self,
        feature_names: Iterable[str],
        schema_version: str,
        params: dict | None = None,
    ):
        self.feature_names = list(feature_names)
        self.schema_version = schema_version
        self.params = params or dict(DEFAULT_PARAMS)
        self._model: XGBRegressor | None = None
        self._metadata: ModelMetadata | None = None

    # -- training -----------------------------------------------------------

    def fit(self, frame: pd.DataFrame, target: str = TARGET_COLUMN) -> "VolatilityModel":
        """Fit on a frame containing the feature columns and the target.

        No walk-forward here: this fits the final production model on all
        available history. Walk-forward validation lives in notebook 08b and in
        `training/backtest.py`; by the time we fit for serving, the modelling
        decision is already made and we want every row of signal.
        """
        missing = [c for c in self.feature_names if c not in frame.columns]
        if missing:
            raise ValueError(f"Training frame missing feature columns: {missing}")
        if target not in frame.columns:
            raise ValueError(f"Training frame missing target column: {target!r}")

        usable = frame.dropna(subset=self.feature_names + [target])
        if usable.empty:
            raise ValueError("No rows left after dropping NaNs in features/target.")

        model = XGBRegressor(**self.params)
        model.fit(usable[self.feature_names], usable[target])

        self._model = model
        self._metadata = ModelMetadata(
            schema_version=self.schema_version,
            feature_names=self.feature_names,
            target=target,
            trained_on=date.today().isoformat(),
            n_train_rows=len(usable),
            params=self.params,
        )
        return self

    # -- serving ------------------------------------------------------------

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict volatility for each row. Guards feature/schema alignment."""
        if self._model is None:
            raise RuntimeError("Model is not fitted or loaded.")
        missing = [c for c in self.feature_names if c not in frame.columns]
        if missing:
            raise ValueError(f"Serving frame missing feature columns: {missing}")
        return self._model.predict(frame[self.feature_names])

    def predict_latest(self, frame: pd.DataFrame, symbol_col: str = "symbol") -> pd.Series:
        """Predict for the most recent row per symbol -- the serving path.

        Returns a Series indexed by symbol. This is what the orchestrator calls
        to fill the `forecast` block: the latest available features per ticker.
        """
        latest = frame.sort_values("date").groupby(symbol_col).tail(1)
        predictions = self.predict(latest)
        return pd.Series(predictions, index=latest[symbol_col].values, name="predicted_volatility")

    def feature_importance(self) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model is not fitted or loaded.")
        return pd.Series(
            self._model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)

    # -- persistence --------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Persist the booster and metadata to a directory."""
        if self._model is None or self._metadata is None:
            raise RuntimeError("Nothing to save: model is not fitted.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._model.save_model(directory / "model.json")
        (directory / "metadata.json").write_text(json.dumps(asdict(self._metadata), indent=2))
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "VolatilityModel":
        """Reload a saved model, restoring the exact feature list and version."""
        directory = Path(directory)
        metadata = ModelMetadata(**json.loads((directory / "metadata.json").read_text()))
        instance = cls(
            feature_names=metadata.feature_names,
            schema_version=metadata.schema_version,
            params=metadata.params,
        )
        booster = XGBRegressor(**metadata.params)
        booster.load_model(directory / "model.json")
        instance._model = booster
        instance._metadata = metadata
        return instance

    @property
    def metadata(self) -> ModelMetadata | None:
        return self._metadata

    @property
    def version(self) -> str:
        """A version string for the response's model_versions block."""
        if self._metadata is None:
            return "unfitted"
        return f"volatility_xgb_{self._metadata.trained_on}"
