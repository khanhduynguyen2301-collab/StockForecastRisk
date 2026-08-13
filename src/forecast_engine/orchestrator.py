"""Assemble the engine's per-ticker response from its two validated halves.

This is the seam where the forecast and risk components meet to produce the
response the service returns. Two design facts, both inherited from the research,
shape what this response can honestly contain:

1. The forecast is **volatility**, not return direction. Notebook 08 showed
   absolute return direction is not predictable from the available features, so
   the response does not claim a predicted return or direction. It reports the
   predictable quantity (volatility) and states the return finding plainly.

2. The risk numbers come from **historical simulation on horizon returns**
   (`risk/historical_var.py`), which calibrated at ~4.6% breach rate, not from
   the Monte Carlo path engine, which overstated the tail. The Monte Carlo
   percentiles may be attached as an illustrative fan chart, clearly labelled.

The orchestrator does no modelling itself. It calls the two components and shapes
their outputs into a stable contract, attaching version stamps so a response can
always be traced to the model and parameters that produced it.
"""
from __future__ import annotations
from pathlib import Path

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import sys

# Ensure the project root is on the import path so `src` can be imported.
PROJECT_ROOT = Path.cwd().resolve()
for parent in (PROJECT_ROOT, *PROJECT_ROOT.parents):
    if (parent / 'src').is_dir():
        PROJECT_ROOT = parent
        break
else:
    raise RuntimeError('Run this notebook from inside the StockForecastRisk repository.')

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecast_engine.risk import filtered_historical_var as historical_var
from src.forecast_engine.forecasting.gbm_boost import VolatilityModel


@dataclass
class ForecastResult:
    predicted_volatility: float | None
    volatility_model_version: str
    return_predictability: str = "none"          # honest: 08's null
    return_note: str = (
        "Absolute return direction is not predictable at this horizon from the "
        "available features; drift is set to risk-free in the risk model."
    )


def build_forecast_block(
    volatility_model: VolatilityModel,
    ticker_features: pd.DataFrame,
) -> ForecastResult:
    """Fill the `forecast` block for one ticker from its latest features.

    `ticker_features` is the feature rows for a single symbol; the model predicts
    from the most recent one. Returns a null volatility (not a fabricated number)
    if the features are insufficient, so the caller can decide how to surface it.
    """
    try:
        latest = volatility_model.predict_latest(ticker_features)
        predicted = float(latest.iloc[0]) if len(latest) else None
    except (ValueError, RuntimeError):
        predicted = None

    return ForecastResult(
        predicted_volatility=predicted,
        volatility_model_version=volatility_model.version,
    )


def build_risk_block(
    daily_log_returns: np.ndarray,
    horizon_days: int,
    confidence: float,
    risk_params_version: str,
) -> dict[str, Any] | None:
    """Fill the `risk` block from the calibrated historical-VaR method.

    Returns None when there is too little history to estimate risk, so the caller
    can emit a clear 'insufficient history' rather than fabricated numbers.
    """
    estimate = historical_var.estimate_risk(
        daily_log_returns, horizon=horizon_days, confidence=confidence
    )
    if estimate is None:
        return None

    return {
        "model": "filtered_historical_simulation",
        "horizon_days": estimate.horizon_days,
        "confidence": estimate.confidence,
        "percentiles": estimate.percentiles,
        "var_95": round(-estimate.var, 4),
        "cvar_95": round(-estimate.cvar, 4),
        "n_samples": estimate.n_samples,
        "vol_window": estimate.vol_window,
        "risk_params_version": risk_params_version,
        # Honest framing (5.7): vol-scaled historical simulation calibrates well in
        # normal conditions (pooled breach ~5.8% vs 5% target on 2010-2026) but still
        # modestly understates risk in severe stress (crisis-year breach ~8%). Surfaced
        # so the number is never read as a guarantee, especially in the tail.
        "calibration_note": (
            "Calibrated in normal conditions; may modestly understate risk in severe "
            "market stress (e.g. 2020, 2022-type regimes)."
        ),
    }


def build_response(
    ticker: str,
    as_of: date | str,
    volatility_model: VolatilityModel,
    ticker_features: pd.DataFrame,
    daily_log_returns: np.ndarray,
    horizon_days: int = historical_var.DEFAULT_HORIZON,
    confidence: float = historical_var.DEFAULT_CONFIDENCE,
    risk_params_version: str = "unversioned",
) -> dict[str, Any]:
    """Produce the full engine response for one ticker.

    The contract mirrors the intended API shape: `ticker`, `as_of`, a `forecast`
    block, a `risk` block, and `model_versions`. Where the research established a
    null (return direction), the response says so rather than inventing a number.
    """
    forecast = build_forecast_block(volatility_model, ticker_features)
    risk = build_risk_block(daily_log_returns, horizon_days, confidence, risk_params_version)

    response: dict[str, Any] = {
        "ticker": ticker,
        "as_of": as_of.isoformat() if isinstance(as_of, date) else as_of,
        "forecast": {
            "model": forecast.volatility_model_version,
            "predicted_volatility": (
                round(forecast.predicted_volatility, 4)
                if forecast.predicted_volatility is not None else None
            ),
            "return_predictability": forecast.return_predictability,
            "note": forecast.return_note,
        },
        "risk": risk,   # may be None -> caller surfaces "insufficient history"
        "model_versions": {
            "forecast": forecast.volatility_model_version,
            "risk_params": risk_params_version,
        },
    }
    return response