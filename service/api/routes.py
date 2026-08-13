"""API routes — the honest forecast endpoint plus ticker list and health.

Prediction logging is wired from the first request (nothing reads it yet; it is the
seam for v2 drift detection — capture from day one or there is no history later).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from .deps import get_cache
from .schemas import (
    ForecastResponse,
    HealthResponse,
    OfferedHorizon,
    TickerListResponse,
)
from .services import (
    CalibrationMismatch,
    HorizonNotAvailable,
    TickerNotFound,
    build_forecast_response,
    list_tickers,
)

router = APIRouter()

# --- prediction log (append-only; the v2 drift-detection seam) ---------------
_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "predictions.jsonl"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_pred_logger = logging.getLogger("predictions")


def _log_prediction(resp: ForecastResponse) -> None:
    """Append one prediction record. Never raises into the request path."""
    try:
        rec = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "ticker": resp.ticker,
            "as_of": resp.as_of,
            "horizon_days": resp.risk.horizon_days,
            "predicted_volatility": resp.forecast.predicted_volatility,
            "var_95": resp.risk.var_95,
            "cvar_95": resp.risk.cvar_95,
            "median": resp.forecast.central_estimate,
            "vol_window": resp.risk.vol_window,
            "risk_params_version": resp.model_versions.risk_params_version,
        }
        with _LOG_PATH.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:  # logging must never break serving
        pass


@router.get("/forecast/{ticker}", response_model=ForecastResponse)
def get_forecast(
    ticker: str,
    horizon: OfferedHorizon = OfferedHorizon.H21,
    cache: pd.DataFrame = Depends(get_cache),
) -> ForecastResponse:
    """Honest risk/volatility forecast for one ticker at a validated horizon.

    `horizon` is constrained to {5, 10, 21}; other values are rejected at parse time
    because only those calibrate. No directional prediction is returned.
    """
    ticker = ticker.upper().strip()
    try:
        resp = build_forecast_response(cache, ticker, int(horizon))
    except TickerNotFound:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")
    except HorizonNotAvailable:
        raise HTTPException(status_code=404,
                            detail=f"No estimate for {ticker} at {int(horizon)}-day horizon.")
    except CalibrationMismatch as e:
        # Refuse to serve a number whose disclosure doesn't match its computation.
        raise HTTPException(status_code=503, detail=f"Calibration guard tripped: {e}")
    _log_prediction(resp)
    return resp


@router.get("/tickers", response_model=TickerListResponse)
def get_tickers(cache: pd.DataFrame = Depends(get_cache)) -> TickerListResponse:
    ts = list_tickers(cache)
    return TickerListResponse(tickers=ts, count=len(ts))


@router.get("/health", response_model=HealthResponse)
def health(cache: pd.DataFrame = Depends(get_cache)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        cache_rows=len(cache),
        tickers=int(cache["symbol"].nunique()),
    )