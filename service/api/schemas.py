"""API request/response schemas — the honest contract, enforced at the type level.

The response deliberately has NO predicted_direction / probability_up / predicted_return
fields. They cannot be added here without a schema change, which is the point: the
honest contract is enforced by the type, not by remembering not to populate a field.
The central estimate is the risk-model median; return_predictability is always "none".
"""
from __future__ import annotations

from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field


# Only the validated horizons are representable. A request for any other horizon
# (e.g. 60-day, which does not calibrate) cannot be constructed — the API rejects
# it at parse time rather than serving an uncalibrated VaR.
class OfferedHorizon(IntEnum):
    H5 = 5
    H10 = 10
    H21 = 21


class ForecastBlock(BaseModel):
    return_predictability: Literal["none"] = "none"
    predicted_volatility: float | None = Field(
        None, description="Forecast daily volatility (the validated signal)."
    )
    central_estimate: float = Field(
        ..., description="Central estimate = risk-model median. NOT a directional forecast."
    )
    central_estimate_source: Literal["risk_model_median"] = "risk_model_median"
    note: str = (
        "Modeled range of outcomes, not a prediction of direction. No validated "
        "directional edge at this horizon."
    )


class RiskBlock(BaseModel):
    model: Literal["filtered_historical_simulation"] = "filtered_historical_simulation"
    horizon_days: int
    confidence: float = 0.95
    percentiles: dict[str, float]
    var_95: float
    cvar_95: float
    vol_window: int = Field(..., description="Conditioning window used (per-horizon).")
    n_samples: int
    breach_rate_disclosed: str = Field(
        ..., description="Backtested pooled breach rate for this horizon/window."
    )
    calibration_note: str


class ModelVersions(BaseModel):
    risk: str = "filtered_historical_simulation"
    cache_as_of: str
    model_version: str = Field(
        ..., description="Volatility model version that produced this forecast."
    )
    risk_params_version: str = Field(
        ..., description="Jump-diffusion parameter fit version (provenance)."
    )


class ForecastResponse(BaseModel):
    ticker: str
    as_of: str
    forecast: ForecastBlock
    risk: RiskBlock
    model_versions: ModelVersions
    disclaimer: str = (
        "For research and educational use only. Not investment advice. Estimates "
        "carry uncertainty and may be wrong, especially in severe market stress."
    )


class TickerListResponse(BaseModel):
    tickers: list[str]
    count: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    cache_rows: int
    tickers: int