"""API services — turn a cache row into the honest response contract.

All the honesty rules live here: central estimate is the risk median, no directional
fields, the disclosed breach rate must match the engine's actual vol_window, and the
horizon must be one of the validated set (enforced again here as defence-in-depth
even though the schema already restricts it).
"""
from __future__ import annotations

import pandas as pd

from .deps import disclosure_for
from .schemas import (
    ForecastBlock,
    ForecastResponse,
    ModelVersions,
    RiskBlock,
)


class TickerNotFound(Exception):
    pass


class HorizonNotAvailable(Exception):
    pass


class CalibrationMismatch(Exception):
    """Engine vol_window disagrees with the disclosed figure's window — never serve."""


def build_forecast_response(cache: pd.DataFrame, ticker: str, horizon: int) -> ForecastResponse:
    row = cache[(cache["symbol"] == ticker) & (cache["horizon"] == horizon)]
    if row.empty:
        # Distinguish "unknown ticker" from "ticker exists but not at this horizon".
        if ticker not in set(cache["symbol"]):
            raise TickerNotFound(ticker)
        raise HorizonNotAvailable(horizon)
    r = row.iloc[0]

    disc = disclosure_for(horizon)
    engine_window = int(r["vol_window"])
    # Honesty guard: the disclosed breach rate was measured at disc["window"]. If the
    # engine used a different window, the disclosure would describe a different
    # computation than the VaR — refuse rather than mislead.
    if disc and engine_window != disc["window"]:
        raise CalibrationMismatch(
            f"{ticker} {horizon}d: engine window {engine_window} != disclosed "
            f"window {disc['window']}"
        )

    pct = r["percentiles"]
    breach = disc.get("breach", "n/a")
    crisis = disc.get("crisis", "unknown")

    forecast = ForecastBlock(
        predicted_volatility=(None if pd.isna(r["predicted_volatility"])
                              else float(r["predicted_volatility"])),
        central_estimate=float(pct["p50"]),
    )
    risk = RiskBlock(
        horizon_days=horizon,
        percentiles={k: float(v) for k, v in pct.items()},
        var_95=float(r["var_95"]),
        cvar_95=float(r["cvar_95"]),
        vol_window=engine_window,
        n_samples=int(r["n_samples"]),
        breach_rate_disclosed=breach,
        calibration_note=(
            f"Pooled {horizon}-day 95% VaR breach rate {breach} vs 5% target "
            f"(2010–2026, point-in-time, vol-window {engine_window}). Calibrated in "
            f"normal conditions; modestly understates severe stress (crisis-year "
            f"breach {crisis})."
        ),
    )
    return ForecastResponse(
        ticker=ticker,
        as_of=str(r["as_of"]),
        forecast=forecast,
        risk=risk,
        model_versions=ModelVersions(
            cache_as_of=str(r["as_of"]),
            model_version=str(r["model_version"]) if "model_version" in r and pd.notna(r["model_version"]) else "unversioned",
            risk_params_version=str(r["risk_params_version"]) if "risk_params_version" in r and pd.notna(r["risk_params_version"]) else "unversioned",
        ),
    )


def list_tickers(cache: pd.DataFrame) -> list[str]:
    return sorted(cache["symbol"].unique().tolist())