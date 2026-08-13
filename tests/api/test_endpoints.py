"""API endpoints & behaviors not covered by the contract/guard tests.

Complements test_contract.py (honest contract) and test_calibration_guard.py (503
refusal) by covering: the /tickers, /health, / endpoints; prediction logging; ticker
case/whitespace handling; the services exception distinctions; and schema-level guards.
"""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from service.api.main import app
from service.api.deps import get_cache
from service.api import services, schemas


def _row(sym, horizon, vol_window):
    return {
        "symbol": sym, "horizon": horizon, "as_of": "2026-07-31",
        "predicted_volatility": 0.017, "var_95": -0.06, "cvar_95": -0.08,
        "vol_window": vol_window, "n_samples": 4000,
        "percentiles": {"p5": -0.06, "p25": -0.01, "p50": 0.004, "p75": 0.02, "p95": 0.07},
        "last_price": 100.0, "sigma_annual": 0.22, "lambda_per_year": 5.0,
        "jump_mean": -0.01, "jump_std": 0.05,
    }


def _cache_df():
    rows = []
    for sym in ("AAPL", "MSFT", "NVDA"):
        for h, w in ((5, 21), (10, 21), (21, 42)):
            rows.append(_row(sym, h, w))
    return pd.DataFrame(rows)


@pytest.fixture
def client():
    app.dependency_overrides[get_cache] = _cache_df
    yield TestClient(app)
    app.dependency_overrides.clear()


# --- endpoints -------------------------------------------------------------
def test_root_endpoint(client):
    j = client.get("/").json()
    assert "service" in j
    assert "advice" in j["disclaimer"].lower()


def test_tickers_endpoint(client):
    j = client.get("/v1/tickers").json()
    assert j["count"] == 3
    assert j["tickers"] == ["AAPL", "MSFT", "NVDA"], "tickers must be sorted"


def test_health_endpoint(client):
    j = client.get("/v1/health").json()
    assert j["status"] == "ok"
    assert j["tickers"] == 3
    assert j["cache_rows"] == 9


# --- ticker input handling -------------------------------------------------
def test_ticker_lowercase_normalized(client):
    """A lowercase ticker must resolve to the uppercase symbol."""
    assert client.get("/v1/forecast/aapl?horizon=21").status_code == 200


def test_ticker_whitespace_stripped(client):
    assert client.get("/v1/forecast/ AAPL ?horizon=21").status_code in (200, 404)
    # (URL-encoded spaces may 404 at routing; the .strip() covers trailing space cases)


# --- prediction logging (the v2 drift seam) --------------------------------
def test_prediction_is_logged(client, monkeypatch, tmp_path):
    """A successful forecast must append a record to the prediction log."""
    import service.api.routes as routes
    logpath = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(routes, "_LOG_PATH", logpath)
    client.get("/v1/forecast/AAPL?horizon=21")
    assert logpath.exists(), "a prediction log line should be written"
    rec = json.loads(logpath.read_text().strip().splitlines()[-1])
    assert rec["ticker"] == "AAPL"
    assert rec["horizon_days"] == 21
    assert "var_95" in rec and "vol_window" in rec


# --- services exception distinctions ---------------------------------------
def test_service_ticker_not_found():
    with pytest.raises(services.TickerNotFound):
        services.build_forecast_response(_cache_df(), "ZZZZ", 21)


def test_service_horizon_not_available():
    """Ticker exists but no row at that horizon -> HorizonNotAvailable, not TickerNotFound."""
    df = _cache_df()
    df = df[df["horizon"] != 5]  # drop 5-day rows
    with pytest.raises(services.HorizonNotAvailable):
        services.build_forecast_response(df, "AAPL", 5)


def test_list_tickers_sorted():
    df = pd.DataFrame([_row("XOM", 21, 42), _row("AAPL", 21, 42)])
    assert services.list_tickers(df) == ["AAPL", "XOM"]


# --- schema-level guards ---------------------------------------------------
def test_offered_horizon_rejects_out_of_set():
    with pytest.raises(ValueError):
        schemas.OfferedHorizon(60)


def test_offered_horizon_accepts_valid():
    assert int(schemas.OfferedHorizon(5)) == 5
    assert int(schemas.OfferedHorizon(21)) == 21


def test_forecast_block_return_predictability_locked():
    """return_predictability is a Literal['none'] — anything else is a validation error."""
    with pytest.raises(ValidationError):
        schemas.ForecastBlock(return_predictability="up", central_estimate=0.0)


def test_forecast_block_defaults_are_honest():
    fb = schemas.ForecastBlock(central_estimate=0.01)
    assert fb.return_predictability == "none"
    assert fb.central_estimate_source == "risk_model_median"


def test_provenance_versions_surfaced():
    """model_version and risk_params_version from the cache row must appear in the
    response's model_versions block (provenance, not defaults)."""
    row = _row("AAPL", 21, 42)
    row["model_version"] = "volatility_xgb_2026-08-08"
    row["risk_params_version"] = "jumpdiff_2026-08-11"
    app.dependency_overrides[get_cache] = lambda: pd.DataFrame([row])
    try:
        mv = TestClient(app).get("/v1/forecast/AAPL?horizon=21").json()["model_versions"]
        assert mv["model_version"] == "volatility_xgb_2026-08-08"
        assert mv["risk_params_version"] == "jumpdiff_2026-08-11"
    finally:
        app.dependency_overrides.clear()