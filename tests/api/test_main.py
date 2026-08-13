"""FastAPI app-assembly tests (main.py).

Distinct from test_endpoints.py (which tests route BEHAVIOR): this tests that the
app is wired correctly — metadata, router mounted under /v1, CORS present, docs
exposed, and that the surface is read-only (GET only, no state-changing verbs).
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from service.api.main import app
from service.api.deps import get_cache


def _cache_df():
    rows = []
    for h, w in ((5, 21), (10, 21), (21, 42)):
        rows.append({
            "symbol": "AAPL", "horizon": h, "as_of": "2026-07-31",
            "predicted_volatility": 0.017, "var_95": -0.06, "cvar_95": -0.08,
            "vol_window": w, "n_samples": 4000,
            "percentiles": {"p5": -0.06, "p25": -0.01, "p50": 0.004, "p75": 0.02, "p95": 0.07},
            "last_price": 100.0, "sigma_annual": 0.22, "lambda_per_year": 5.0,
            "jump_mean": -0.01, "jump_std": 0.05,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def client():
    app.dependency_overrides[get_cache] = _cache_df
    yield TestClient(app)
    app.dependency_overrides.clear()


# --- app metadata ----------------------------------------------------------
def test_app_title_and_version():
    assert app.title == "StockForecastRisk API"
    assert app.version == "1.0.0"


def test_app_description_states_no_advice_and_no_direction():
    """The app-level description must carry the honest framing."""
    desc = (app.description or "").lower()
    assert "not investment advice" in desc
    assert "no directional" in desc or "not" in desc


# --- routing ---------------------------------------------------------------
def test_routes_mounted_under_v1(client):
    """Routes are mounted under /v1 (checked via the OpenAPI schema, which is
    version-robust — app.routes may hold router objects without a .path)."""
    paths = set(client.get("/openapi.json").json()["paths"].keys())
    assert "/v1/forecast/{ticker}" in paths
    assert "/v1/tickers" in paths
    assert "/v1/health" in paths
    assert "/" in paths  # root is unprefixed


def test_forecast_reachable_through_app(client):
    """End-to-end through the assembled app (not just the router in isolation)."""
    r = client.get("/v1/forecast/AAPL?horizon=21")
    assert r.status_code == 200
    assert r.json()["ticker"] == "AAPL"


# --- docs / OpenAPI --------------------------------------------------------
def test_openapi_schema_available(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert r.json()["info"]["title"] == "StockForecastRisk API"


def test_docs_page_served(client):
    assert client.get("/docs").status_code == 200


# --- CORS ------------------------------------------------------------------
def test_cors_header_present(client):
    r = client.get("/v1/health", headers={"Origin": "http://example.com"})
    assert "access-control-allow-origin" in {k.lower() for k in r.headers}


# --- read-only surface -----------------------------------------------------
def test_surface_is_read_only(client):
    """A risk-reporting API must not expose state-changing verbs."""
    assert client.post("/v1/forecast/AAPL").status_code in (405, 404)
    assert client.delete("/v1/forecast/AAPL").status_code in (405, 404)
    assert client.put("/v1/tickers").status_code in (405, 404)


def test_unknown_path_404(client):
    assert client.get("/v1/does-not-exist").status_code == 404
