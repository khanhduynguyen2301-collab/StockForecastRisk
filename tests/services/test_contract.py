"""Honest-contract invariants at the API surface.

These are the guards that keep the API from ever serving something dishonest:
no directional fields, only calibrated horizons, and the calibration disclosure
matching the window actually used. If any of these fail, the API is shipping a
claim the project deliberately does not make.
"""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pathlib import Path
import sys

# Ensure the project root is on the import path so `service` can be imported.
PROJECT_ROOT = Path.cwd().resolve()
for parent in (PROJECT_ROOT, *PROJECT_ROOT.parents):
    if (parent / 'service').is_dir():
        PROJECT_ROOT = parent
        break
else:
    raise RuntimeError('Run this notebook from inside the StockForecastRisk repository.')

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.api.main import app
from service.api.deps import get_cache


@pytest.fixture(autouse=True)
def _cache():
    """Inject a tiny in-memory cache via FastAPI dependency_overrides (the correct
    mechanism — monkeypatching the module attr does not affect the captured Depends)."""
    rows = []
    for sym in ("AAPL", "MSFT"):
        for h, win in ((5, 21), (10, 21), (21, 42)):
            rows.append({
                "symbol": sym, "horizon": h, "as_of": "2026-07-31",
                "predicted_volatility": 0.017,
                "var_95": -0.06, "cvar_95": -0.08, "vol_window": win, "n_samples": 4000,
                "percentiles": {"p5": -0.06, "p25": -0.01, "p50": 0.004,
                                "p75": 0.02, "p95": 0.07},
                "last_price": 100.0, "sigma_annual": 0.22, "lambda_per_year": 5.0,
                "jump_mean": -0.01, "jump_std": 0.05,
            })
    df = pd.DataFrame(rows)
    app.dependency_overrides[get_cache] = lambda: df
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_no_directional_fields(client):
    """The forecast block must never carry direction/probability/return fields."""
    r = client.get("/v1/forecast/AAPL?horizon=21")
    assert r.status_code == 200
    fc = r.json()["forecast"]
    for banned in ("predicted_direction", "probability_up", "predicted_return"):
        assert banned not in fc, f"{banned} must not appear in the honest contract"


def test_return_predictability_is_none(client):
    r = client.get("/v1/forecast/AAPL?horizon=21")
    assert r.json()["forecast"]["return_predictability"] == "none"


def test_central_estimate_is_risk_median(client):
    j = client.get("/v1/forecast/AAPL?horizon=21").json()
    assert j["forecast"]["central_estimate_source"] == "risk_model_median"
    assert j["forecast"]["central_estimate"] == j["risk"]["percentiles"]["p50"]


@pytest.mark.parametrize("bad", [60, 30, 1, 252])
def test_uncalibrated_horizons_rejected(client, bad):
    """Only 5/10/21 are calibrated; anything else must be rejected at parse (422)."""
    r = client.get(f"/v1/forecast/AAPL?horizon={bad}")
    assert r.status_code == 422, f"horizon {bad} should be rejected, got {r.status_code}"


@pytest.mark.parametrize("h", [5, 10, 21])
def test_calibrated_horizons_served(client, h):
    assert client.get(f"/v1/forecast/AAPL?horizon={h}").status_code == 200


def test_disclosure_window_matches_engine(client):
    """The disclosed breach figure's window must equal the engine's vol_window."""
    for h, expected_win in ((5, 21), (10, 21), (21, 42)):
        j = client.get(f"/v1/forecast/AAPL?horizon={h}").json()
        assert j["risk"]["vol_window"] == expected_win


def test_var_cvar_coherent_in_response(client):
    j = client.get("/v1/forecast/AAPL?horizon=21").json()["risk"]
    assert j["cvar_95"] <= j["var_95"], "CVaR must be at least as negative as VaR"


def test_unknown_ticker_404(client):
    assert client.get("/v1/forecast/ZZZZ?horizon=21").status_code == 404


def test_disclaimer_present(client):
    assert "advice" in client.get("/v1/forecast/AAPL?horizon=21").json()["disclaimer"].lower()