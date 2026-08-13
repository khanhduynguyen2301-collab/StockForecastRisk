"""Calibration-guard failure path — the API must REFUSE, not mislead.

The window-match guard is the safety net that keeps the disclosed breach figure from
describing a different computation than the VaR returned. This tests the guard's
FAILURE path: if a cache row's vol_window disagrees with the disclosed window, the
API returns 503 rather than serving a mismatched number.

Uses app.dependency_overrides (not monkeypatch) because FastAPI captures the
Depends() reference at import time — monkeypatching the module attribute does not
affect the already-wired dependency.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from service.api.main import app
from service.api.deps import get_cache


def _row(sym, horizon, vol_window):
    return {
        "symbol": sym, "horizon": horizon, "as_of": "2026-07-31",
        "predicted_volatility": 0.017, "var_95": -0.06, "cvar_95": -0.08,
        "vol_window": vol_window, "n_samples": 4000,
        "percentiles": {"p5": -0.06, "p25": -0.01, "p50": 0.004, "p75": 0.02, "p95": 0.07},
        "last_price": 100.0, "sigma_annual": 0.22, "lambda_per_year": 5.0,
        "jump_mean": -0.01, "jump_std": 0.05,
    }


@pytest.fixture
def client_with_mismatch():
    """Cache where AAPL 21d was computed at the WRONG window (21 instead of 42)."""
    df = pd.DataFrame([_row("AAPL", 21, vol_window=21)])  # disclosed 21d window is 42
    app.dependency_overrides[get_cache] = lambda: df
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client_ok():
    df = pd.DataFrame([_row("AAPL", 21, vol_window=42)])  # correct 21d window
    app.dependency_overrides[get_cache] = lambda: df
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_window_mismatch_returns_503(client_with_mismatch):
    r = client_with_mismatch.get("/v1/forecast/AAPL?horizon=21")
    assert r.status_code == 503, "a window mismatch must refuse to serve (503)"
    assert "calibration" in r.json()["detail"].lower()


def test_matching_window_serves_normally(client_ok):
    r = client_ok.get("/v1/forecast/AAPL?horizon=21")
    assert r.status_code == 200
    assert r.json()["risk"]["vol_window"] == 42
