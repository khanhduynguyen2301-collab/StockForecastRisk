"""Shared pytest configuration and fixtures for the whole suite.

Puts the repo root on sys.path so `src` and `service` import regardless of where
pytest is invoked, and provides the common fixtures (a synthetic serving cache and a
deterministic returns series) that the API and risk tests share.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# --- make src / service importable from anywhere ---------------------------
_ROOT = Path(__file__).resolve().parent
for parent in (_ROOT, *_ROOT.parents):
    if (parent / "src").is_dir():
        _ROOT = parent
        break
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# --- shared fixtures -------------------------------------------------------
def _cache_row(symbol, horizon, vol_window):
    """One serving-cache row with the honest-contract shape."""
    return {
        "symbol": symbol, "horizon": horizon, "as_of": "2026-07-31",
        "predicted_volatility": 0.017, "var_95": -0.06, "cvar_95": -0.08,
        "vol_window": vol_window, "n_samples": 4000,
        "percentiles": {"p5": -0.06, "p25": -0.01, "p50": 0.004, "p75": 0.02, "p95": 0.07},
        "last_price": 100.0, "sigma_annual": 0.22, "lambda_per_year": 5.0,
        "jump_mean": -0.01, "jump_std": 0.05,
    }


@pytest.fixture
def cache_row():
    """Factory for a single cache row: cache_row('AAPL', 21, 42)."""
    return _cache_row


@pytest.fixture
def serving_cache():
    """A small, well-formed serving cache: 3 tickers x 3 horizons, correct windows."""
    rows = []
    for sym in ("AAPL", "MSFT", "NVDA"):
        for h, w in ((5, 21), (10, 21), (21, 42)):
            rows.append(_cache_row(sym, h, w))
    return pd.DataFrame(rows)


@pytest.fixture
def returns_series():
    """A long, mildly fat-tailed daily log-return series (deterministic)."""
    rng = np.random.default_rng(42)
    base = rng.normal(0.0003, 0.015, 4000)
    jumps = np.where(rng.random(4000) < 0.01, rng.normal(-0.03, 0.05, 4000), 0.0)
    return base + jumps


@pytest.fixture
def api_client(serving_cache):
    """A TestClient with the serving cache injected via dependency_overrides.

    Uses FastAPI's dependency_overrides (not monkeypatch): FastAPI captures the
    Depends() reference at import, so patching the module attribute does not work.
    """
    from fastapi.testclient import TestClient
    from service.api.main import app
    from service.api.deps import get_cache

    app.dependency_overrides[get_cache] = lambda: serving_cache
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
