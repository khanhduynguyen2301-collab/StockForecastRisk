"""API dependencies — cache loading and shared lookups.

The API reads the precomputed serving cache (same source as the Streamlit deploy
app), so both surfaces serve identical numbers. The cache is loaded ONCE at startup
and held in memory; requests are lookups, not computation.

Paths and the per-horizon disclosure come from the project config (single source of
truth) so the API can never drift from what training/apps use.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

try:
    # Project config is the single source of truth for paths and disclosure.
    from src.forecast_engine.config import settings
    CACHE_PATH = Path(settings.serving_cache_path)
    HORIZON_DISCLOSURE = {
        h: {"breach": d["breach"], "crisis": d["crisis"], "window": int(d["window"])}
        for h, d in settings.horizon_disclosure.items()
    }
except Exception:
    # Fallback if config import path differs — keeps the API runnable standalone.
    _ROOT = Path(__file__).resolve()
    for parent in _ROOT.parents:
        if (parent / "models").is_dir() or (parent / "src").is_dir():
            _ROOT = parent
            break
    CACHE_PATH = _ROOT / "models" / "serving_cache" / "serving_cache.parquet"
    HORIZON_DISCLOSURE = {
        5:  {"breach": "5.4%", "crisis": "~7%",   "window": 21},
        10: {"breach": "5.5%", "crisis": "~7.5%", "window": 21},
        21: {"breach": "5.8%", "crisis": "~8%",   "window": 42},
    }


@lru_cache(maxsize=1)
def get_cache() -> pd.DataFrame:
    """Load the serving cache once (cached for the process lifetime)."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Serving cache not found at {CACHE_PATH}. Run "
            "training/precompute_serving_cache.py first."
        )
    df = pd.read_parquet(CACHE_PATH)
    df["percentiles"] = df["percentiles"].apply(json.loads)
    return df


def disclosure_for(horizon: int) -> dict:
    return HORIZON_DISCLOSURE.get(horizon, {})