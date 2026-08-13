"""Forecasting / risk-math correctness gates.

These pin the jump-diffusion fit behaviour and the empirical statistics used to
justify the jump model. They are the acceptance-checklist "risk math" gates in
executable form.
"""
import numpy as np
import pandas as pd
import pytest

from pathlib import Path
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

from src.forecast_engine.risk.jump_diffusion import (
    fit_diffusion_and_flag_jumps,
    fit_jump_diffusion_parameters,
    empirical_kurtosis,
)


def test_diffusion_fit_separates_jumps():
    """A series with injected large moves should flag them as jumps, leaving a
    calmer diffusion core."""
    rng = np.random.default_rng(0)
    core = rng.normal(0, 0.01, 1000)
    core[::100] = -0.15  # inject 10 large negative jumps
    fit = fit_diffusion_and_flag_jumps(core)
    assert fit is not None
    assert len(fit.jump_returns) >= 5, "should flag the injected jumps"
    # the diffusion sigma should be much smaller than the raw std (jumps removed)
    assert fit.sigma_daily < core.std(ddof=1)


def test_diffusion_fit_insufficient_history_returns_none():
    assert fit_diffusion_and_flag_jumps(np.array([0.01, -0.01])) is None


def test_empirical_kurtosis_fat_tails_positive():
    """Fat-tailed data should have excess kurtosis well above 0 (normal ~0)."""
    rng = np.random.default_rng(1)
    normal = rng.normal(0, 1, 5000)
    fat = np.concatenate([normal, rng.normal(0, 6, 200)])  # add heavy tails
    assert empirical_kurtosis(fat) > empirical_kurtosis(normal)


def test_fit_jump_diffusion_parameters_shape_and_pooling():
    """Per-ticker sigma/lambda, but jump_mean/jump_std pooled (identical across rows)."""
    rng = np.random.default_rng(2)
    rows = []
    for sym in ("A", "B", "C"):
        r = rng.normal(0.0003, 0.015, 1500)
        r[::150] = -0.12  # jumps
        for i, val in enumerate(r):
            rows.append({"symbol": sym, "log_return": val})
    df = pd.DataFrame(rows)
    params = fit_jump_diffusion_parameters(df)
    assert set(params.index) == {"A", "B", "C"}
    for col in ("sigma_annual", "lambda_per_year", "jump_mean", "jump_std"):
        assert col in params.columns
    # jump_mean / jump_std are pooled -> one unique value across tickers
    assert params["jump_mean"].nunique() == 1, "jump_mean must be pooled"
    assert params["jump_std"].nunique() == 1, "jump_std must be pooled"
    # sigma is per-ticker -> may differ
    assert (params["sigma_annual"] > 0).all()


def test_fit_jump_diffusion_no_valid_tickers_raises():
    df = pd.DataFrame({"symbol": ["A", "A"], "log_return": [0.01, -0.01]})
    with pytest.raises(ValueError):
        fit_jump_diffusion_parameters(df)