"""Monte-Carlo simulator invariants (fan chart only — never a VaR source).

The simulator feeds the illustrative fan and histogram. These tests pin the shape,
the widening of dispersion over the horizon, and — the acceptance-checklist gate —
that the jump compensator keeps terminal prices unbiased and that zero-vol/zero-jump
reduces to a clean drift.
"""
import numpy as np
import pytest

from src.forecast_engine.risk.monte_carlo import (
    JumpDiffusionParams,
    simulate_paths,
    fan_chart_percentiles,
    terminal_distribution,
)


def _params(mu=0.0, sigma=0.2, lam=5.0, jmean=-0.02, jstd=0.05):
    return JumpDiffusionParams(mu_annual=mu, sigma_annual=sigma,
                               lambda_per_year=lam, jump_mean=jmean, jump_std=jstd)


def test_output_shape():
    paths = simulate_paths(100.0, _params(), horizon_days=21, n_paths=500)
    assert paths.shape == (500, 21)


def test_dispersion_widens_over_horizon():
    """Cross-path spread must grow with time (uncertainty compounds)."""
    paths = simulate_paths(100.0, _params(), horizon_days=60, n_paths=2000)
    early_std = paths[:, 2].std()
    late_std = paths[:, -1].std()
    assert late_std > early_std, "dispersion should widen over the horizon"


def test_fan_percentiles_ordered():
    paths = simulate_paths(100.0, _params(), horizon_days=21, n_paths=2000)
    fan = fan_chart_percentiles(paths)
    # at the final step, percentiles must be ordered
    assert fan["p5"][-1] < fan["p50"][-1] < fan["p95"][-1]


def test_compensator_keeps_mean_unbiased():
    """With mu=0 and the jump compensator, mean terminal price should stay ~initial,
    NOT be dragged down by the negative jump mean. This is the compensator gate."""
    paths = simulate_paths(100.0, _params(mu=0.0, jmean=-0.05, lam=10.0),
                           horizon_days=252, n_paths=20000)
    mean_terminal = paths[:, -1].mean()
    # within ~3% of initial — compensator offsets the asymmetric jumps
    assert abs(mean_terminal - 100.0) < 3.0, \
        f"compensator failed: mean terminal {mean_terminal:.2f} drifted from 100"


def test_zero_vol_zero_jump_is_deterministic_drift():
    """No diffusion, no jumps -> every path identical (pure drift)."""
    p = JumpDiffusionParams(mu_annual=0.10, sigma_annual=0.0,
                            lambda_per_year=0.0, jump_mean=0.0, jump_std=0.0)
    paths = simulate_paths(100.0, p, horizon_days=10, n_paths=50)
    # all paths equal (deterministic)
    assert np.allclose(paths, paths[0], rtol=1e-9), "zero vol/jump must be deterministic"
    # positive drift -> terminal above initial
    assert paths[0, -1] > 100.0


def test_higher_vol_widens_terminal_distribution():
    lo = simulate_paths(100.0, _params(sigma=0.10), horizon_days=21, n_paths=3000)
    hi = simulate_paths(100.0, _params(sigma=0.40), horizon_days=21, n_paths=3000)
    assert terminal_distribution(hi, 100.0).std() > terminal_distribution(lo, 100.0).std()


def test_terminal_distribution_is_returns():
    paths = simulate_paths(100.0, _params(), horizon_days=21, n_paths=500)
    term = terminal_distribution(paths, 100.0)
    # simple returns: (price/initial - 1), centered near 0 for mu=0
    assert term.shape == (500,)
    assert -1.0 < np.median(term) < 1.0
