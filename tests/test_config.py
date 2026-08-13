"""Config invariants — single source of truth, no stale values, agrees with engine."""
from src.forecast_engine.config import settings


def test_config_horizon_is_validated():
    assert settings.horizon in settings.offered_horizons
    assert settings.horizon != 7, "stale horizon=7 was never validated"


def test_offered_horizons_are_5_10_21():
    assert settings.offered_horizons == [5, 10, 21]


def test_config_window_map_matches_offered():
    for h in settings.offered_horizons:
        assert h in settings.horizon_vol_window
    assert settings.window_for(5) == 21
    assert settings.window_for(21) == 42


def test_config_disclosure_covers_offered_horizons():
    for h in settings.offered_horizons:
        assert h in settings.horizon_disclosure
        assert "breach" in settings.horizon_disclosure[h]


def test_config_window_map_agrees_with_engine():
    from src.forecast_engine.risk import filtered_historical_var as filt
    for h in settings.offered_horizons:
        assert settings.window_for(h) == filt.window_for_horizon(h)
