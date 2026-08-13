"""Schema invariants — the vol feature list must exclude sparse/dead features."""
from src.forecast_engine.features import schema


def test_vol_features_exclude_fundamentals_si_macro():
    vol = set(schema.VOLATILITY_FEATURE_NAMES)
    assert not (vol & set(schema.FUNDAMENTAL_FEATURE_NAMES)), "fundamentals leaked into vol features"
    assert not (vol & set(schema.SHORT_INTEREST_FEATURE_NAMES)), "short interest leaked in"
    assert not (vol & set(schema.MACRO_FEATURE_NAMES)), "macro leaked in"


def test_vol_features_nonempty_and_technical():
    assert len(schema.VOLATILITY_FEATURE_NAMES) > 20
    for f in ("macd", "rsi_14", "atr_14", "realized_vol_20_annualized"):
        assert f in schema.VOLATILITY_FEATURE_NAMES


def test_vol_features_exclude_nonstationary_levels():
    vol = set(schema.VOLATILITY_FEATURE_NAMES)
    for level in ("sma_50", "sma_200", "ema_12", "vwap_20"):
        assert level not in vol
