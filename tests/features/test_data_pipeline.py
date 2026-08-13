"""Data-pipeline invariants — symbol normalization, FINRA parsing, regimes, IO.

Several of these guard bugs actually hit during the project: the dot-to-dash symbol
normalization (share-class tickers failing to merge) and the FINRA short-interest
parser's tolerance of column-name variants.
"""
import numpy as np
import pandas as pd
import pytest

# --- short interest --------------------------------------------------------
from src.forecast_engine.data.short_interest import (
    normalize_symbol,
    parse_finra_short_interest,
)


def test_symbol_dot_to_dash():
    """FINRA writes BRK.B; the price panel uses BRK-B. Must normalize to dash."""
    assert normalize_symbol("BRK.B") == "BRK-B"
    assert normalize_symbol("bf.b") == "BF-B"
    assert normalize_symbol(" aapl ") == "AAPL"


def test_finra_parser_basic():
    raw = "symbolCode|settlementDate|currentShortPositionQuantity|averageDailyVolumeQuantity\n"
    raw += "AAPL|20240115|1000000|5000000\n"
    raw += "BRK.B|20240115|2000|10000\n"
    df = parse_finra_short_interest(raw)
    assert set(["symbol", "settlement_date", "short_interest"]).issubset(df.columns)
    assert "BRK-B" in df["symbol"].values, "share-class symbol must be normalized"
    assert df.loc[df["symbol"] == "AAPL", "short_interest"].iloc[0] == 1_000_000


def test_finra_parser_tolerates_column_variants():
    """Different historical FINRA column names should still parse."""
    raw = "symbol|settlementDate|shortInterest\nAAPL|20200101|500\n"
    df = parse_finra_short_interest(raw)
    assert len(df) == 1
    assert df["short_interest"].iloc[0] == 500


def test_finra_parser_missing_required_columns_raises():
    raw = "foo|bar\n1|2\n"
    with pytest.raises(ValueError):
        parse_finra_short_interest(raw)


def test_finra_parser_drops_unparseable_rows():
    raw = "symbol|settlementDate|shortInterest\nAAPL|20200101|500\nBAD|notadate|xyz\n"
    df = parse_finra_short_interest(raw)
    # the bad row (unparseable date + short interest) is dropped
    assert len(df) == 1
    assert df["symbol"].iloc[0] == "AAPL"


# --- regimes ---------------------------------------------------------------
from src.forecast_engine.evaluation.regimes import label_regimes


def test_label_regimes_up_down_flat():
    df = pd.DataFrame({"close": [100, 105, 100, 100.5]})  # +5%, -4.8%, +0.5%
    labels = label_regimes(df, threshold=0.01)
    assert labels.iloc[1] == "up"     # +5% > 1%
    assert labels.iloc[2] == "down"   # -4.8% < -1%
    assert labels.iloc[3] == "flat"   # +0.5% within +/-1%


# --- synthetic -------------------------------------------------------------
from src.forecast_engine.data.synthetic import generate_synthetic_stock_data


def test_synthetic_data_shape_and_positivity():
    df = generate_synthetic_stock_data(n_days=50, start_price=100.0)
    assert len(df) == 50
    # prices should be positive
    price_cols = [c for c in df.columns if c.lower() in ("open", "high", "low", "close")]
    for c in price_cols:
        assert (df[c] > 0).all(), f"{c} must be positive"


# --- store round-trip ------------------------------------------------------
from src.forecast_engine.data.store import save_dataframe, load_dataframe


def test_store_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "sub" / "df.parquet"
    save_dataframe(df, path)
    loaded = load_dataframe(path)
    pd.testing.assert_frame_equal(df, loaded)
