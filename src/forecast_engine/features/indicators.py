from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

# Ensure the project root is on the import path so `src` can be imported.
project_root = Path.cwd().resolve()
for candidate in (project_root, *project_root.parents):
    if (candidate / "src").exists():
        sys.path.insert(0, str(candidate))
        break

from src.forecast_engine.data.loader import (
    EARNINGS_CALENDAR_PATH,
    RAW_PRICES_PATH,
    load_earnings_calendar,
    load_raw_prices,
)


DEFAULT_LAG_DAYS = (1, 2, 3, 4, 5)
DEFAULT_SMA_WINDOWS = (10, 20, 50, 200)
DEFAULT_TARGET_HORIZON = 5
DEFAULT_MIN_HISTORY_ROWS = 504  # ~2 trading years; fills sma_200 and the 60-day beta window


def _to_snake(name: str) -> str:
    """Lowercase a name and collapse every non-alphanumeric run to one underscore."""
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def _column(frame: pd.DataFrame, name: str) -> str:
    """Return an existing column name, accepting common case and spacing variants.

    Ingestion writes snake_case ("adj_close", "gics_sector"), but vendor exports
    and older parquet files may still use title case with spaces ("Adj Close").
    Resolving both here means the feature code never has to care which it got.
    """
    snake = _to_snake(name)
    candidates = (name, snake, name.lower(), name.upper(), name.title())
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    # Fall back to matching any column whose snake_case form is equivalent.
    for column in frame.columns:
        if _to_snake(column) == snake:
            return column

    raise KeyError(f"Required column not found: {name}")


def _optional_column(frame: pd.DataFrame, name: str) -> str | None:
    try:
        return _column(frame, name)
    except KeyError:
        return None


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _feature_group(
    group: pd.DataFrame,
    price_col: str,
    close_col: str,
    high_col: str,
    low_col: str,
    volume_col: str | None,
    lag_days: tuple[int, ...],
    sma_windows: tuple[int, ...],
    include_williams_r: bool,
) -> pd.DataFrame:
    # Compute each symbol independently so no history crosses ticker boundaries.
    group = group.sort_values("date").copy()

    price = group[price_col].astype(float)
    close = group[close_col].astype(float)
    high = group[high_col].astype(float)
    low = group[low_col].astype(float)
    volume = (
        group[volume_col].astype(float)
        if volume_col is not None
        else pd.Series(np.nan, index=group.index)
    )

    # The one-row shift is the point-in-time boundary: row t can only see t - 1 and earlier.
    past_price = price.shift(1)
    past_close = close.shift(1)
    past_high = high.shift(1)
    past_low = low.shift(1)
    past_volume = volume.shift(1)

    one_day_return = price.pct_change()
    log_return = np.log(price / price.shift(1))
    past_log_return = log_return.shift(1)

    # Trend: rolling simple and exponentially weighted moving averages.
    for window in sma_windows:
        group[f"sma_{window}"] = past_price.rolling(window=window, min_periods=window).mean()

    if 50 not in sma_windows:
        group["sma_50"] = past_price.rolling(window=50, min_periods=50).mean()

    # The 12/26 EMAs are the standard MACD inputs.
    group["ema_12"] = past_price.ewm(span=12, adjust=False, min_periods=12).mean()
    group["ema_26"] = past_price.ewm(span=26, adjust=False, min_periods=26).mean()
    group["price_to_sma_50"] = _safe_divide(past_price, group["sma_50"]) - 1

    # MACD: the spread between short- and long-term EMAs, plus its signal line.
    group["macd"] = group["ema_12"] - group["ema_26"]
    group["macd_signal"] = group["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    group["macd_hist"] = group["macd"] - group["macd_signal"]

    # Momentum: RSI uses prior price changes; stochastic uses prior OHLC ranges.
    delta = price.diff().shift(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = _safe_divide(avg_gain, avg_loss)
    group["rsi_14"] = 100 - (100 / (1 + rs))

    lowest_low = past_low.rolling(window=14, min_periods=14).min()
    highest_high = past_high.rolling(window=14, min_periods=14).max()
    group["stoch_k_14"] = 100 * _safe_divide(past_close - lowest_low, highest_high - lowest_low)
    group["stoch_d_3"] = group["stoch_k_14"].rolling(window=3, min_periods=3).mean()

    group["roc_5"] = past_price.pct_change(periods=5) * 100
    group["roc_10"] = past_price.pct_change(periods=10) * 100

    if include_williams_r:
        group["williams_r_14"] = -100 * _safe_divide(
            highest_high - past_close,
            highest_high - lowest_low,
        )

    # Volatility: Bollinger Bands describe the past 20-day price distribution.
    bb_middle = past_price.rolling(window=20, min_periods=20).mean()
    bb_std = past_price.rolling(window=20, min_periods=20).std()
    group["bb_middle_20"] = bb_middle
    group["bb_upper_20"] = bb_middle + (2 * bb_std)
    group["bb_lower_20"] = bb_middle - (2 * bb_std)
    group["bb_width_20"] = _safe_divide(group["bb_upper_20"] - group["bb_lower_20"], bb_middle)
    group["bb_percent_b_20"] = _safe_divide(
        past_price - group["bb_lower_20"],
        group["bb_upper_20"] - group["bb_lower_20"],
    )

    # ATR measures the past true range, including overnight gaps.
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    group["atr_14"] = true_range.shift(1).rolling(window=14, min_periods=14).mean()

    # Log-return dispersion is also the sigma input for GBM/jump-diffusion models.
    for window in (10, 20):
        std = past_log_return.rolling(window=window, min_periods=window).std()
        group[f"log_return_std_{window}"] = std
        group[f"realized_vol_{window}_annualized"] = std * np.sqrt(252)

    # Volume: OBV carries cumulative signed volume forward from completed sessions.
    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()
    group["obv"] = obv.shift(1)
    group["obv_change_5"] = group["obv"].diff(5)

    # VWAP uses typical price times volume; both inputs are shifted before aggregation.
    typical_price = (high + low + close) / 3
    past_typical_price = typical_price.shift(1)
    group["volume_sma_20"] = past_volume.rolling(window=20, min_periods=20).mean()
    group["volume_to_sma_20"] = _safe_divide(past_volume, group["volume_sma_20"])

    rolling_dollar_volume = (past_typical_price * past_volume).rolling(
        window=20,
        min_periods=20,
    ).sum()
    rolling_volume = past_volume.rolling(window=20, min_periods=20).sum()
    group["vwap_20"] = _safe_divide(rolling_dollar_volume, rolling_volume)

    # Measured against the rolling 20-day VWAP, not a cumulative-from-inception one.
    # A cumulative VWAP averages the symbol's entire history, so it drifts far from
    # recent price and makes this ratio a comparison against years-old trading.
    group["price_to_vwap"] = _safe_divide(past_price, group["vwap_20"]) - 1

    # Cumulative VWAP retained only for existing consumers; not a model feature.
    cumulative_dollar_volume = (past_typical_price * past_volume).cumsum()
    cumulative_volume = past_volume.cumsum()
    group["vwap_cumulative"] = _safe_divide(cumulative_dollar_volume, cumulative_volume)

    # Lagged returns expose prior N-day price moves to the forecasting model.
    for days in lag_days:
        group[f"log_return_lag_{days}d"] = past_log_return.shift(days - 1)
        # Retained for existing consumers; use log_return_lag_* for new models.
        group[f"return_lag_{days}d"] = past_price.pct_change(periods=days)

    group["volatility_lag_1d"] = group["log_return_std_20"].shift(1)
    group["volatility_lag_5d"] = group["log_return_std_20"].shift(5)
    group["return_20d"] = past_price.pct_change(periods=20)

    return group


def _add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic date features that are known before each trading session."""
    frame = frame.copy()
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["month"] = frame["date"].dt.month
    frame["quarter"] = frame["date"].dt.quarter
    return frame


def _add_pooled_features(
    frame: pd.DataFrame,
    symbol_col: str | None,
    benchmark: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add identifiers and peer-relative features for the pooled ticker model."""
    if symbol_col is None:
        return frame

    frame = frame.copy()
    frame["ticker_id"] = pd.factorize(frame[symbol_col], sort=True)[0]
    sector_col = _optional_column(frame, "GICS Sector")

    if sector_col is not None:
        # Encode the source sector column directly. A separate "sector" string copy
        # would duplicate "GICS Sector" verbatim; the readable label already exists
        # there, and models consume the numeric encoding below.
        frame["sector_id"] = pd.factorize(frame[sector_col].astype("string"), sort=True)[0]
        sector_return = frame.groupby(["date", sector_col])["return_20d"].transform("mean")
        frame["sector_avg_return_20d"] = sector_return
        frame["relative_strength_sector_20d"] = frame["return_20d"] - sector_return

    if benchmark is not None:
        frame = _add_benchmark_features(frame, symbol_col, benchmark)

    return frame


def _add_benchmark_features(
    frame: pd.DataFrame,
    symbol_col: str,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Merge an external index series and derive market-relative features."""
    benchmark = benchmark.copy()
    benchmark_date_col = _column(benchmark, "Date")
    benchmark_price_col = _optional_column(benchmark, "Adj Close") or _column(benchmark, "Close")
    benchmark = benchmark.rename(columns={benchmark_date_col: "date"})
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark = benchmark.sort_values("date").drop_duplicates("date", keep="last")

    benchmark_price = benchmark[benchmark_price_col].astype(float)
    benchmark_log_return = np.log(benchmark_price / benchmark_price.shift(1)).shift(1)
    benchmark_features = pd.DataFrame(
        {
            "date": benchmark["date"],
            "market_return_20d": benchmark_price.shift(1).pct_change(20),
            "market_log_return": benchmark_log_return,
            "market_realized_vol_20_annualized": (
                benchmark_log_return.rolling(20, min_periods=20).std() * np.sqrt(252)
            ),
        }
    )
    frame = frame.merge(benchmark_features, on="date", how="left", validate="many_to_one")
    frame["relative_strength_market_20d"] = (
        frame["return_20d"] - frame["market_return_20d"]
    )

    beta = pd.concat(
        [_rolling_beta(group) for _, group in frame.groupby(symbol_col, sort=False)]
    )
    frame["beta_60"] = beta.reindex(frame.index)
    return frame


def _rolling_beta(group: pd.DataFrame) -> pd.Series:
    """Calculate 60-day beta using returns already shifted to the prior session."""
    covariance = group["log_return_lag_1d"].rolling(60, min_periods=60).cov(
        group["market_log_return"]
    )
    market_variance = group["market_log_return"].rolling(60, min_periods=60).var()
    return _safe_divide(covariance, market_variance)


def _add_days_to_next_earnings(
    frame: pd.DataFrame,
    earnings_calendar: pd.DataFrame,
    symbol_col: str | None,
) -> pd.DataFrame:
    """Add the next scheduled earnings date, when the calendar has one."""
    if symbol_col is None or earnings_calendar.empty:
        return frame

    earnings_symbol_col = _column(earnings_calendar, "Symbol")
    earnings_date_col = _column(earnings_calendar, "earnings_date")

    # PIT provenance guard. days_to_next_earnings is only leakage-free if the
    # calendar holds ANNOUNCED/scheduled dates (knowable at t), not backfilled
    # ACTUAL reported dates. We cannot verify provenance from here, but a purely
    # historical dump (no dates beyond the last price date) is a strong signal
    # the calendar was backfilled --- a live scheduled calendar always carries at
    # least the next upcoming announcement. Warn loudly rather than silently
    # trusting what is often the model's top SHAP feature.
    max_price_date = pd.to_datetime(frame["date"]).max()
    max_earnings_date = pd.to_datetime(earnings_calendar[earnings_date_col]).max()
    if pd.notna(max_earnings_date) and max_earnings_date <= max_price_date:
        import warnings
        warnings.warn(
            "earnings_calendar has no dates beyond the price history "
            f"(latest earnings {max_earnings_date.date()} <= latest price "
            f"{max_price_date.date()}). This is consistent with a BACKFILLED "
            "actual-date calendar, which makes days_to_next_earnings leak future "
            "information. Confirm the calendar stores announced/scheduled dates "
            "before trusting this feature.",
            stacklevel=2,
        )

    earnings_by_symbol = {
        symbol: pd.DatetimeIndex(pd.to_datetime(group[earnings_date_col]).dropna().sort_values())
        for symbol, group in earnings_calendar.groupby(earnings_symbol_col, sort=False)
    }

    values = []
    for symbol, date in zip(frame[symbol_col], frame["date"], strict=True):
        dates = earnings_by_symbol.get(symbol)
        if dates is None:
            values.append(np.nan)
            continue
        position = dates.searchsorted(date, side="left")
        values.append((dates[position] - date).days if position < len(dates) else np.nan)

    frame = frame.copy()
    frame["days_to_next_earnings"] = values
    return frame


def _add_targets(
    frame: pd.DataFrame,
    price_col: str,
    symbol_col: str | None,
    horizon: int,
) -> pd.DataFrame:
    """Add forward-looking labels; these columns must be excluded from predictors.

    Labels are built from the ADJUSTED price series, not raw close. A split or
    dividend ex-date falling inside the [t, t+horizon] window would otherwise
    corrupt the label: a 2:1 split halves raw close, flipping future_direction
    to "down" for a name that actually rose. For a directional model this is
    worse than noise --- it inverts the class. Adjusted price is continuous
    through corporate actions, so the sign is trustworthy.
    """
    def add_group_targets(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        close = group[price_col].astype(float)
        group["future_return"] = np.log(close.shift(-horizon) / close)
        group["future_direction"] = pd.Series(
            np.where(group["future_return"].notna(), group["future_return"] > 0, pd.NA),
            index=group.index,
            dtype="Int64",
        )

        future_log_returns = np.log(close / close.shift(1)).shift(-1)
        group["future_realized_vol"] = (
            future_log_returns.iloc[::-1]
            .rolling(horizon, min_periods=horizon)
            .std()
            .iloc[::-1]
        )
        return group

    if symbol_col is None:
        return add_group_targets(frame)
    return pd.concat(
        [add_group_targets(group) for _, group in frame.groupby(symbol_col, sort=False)],
        ignore_index=True,
    )


def flag_short_history(
    df: pd.DataFrame,
    min_rows: int = DEFAULT_MIN_HISTORY_ROWS,
) -> pd.DataFrame:
    """Attach per-symbol history metadata without dropping any rows.

    ``short_history`` marks symbols with fewer than ``min_rows`` sessions --- too
    few to fill the long rolling windows (sma_200, 60-day beta) or to power the
    return-space ADF test, which flags such names (e.g. a 2025 spin-off with
    ~360 rows) as non-stationary purely for lack of sample. This is universe
    metadata, not a feature of the price series, so it is computed once at the
    frame level rather than inside the per-symbol feature path. Consumers apply
    their own policy: training excludes short_history; the risk engine may keep
    and down-weight it.
    """
    frame = df.copy()

    symbol_col = _optional_column(frame, "Symbol")
    if symbol_col is None:
        frame["history_rows"] = len(frame)
        frame["short_history"] = len(frame) < min_rows
        return frame

    counts = frame.groupby(symbol_col)[_column(frame, "Date")].transform("count")
    frame["history_rows"] = counts
    frame["short_history"] = counts < min_rows
    return frame

def add_technical_indicators(
    df: pd.DataFrame,
    lag_days: tuple[int, ...] = DEFAULT_LAG_DAYS,
    sma_windows: tuple[int, ...] = DEFAULT_SMA_WINDOWS,
    rolling_windows: tuple[int, ...] | None = None,
    price_column: str | None = None,
    earnings_calendar: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
    target_horizon: int = DEFAULT_TARGET_HORIZON,
    include_williams_r: bool = False,
) -> pd.DataFrame:
    """
    Add point-in-time technical indicators to a price frame.

    Predictor values at row t use only earlier rows from the same symbol.
    Labels deliberately use future prices and must never be used as model inputs.
    """
    if df.empty:
        return df.copy()

    # Resolve common vendor column variants before applying the shared calculations.
    frame = df.copy()
    date_col = _column(frame, "Date")
    close_col = _column(frame, "Close")
    high_col = _optional_column(frame, "High") or close_col
    low_col = _optional_column(frame, "Low") or close_col
    volume_col = _optional_column(frame, "Volume")
    symbol_col = _optional_column(frame, "Symbol")
    price_col = price_column or (_optional_column(frame, "Adj Close") or close_col)

    if target_horizon < 1:
        raise ValueError("target_horizon must be at least 1")

    # Preserve the former keyword while naming the parameter after its actual use.
    if rolling_windows is not None:
        sma_windows = rolling_windows

    if date_col != "date":
        frame = frame.rename(columns={date_col: "date"})

    frame["date"] = pd.to_datetime(frame["date"])
    sort_columns = ["date"] if symbol_col is None else [symbol_col, "date"]
    frame = frame.sort_values(sort_columns).reset_index(drop=True)

    # A multi-symbol frame must be grouped before calculating any rolling feature.
    if symbol_col is None:
        featured = _feature_group(
            frame,
            price_col=price_col,
            close_col=close_col,
            high_col=high_col,
            low_col=low_col,
            volume_col=volume_col,
            lag_days=tuple(lag_days),
            sma_windows=tuple(sma_windows),
            include_williams_r=include_williams_r,
        )
    else:
        featured = pd.concat(
            [
                _feature_group(
                    group,
                    price_col=price_col,
                    close_col=close_col,
                    high_col=high_col,
                    low_col=low_col,
                    volume_col=volume_col,
                    lag_days=tuple(lag_days),
                    sma_windows=tuple(sma_windows),
                    include_williams_r=include_williams_r,
                )
                for _, group in frame.groupby(symbol_col, sort=False)
            ],
            ignore_index=True,
        )

    featured = _add_calendar_features(featured)
    featured = _add_pooled_features(featured, symbol_col, benchmark)

    if earnings_calendar is not None:
        featured = _add_days_to_next_earnings(featured, earnings_calendar, symbol_col)

    return _add_targets(featured, price_col, symbol_col, target_horizon)


def add_moving_average(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compatibility wrapper for older training placeholders."""
    frame = df.copy()
    close_col = _column(frame, "Close")
    symbol_col = _optional_column(frame, "Symbol")

    if symbol_col is None:
        frame[f"ma_{window}"] = frame[close_col].shift(1).rolling(window=window).mean()
    else:
        frame[f"ma_{window}"] = (
            frame.sort_values([symbol_col, _column(frame, "Date")])
            .groupby(symbol_col)[close_col]
            .transform(lambda values: values.shift(1).rolling(window=window).mean())
        )
    return frame


def build_feature_dataset(
    input_path: str | Path = RAW_PRICES_PATH,
    earnings_path: str | Path = EARNINGS_CALENDAR_PATH,
    output_path: str | Path = "data/processed/sp500_features.parquet",
    drop_warmup_rows: bool = False,
    benchmark: pd.DataFrame | None = None,
    target_horizon: int = DEFAULT_TARGET_HORIZON,
    macro_path: str | Path | None = "data/external/macro_daily.parquet",
    short_interest_path: str | Path | None = "data/external/short_interest.parquet",
    fundamentals_path: str | Path | None = "data/external/fundamentals.parquet",
) -> pd.DataFrame:
    """Load local price and earnings data, create features and labels, then save them.

    If ``macro_path`` points at an existing file, market-wide macro factors
    (from ``macro.py`` / FRED) are merged on ``date``. These are the same for
    every ticker on a given day, so they condition the volatility regime rather
    than distinguishing tickers. The merge is skipped silently when the file is
    absent, so the pipeline still runs without macro data.

    If ``short_interest_path`` points at an existing file, FINRA consolidated
    short interest is merged point-in-time on the dissemination date (see
    ``short_interest.merge_short_interest_pit``). Coverage begins ~2018
    (``schema.SHORT_INTEREST_START``), so rows before that carry NaN short-interest
    features -- intended, and the reason models using them should scope to 2018+.
    Also skipped silently when the file is absent.
    """
    input_path = Path(input_path)
    earnings_path = Path(earnings_path)
    output_path = Path(output_path)

    # Keep data access in the loader helper; this module only creates features.
    frame = load_raw_prices(path=input_path)
    frame = flag_short_history(frame)

    n_short = frame.loc[frame["short_history"], "symbol"].nunique()
    if n_short:
        symbols = sorted(frame.loc[frame["short_history"], "symbol"].unique())
        print(f"Flagged {n_short} short-history symbols (< {DEFAULT_MIN_HISTORY_ROWS} rows): {symbols}")
        
    earnings_calendar = load_earnings_calendar(path=earnings_path)
    features = add_technical_indicators(
        frame,
        earnings_calendar=earnings_calendar,
        benchmark=benchmark,
        target_horizon=target_horizon,
    )

    # Optional market-wide macro factors, joined on date only.
    if macro_path is not None and Path(macro_path).exists():
        from forecast_engine.data.macro import merge_macro_into_features

        macro = pd.read_parquet(macro_path)
        features = merge_macro_into_features(features, macro)

    # Optional non-price feature: FINRA consolidated short interest, PIT-merged on
    # the dissemination date. Skipped silently when the file is absent, like macro.
    # Coverage begins ~2018, so earlier rows carry NaN short-interest features
    # (schema.SHORT_INTEREST_START documents the floor) -- intended, not a bug.
    if short_interest_path is not None and Path(short_interest_path).exists():
        from forecast_engine.data.short_interest import merge_short_interest_pit

        short = pd.read_parquet(short_interest_path)
        features = merge_short_interest_pit(features, short)

    # Optional non-price feature: fundamentals (EDGAR / Sharadar), PIT-merged on the
    # filing date. Value/quality factors -- the academically-supported non-price
    # return signal, tested at the 120-day horizon. Skipped silently when absent.
    # EDGAR XBRL coverage is generally good back to ~2010, so there is usually no
    # large pre-coverage gap (unlike short interest). The merge also computes the
    # price-based valuation multiples (earnings_yield, book_to_price) at merge time.
    if fundamentals_path is not None and Path(fundamentals_path).exists():
        from forecast_engine.data.fundamentals import merge_fundamentals_pit

        fundamentals = pd.read_parquet(fundamentals_path)
        features = merge_fundamentals_pit(features, fundamentals)

    if drop_warmup_rows:
        # Early rows lack enough history for long rolling windows and are optional to remove.
        feature_columns = [
            column
            for column in features.columns
            if column not in frame.columns
        ]
        features = features.dropna(subset=feature_columns, how="all").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        features.to_parquet(output_path, index=False)
    except PermissionError:
        # A timestamped file avoids failing when the standard output is open elsewhere.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
        features.to_parquet(fallback_path, index=False)

    return features


def main() -> None:
    features = build_feature_dataset()
    print(
        "Feature dataset complete: "
        f"{len(features):,} rows, {features['symbol'].nunique():,} symbols"
    )


if __name__ == "__main__":
    main()