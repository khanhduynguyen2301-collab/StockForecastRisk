"""Feature and label names emitted by the processed market dataset."""

FEATURE_NAMES = [
    # Trend
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_12",
    "ema_26",
    "macd",
    "macd_signal",
    "macd_hist",
    "price_to_sma_50",
    # Momentum
    "rsi_14",
    "stoch_k_14",
    "stoch_d_3",
    "roc_5",
    "roc_10",
    # Volatility
    "atr_14",
    "bb_width_20",
    "bb_percent_b_20",
    "log_return_std_10",
    "log_return_std_20",
    "realized_vol_10_annualized",
    "realized_vol_20_annualized",
    # Volume
    "obv_change_5",
    "volume_to_sma_20",
    "price_to_vwap",
    "vwap_20",
    # Autoregressive
    "log_return_lag_1d",
    "log_return_lag_2d",
    "log_return_lag_3d",
    "log_return_lag_4d",
    "log_return_lag_5d",
    "volatility_lag_1d",
    "volatility_lag_5d",
    "return_20d",
    # Pooled model and calendar
    "ticker_id",
    "sector_id",
    "relative_strength_sector_20d",
    "day_of_week",
    "month",
    "quarter",
    "days_to_next_earnings",
    # Non-price: FINRA consolidated short interest (biweekly, PIT-merged on the
    # dissemination date). Coverage begins ~2018 (see SHORT_INTEREST_START); rows
    # before that are NaN, so models using these should scope to 2018+.
    "short_interest",
    "short_interest_change",
    "days_to_cover",
    "days_to_cover_change",
    # Non-price: fundamentals (SEC EDGAR / Sharadar), PIT-merged on the filing date.
    # Value/quality factors -- the academically-supported non-price return signal.
    # Quarterly cadence; see FUNDAMENTAL_FEATURE_NAMES and FUNDAMENTALS_START.
    "revenue_growth_yoy",
    "earnings_growth_yoy",
    "eps_growth_yoy",
    "grossmargin",
    "netmargin",
    "roe",
    "roa",
    "debt_to_equity",
    "earnings_yield",
    "book_to_price",
]

# Added only when add_technical_indicators receives an external index series.
BENCHMARK_FEATURE_NAMES = [
    "market_return_20d",
    "market_realized_vol_20_annualized",
    "relative_strength_market_20d",
    "beta_60",
]

# Market-wide macro factors from macro.py (FRED), merged on date. Identical
# across tickers on a given day: they condition the volatility regime but cannot
# aid cross-sectional ranking. Present only when data/external/macro_daily.parquet
# exists and was merged in build_feature_dataset; optional, like the benchmark set.
MACRO_FEATURE_NAMES = [
    "vix",
    "vix_3m",
    "hy_credit_spread",
    "ig_credit_spread",
    "treasury_10y",
    "treasury_2y",
    "fed_funds_rate",
    "vix_term_ratio",
    "yield_curve_slope",
    "vix_change_5d",
    "vix_change_20d",
    "hy_credit_spread_change_5d",
    "hy_credit_spread_change_20d",
    "yield_curve_slope_change_5d",
    "yield_curve_slope_change_20d",
]

# The ICE BofA credit-spread series (hy_/ig_credit_spread) begin only in mid-2023
# at the FRED daily resolution used here -- roughly two years into the price
# history. Including them forces a choice between dropping ~40% of training rows
# and leaving NaNs that a tree model reads as a "before mid-2023" date proxy (a
# leak, not signal). Notebook 01's macro coverage check surfaces this. Since
# credit spreads track VIX closely anyway, the resolution is to exclude them and
# use the full-history subset below for modelling. Use MACRO_FEATURE_NAMES only
# if the credit series' coverage improves on a future data refresh.
CREDIT_SPREAD_FEATURE_NAMES = [
    "hy_credit_spread",
    "ig_credit_spread",
    "hy_credit_spread_change_5d",
    "hy_credit_spread_change_20d",
]

MACRO_FEATURE_NAMES_FULL_HISTORY = [
    name for name in MACRO_FEATURE_NAMES if name not in CREDIT_SPREAD_FEATURE_NAMES
]

# FINRA consolidated short interest via the API only reaches back to ~2018 at the
# daily merge resolution, roughly 8 years into the 2010 price history. Rows before
# the coverage start are NaN. Unlike the credit spreads (which are simply excluded),
# short interest is kept as a predictor, but any experiment or model that relies on
# it should be scoped to SHORT_INTEREST_START onward -- otherwise a tree can read
# the pre-2018 NaN block as a "before-2018" date proxy (a leak, not signal). The
# audit notebook's short-interest section surfaces this.
SHORT_INTEREST_FEATURE_NAMES = [
    "short_interest",
    "short_interest_change",
    "days_to_cover",
    "days_to_cover_change",
]

# Coverage floor: scope short-interest experiments to this date onward.
SHORT_INTEREST_START = "2018-01-01"

# Fundamentals (SEC EDGAR / Sharadar), PIT-merged on the filing date. The
# value/quality features -- the academically-supported non-price return signal,
# tested at the 120-day horizon where a real return edge was established. Two
# columns (earnings_yield, book_to_price) are computed at merge time from the
# fundamental numerator and the price denominator, so they are added by
# merge_fundamentals_pit rather than existing in the raw fundamentals parquet.
# EDGAR/XBRL coverage is generally good back to ~2010 (mandatory XBRL tagging began
# ~2009), so unlike short interest there is usually no large pre-coverage gap -- but
# scope to FUNDAMENTALS_START if the audit shows a later floor for your universe.
FUNDAMENTAL_FEATURE_NAMES = [
    "revenue_growth_yoy",
    "earnings_growth_yoy",
    "eps_growth_yoy",
    "grossmargin",
    "netmargin",
    "roe",
    "roa",
    "debt_to_equity",
    "earnings_yield",
    "book_to_price",
]

# Coverage floor for fundamentals experiments. Confirm against the audit; ~2010 is
# typical for EDGAR XBRL data.
FUNDAMENTALS_START = "2010-01-01"

# Labels intentionally contain future information and must not enter model inputs.
TARGET_NAMES = ["future_return", "future_direction", "future_realized_vol"]

# Columns the pipeline emits that are deliberately NOT model inputs. Listed so the
# exclusion is explicit and testable rather than accidental: a new column that
# appears in neither FEATURE_NAMES nor here is a schema drift bug.
NON_FEATURE_COLUMNS = [
    # Intermediates: inputs to real features, not predictors themselves.
    "bb_middle_20",
    "bb_upper_20",
    "bb_lower_20",
    "volume_sma_20",
    "sector_avg_return_20d",
    "market_log_return",
    # Universe metadata: per-symbol history gate, not a property of the price
    # series. history_rows is a static per-ticker integer -- feeding it to a
    # model leaks ticker identity. Consumers filter on short_history instead.
    "history_rows",
    "short_history",
    # Short-interest bookkeeping carried through the PIT merge but not a predictor.
    "settlement_date",
    "dissemination_date",
    "avg_daily_volume",
    # Fundamentals bookkeeping carried through the PIT merge but not a predictor.
    # filing_date is the PIT key; calendardate is the fiscal period end (must never
    # be a merge key or a feature); the raw levels are inputs to the engineered
    # ratios/growth, not predictors themselves.
    "filing_date",
    "calendardate",
    "reportperiod",
    "revenue",
    "netinc",
    "eps",
    "epsdil",
    "equity",
    "assets",
    "liabilities",
    "cost_of_revenue",
    "ebitda",
    "fcf",
    "sharesbas",
    "dps",
    "dimension",
    "end",
    "fp",
    "form",
    "fy",
    # Legacy percentage lags; the log_return_lag_* versions are canonical.
    "return_lag_1d",
    "return_lag_2d",
    "return_lag_3d",
    "return_lag_4d",
    "return_lag_5d",
    # Opt-in only, via include_williams_r.
    "williams_r_14",
]

# Raw columns carried through from the source data. Present in the processed frame
# for identification, joins, and plotting -- but never model inputs.
#
# The price and volume columns here are UNSHIFTED current-session values. Feeding
# them to a model would leak the present into a prediction made about the future:
# every genuine predictor is derived from their shifted counterparts instead.
RAW_COLUMNS = [
    "date",
    "symbol",
    "security",
    "gics_sector",
    "gics_sub_industry",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]

# 2.1: price_to_vwap now measured against the rolling vwap_20 rather than a
# cumulative-from-inception VWAP; `sector` replaced by the numeric `sector_id`;
# `return_20d` promoted to a listed feature. Models trained on 2.0 features are
# not compatible with 2.1 features.
# 2.2: optional MACRO_FEATURE_NAMES (FRED market-wide factors) may be merged in.
# These are absent unless data/external/macro_daily.parquet exists, so a 2.2
# dataset without macro data is column-identical to 2.1.
# 2.3: dropped obv, vwap_cumulative (cumulative-from-inception, broke train/serve
# parity); keep obv_change_5.
# 2.4: added history_rows / short_history universe-metadata columns (non-predictor).
# Additive and column-safe: a 2.3 model still trains, since both are in
# NON_FEATURE_COLUMNS and never enter the feature matrix. short_history is a
# computed column (recomputed from live row counts at serve time), not a static
# per-ticker attribute -- a borderline name flips to False once it accrues history.
# 2.5: added short_interest / short_interest_change / days_to_cover /
# days_to_cover_change -- the first NON-PRICE features (FINRA consolidated short
# interest, PIT-merged on dissemination date). Predictors, so in FEATURE_NAMES.
# Coverage begins ~2018 (SHORT_INTEREST_START); pre-2018 rows are NaN, so models
# using these should be scoped to 2018+ (see SHORT_INTEREST_FEATURE_NAMES note).
# The merge also carries settlement_date / dissemination_date / avg_daily_volume,
# which are bookkeeping and listed in NON_FEATURE_COLUMNS. Additive: a 2.4 model
# that does not request these columns is unaffected.
# 2.6: added fundamentals (SEC EDGAR / Sharadar), PIT-merged on the filing date:
# revenue_growth_yoy, earnings_growth_yoy, eps_growth_yoy, grossmargin, netmargin,
# roe, roa, debt_to_equity (engineered), plus earnings_yield and book_to_price
# (computed at merge time from the fundamental numerator and price denominator).
# These are the value/quality signals tested at the 120-day horizon. The merge
# carries fundamentals bookkeeping (filing_date, calendardate, raw levels, XBRL
# metadata), all in NON_FEATURE_COLUMNS. Additive: a 2.5 model that does not
# request these columns is unaffected.
SCHEMA_VERSION = "2.6"

# ---------------------------------------------------------------------------
# Canonical feature list for the VOLATILITY model (the v1 shipping model).
#
# This is defined ONCE here so training, serving, and the research notebooks all
# read the same list rather than each rebuilding an ad-hoc EXCLUDE set (a
# train/serve/research parity risk -- Section 5.3). It is the base *technical*
# feature set only, and deliberately excludes:
#
#   - Fundamentals & short interest: the A/B tests showed no lift, and both carry
#     large pre-coverage NaN blocks (pre-2010 / pre-2018). Because the model's
#     fit() drops any row with a NaN in ANY feature, including these would
#     silently truncate the training panel by years. Excluding them by
#     construction is what prevents that panel-truncation bug.
#   - Macro (FRED): the 08b A/B showed macro REDUCES out-of-sample R^2, so
#     gbm_boost excludes it by design.
#   - Non-stationary raw levels (sma_*, ema_*, vwap_20): price-scale levels that
#     don't belong in a pooled cross-ticker model; their stationary derivatives
#     (price_to_sma_50, macd, price_to_vwap, etc.) carry the signal instead.
#   - Universe metadata (short_history, history_rows): not price features.
#
# If you need the model to see a different set, change it HERE, not in a notebook.
NON_STATIONARY_LEVELS = [
    "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "vwap_20",
]

# Excluded from the volatility feature matrix (union of the groups above).
VOLATILITY_MODEL_EXCLUDE = (
    set(NON_STATIONARY_LEVELS)
    | set(SHORT_INTEREST_FEATURE_NAMES)
    | set(FUNDAMENTAL_FEATURE_NAMES)
    | set(MACRO_FEATURE_NAMES)
    | {"short_history", "history_rows"}
)

# The actual list the volatility model trains and serves on: base technical
# features, in FEATURE_NAMES order, minus the exclusions above.
VOLATILITY_FEATURE_NAMES = [
    name for name in FEATURE_NAMES if name not in VOLATILITY_MODEL_EXCLUDE
]