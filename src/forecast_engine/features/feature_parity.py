"""Train/serve feature parity.

The most common way an ML system degrades silently in production is a mismatch
between how features are computed at training time (over the full history, in a
batch) and at request time (over a trailing window, per ticker). Rolling-window
edge effects, differing warm-up handling, or a subtly different code path make the
served features drift from what the model was trained on -- and nothing errors;
the predictions just quietly get worse.

The structural defence is already in place: training and serving both call
`add_technical_indicators` from `features/indicators.py`. This module turns that
shared-code convention into a *checked guarantee* -- a function that recomputes a
recent slice the way serving would and asserts it matches the training-path
values, and a CI test that fails if they ever diverge.

The rule this enforces: **a feature value for a given (symbol, date) must be
identical whether it was computed in a full-history batch or from a trailing
window at serve time.** If that ever stops being true, this fails loudly instead
of degrading silently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Longest look-back any feature needs (e.g. sma_200). Serving must feed at least
# this much trailing history for the most recent row's features to be correct.
MAX_LOOKBACK = 200
# Extra cushion so rolling windows are fully warmed before the compared rows.
SERVE_WARMUP = MAX_LOOKBACK + 100  # +100 so MACD EMAs (26+9) fully converge in the serving window


def serving_window(
    full_history: pd.DataFrame,
    as_of_index: int,
    lookback: int = SERVE_WARMUP,
    date_column: str = "date",
) -> pd.DataFrame:
    """Extract the trailing window a serving request would have for one ticker.

    Given a single ticker's full history and the position being served, returns
    the trailing `lookback` rows up to and including that position -- exactly what
    the serving path would compute features from.
    """
    frame = full_history.sort_values(date_column).reset_index(drop=True)
    start = max(0, as_of_index - lookback + 1)
    return frame.iloc[start:as_of_index + 1].copy()


def parity_mismatches(
    training_features: pd.DataFrame,
    serving_features: pd.DataFrame,
    feature_columns: list[str],
    key_columns: tuple[str, str] = ("symbol", "date"),
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> pd.DataFrame:
    """Return the rows/features where the training and serving paths disagree.

    Both frames must contain the key columns and the feature columns. The
    comparison is done on the intersection of keys (the overlap between the full
    batch and the served window). An empty result means perfect parity.
    """
    keys = list(key_columns)
    merged = training_features.merge(
        serving_features, on=keys, suffixes=("_train", "_serve"), how="inner"
    )
    if merged.empty:
        raise ValueError("No overlapping (symbol, date) rows to compare.")

    mismatches = []
    for feature in feature_columns:
        train_col = f"{feature}_train"
        serve_col = f"{feature}_serve"
        if train_col not in merged or serve_col not in merged:
            continue
        a = merged[train_col].to_numpy(dtype=float)
        b = merged[serve_col].to_numpy(dtype=float)
        # NaN in the same place is agreement; a value vs NaN is a mismatch.
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=rtol, atol=atol) | both_nan
        bad = ~close
        if bad.any():
            block = merged.loc[bad, keys].copy()
            block["feature"] = feature
            block["train_value"] = a[bad]
            block["serve_value"] = b[bad]
            mismatches.append(block)

    if not mismatches:
        return merged.iloc[0:0][keys].assign(feature=[], train_value=[], serve_value=[])
    return pd.concat(mismatches, ignore_index=True)


def assert_feature_parity(
    training_features: pd.DataFrame,
    serving_features: pd.DataFrame,
    feature_columns: list[str],
    **kwargs,
) -> None:
    """Raise AssertionError if any feature disagrees between the two paths."""
    mismatches = parity_mismatches(
        training_features, serving_features, feature_columns, **kwargs
    )
    if len(mismatches):
        sample = mismatches.head(10).to_string(index=False)
        raise AssertionError(
            f"Train/serve feature parity broken for {len(mismatches)} "
            f"(row, feature) pairs. First few:\n{sample}"
        )