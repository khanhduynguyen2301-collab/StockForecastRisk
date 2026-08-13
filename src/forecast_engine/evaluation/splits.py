"""Walk-forward cross-validation with purging for time-series models.

This is the validation backbone of the whole project. Every modelling result --
the return null (08), the volatility signal (08b), the cross-sectional work
(08c-e) -- was produced through splits like these. Getting this right is what
separates an honest out-of-sample number from a leaked one.

Two properties are non-negotiable:

1. **Expanding-window walk-forward, never shuffled k-fold.** Standard k-fold
   shuffles time and lets future data train a model tested on the past. For time
   series that is look-ahead leakage and inflates every metric. Here each fold
   trains only on data *before* its test window.

2. **A purge gap between train and test.** The target is a forward-looking window
   (e.g. 5- or 21-day realised volatility). Rows near the train/test boundary
   have targets that overlap the test period, so training on them leaks the test
   outcome. The purge drops a gap at least as long as the target horizon. The
   most common silent bug is a purge shorter than the horizon -- so the purge is
   an explicit argument and should be set to the target horizon by the caller.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

DEFAULT_N_SPLITS = 5


def walk_forward_splits(
    frame: pd.DataFrame,
    purge_days: int,
    n_splits: int = DEFAULT_N_SPLITS,
    date_column: str = "date",
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield (train_index, test_index) pairs for an expanding-window backtest.

    The unique dates are split into `n_splits + 1` contiguous blocks. Fold k
    trains on everything up to `purge_days` before block k's end and tests on
    block k+1. `purge_days` MUST be at least the target's forward horizon, or the
    training targets overlap the test window and leak.

    Yields index objects into `frame`, so the caller keeps full control of which
    columns to use. Folds with an empty train or test side are skipped.
    """
    if purge_days < 0:
        raise ValueError("purge_days must be non-negative.")
    if date_column not in frame.columns:
        raise ValueError(f"frame has no {date_column!r} column.")

    dates = np.sort(frame[date_column].unique())
    if len(dates) < n_splits + 1:
        raise ValueError(
            f"Not enough distinct dates ({len(dates)}) for {n_splits} splits."
        )

    blocks = np.array_split(dates, n_splits + 1)
    for fold in range(n_splits):
        train_end = blocks[fold][-1]
        test_dates = blocks[fold + 1]
        purge_cutoff = train_end - pd.Timedelta(days=purge_days)

        train_idx = frame.index[frame[date_column] <= purge_cutoff]
        test_idx = frame.index[frame[date_column].isin(test_dates)]
        if len(train_idx) and len(test_idx):
            yield train_idx, test_idx


def split_summary(
    frame: pd.DataFrame,
    purge_days: int,
    n_splits: int = DEFAULT_N_SPLITS,
    date_column: str = "date",
) -> pd.DataFrame:
    """Describe each fold's train/test rows and date ranges.

    Useful as a sanity check before a backtest: confirms the folds expand, the
    purge gap is present, and no test window precedes its training data.
    """
    rows = []
    for fold, (train_idx, test_idx) in enumerate(
        walk_forward_splits(frame, purge_days, n_splits, date_column), start=1
    ):
        train_dates = frame.loc[train_idx, date_column]
        test_dates = frame.loc[test_idx, date_column]
        gap_days = (test_dates.min() - train_dates.max()).days
        rows.append({
            "fold": fold,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_end": train_dates.max().date(),
            "test_start": test_dates.min().date(),
            "gap_days": gap_days,
            "purge_respected": gap_days >= purge_days,
        })
    return pd.DataFrame(rows).set_index("fold")
