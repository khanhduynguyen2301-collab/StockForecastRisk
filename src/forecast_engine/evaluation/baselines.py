"""Baseline predictors: the bar every model must clear to earn its place.

A recurring discipline in this project is that a model only counts if it beats a
simple baseline on the same data and metric. The return models failed this test
(naive beat them); the volatility model passed it (beat persistence in every
fold). These baselines are those bars, in reusable form.

A subtle but critical rule, learned the hard way in notebook 08b: **a baseline
must be on the same scale as the target.** An early draft compared an annualised
volatility feature (~0.3) against a raw 5-day-std target (~0.018) and made
persistence look absurdly bad. The correct volatility baseline is the target's
own backward-looking twin, computed the same way over a trailing window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive_return_baseline(train_target: pd.Series, n_test: int) -> np.ndarray:
    """Predict the train-set mean for every test row (the return baseline).

    For return forecasting this is close to predicting zero drift. Notebook 08
    showed Ridge and XGBoost could not beat this on MAE -- the null that justifies
    keeping the engine's drift simple.
    """
    return np.full(n_test, float(train_target.mean()))


def persistence_volatility_baseline(
    frame: pd.DataFrame,
    window: int = 5,
    symbol_column: str = "symbol",
    price_column: str = "adj_close",
) -> pd.Series:
    """Trailing realised volatility -- the backward-looking twin of the target.

    Computes the rolling standard deviation of daily log returns over `window`
    days, per symbol. This is the honest baseline for `future_realized_vol`
    because it is the same quantity, measured over the trailing window instead of
    the forward one. The volatility model must beat *this*, not a mis-scaled
    stand-in.
    """
    def trailing(group: pd.DataFrame) -> pd.Series:
        log_return = np.log(group[price_column] / group[price_column].shift(1))
        return log_return.rolling(window, min_periods=window).std()

    return frame.groupby(symbol_column, group_keys=False).apply(trailing)


def majority_class_baseline(train_labels: pd.Series, n_test: int) -> np.ndarray:
    """Predict the training majority class for every test row (direction baseline).

    For direction prediction on equities this is usually "up", since prices drift
    upward. A classifier that cannot beat this has found no directional signal --
    the bar notebook 08's 20-day direction test measured against.
    """
    majority = int(train_labels.mean() >= 0.5)
    return np.full(n_test, majority)


def beats_baseline(
    model_metric: float,
    baseline_metric: float,
    lower_is_better: bool = True,
) -> bool:
    """Did the model beat the baseline on this metric?

    `lower_is_better` is True for error metrics (MAE, RMSE) and False for skill
    metrics (R^2, accuracy, IC). Kept explicit so the direction of comparison is
    never ambiguous at the call site.
    """
    if lower_is_better:
        return model_metric < baseline_metric
    return model_metric > baseline_metric
