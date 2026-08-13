import pandas as pd


def label_regimes(df: pd.DataFrame, threshold: float = 0.01) -> pd.Series:
    returns = df["close"].pct_change()
    return pd.cut(returns.fillna(0), bins=[-1.0, -threshold, threshold, 1.0], labels=["down", "flat", "up"])
