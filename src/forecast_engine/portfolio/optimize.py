import numpy as np


def cvar_portfolio(weights, returns, alpha=0.95):
    portfolio_returns = returns.dot(weights)
    loss = -portfolio_returns
    var = np.percentile(loss, alpha * 100)
    return loss[loss >= var].mean()
