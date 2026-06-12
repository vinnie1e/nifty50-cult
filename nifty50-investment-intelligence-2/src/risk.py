"""
Risk Assessment Module.

Computes the standard historical risk/return measures for a single asset or a
portfolio return series:

  * Annualised volatility
  * Sharpe ratio        (excess return / total volatility)
  * Sortino ratio       (excess return / downside volatility)
  * Maximum drawdown    (worst peak-to-trough decline)
  * CAGR / annualised return
  * Calmar ratio        (CAGR / |max drawdown|), a risk-adjusted return measure

All metrics are derived solely from historical prices in the provided dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def annualised_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return float("nan")
    cum = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return float(cum ** (1 / years) - 1) if years > 0 else float("nan")


def annualised_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: float = 0.06) -> float:
    rf_daily = rf / TRADING_DAYS
    excess = returns - rf_daily
    vol = excess.std(ddof=0)
    return float(excess.mean() / vol * np.sqrt(TRADING_DAYS)) if vol > 0 else float("nan")


def sortino_ratio(returns: pd.Series, rf: float = 0.06) -> float:
    rf_daily = rf / TRADING_DAYS
    excess = returns - rf_daily
    downside = excess[excess < 0]
    dd = downside.std(ddof=0)
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS)) if dd > 0 else float("nan")


def max_drawdown(close: pd.Series) -> float:
    """Worst peak-to-trough decline as a (negative) fraction."""
    cum = (1 + close.pct_change().fillna(0)).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(close: pd.Series) -> float:
    r = daily_returns(close)
    mdd = abs(max_drawdown(close))
    car = annualised_return(r)
    return float(car / mdd) if mdd > 0 else float("nan")


def risk_profile(close: pd.Series, rf: float = 0.06) -> dict:
    """Full risk/return summary for one price series."""
    r = daily_returns(close)
    return {
        "annualised_return": round(annualised_return(r), 4),
        "annualised_volatility": round(annualised_volatility(r), 4),
        "sharpe": round(sharpe_ratio(r, rf), 4),
        "sortino": round(sortino_ratio(r, rf), 4),
        "max_drawdown": round(max_drawdown(close), 4),
        "calmar": round(calmar_ratio(close), 4),
        "n_days": int(len(close)),
    }


def portfolio_returns(close_wide: pd.DataFrame, weights: dict) -> pd.Series:
    """Daily returns of a weighted portfolio given a wide close-price matrix."""
    cols = [c for c in weights if c in close_wide.columns]
    w = np.array([weights[c] for c in cols])
    w = w / w.sum()
    rets = close_wide[cols].pct_change().dropna()
    return pd.Series(rets.to_numpy() @ w, index=rets.index)


def portfolio_risk_profile(close_wide: pd.DataFrame, weights: dict, rf: float = 0.06) -> dict:
    port_ret = portfolio_returns(close_wide, weights)
    synthetic_price = (1 + port_ret).cumprod()
    return {
        "annualised_return": round(annualised_return(port_ret), 4),
        "annualised_volatility": round(annualised_volatility(port_ret), 4),
        "sharpe": round(sharpe_ratio(port_ret, rf), 4),
        "sortino": round(sortino_ratio(port_ret, rf), 4),
        "max_drawdown": round(max_drawdown(synthetic_price), 4),
        "calmar": round(calmar_ratio(synthetic_price), 4),
    }
