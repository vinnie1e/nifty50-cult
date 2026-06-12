"""
Portfolio Construction Module.

Builds recommended allocations for three investor profiles by trading off
expected return against risk:

  * Conservative -> minimum-variance, capped per-name, tilts to low-vol names.
  * Balanced     -> maximum Sharpe (tangency) portfolio.
  * Aggressive   -> return-tilted, higher concentration / risk budget allowed.

Optimisation uses historical mean returns and the sample covariance matrix
(estimated only from the provided dataset). Long-only weights, summing to 1,
solved with SLSQP. Each recommendation comes with a justification string.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def _annualised_stats(close_wide: pd.DataFrame):
    rets = close_wide.pct_change().dropna()
    mu = rets.mean().to_numpy() * TRADING_DAYS
    cov = rets.cov().to_numpy() * TRADING_DAYS
    return mu, cov, list(close_wide.columns)


def _port_perf(w, mu, cov):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol


def _optimise(mu, cov, objective, max_weight=0.35, rf=0.06):
    n = len(mu)
    w0 = np.repeat(1 / n, n)
    bounds = [(0.0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_sharpe(w):
        ret, vol = _port_perf(w, mu, cov)
        return -(ret - rf) / vol if vol > 0 else 1e6

    def variance(w):
        return float(w @ cov @ w)

    def neg_return_per_risk(w):
        ret, vol = _port_perf(w, mu, cov)
        return -(ret) / (vol ** 0.5) if vol > 0 else 1e6

    obj = {"sharpe": neg_sharpe, "minvar": variance,
           "return_tilt": neg_return_per_risk}[objective]
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-9})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def construct_portfolios(close_wide: pd.DataFrame, rf: float = 0.06) -> dict:
    """Return allocations + stats + justification for the three profiles."""
    close_wide = close_wide.dropna(axis=1, how="any")
    mu, cov, symbols = _annualised_stats(close_wide)

    profiles = {
        "Conservative": dict(objective="minvar", max_weight=0.20,
                             rationale="Minimum-variance allocation; caps single-name "
                                       "exposure at 20% to prioritise capital preservation "
                                       "and low drawdown over upside."),
        "Balanced": dict(objective="sharpe", max_weight=0.30,
                         rationale="Maximum-Sharpe (tangency) portfolio; best historical "
                                   "risk-adjusted return, diversified across names."),
        "Aggressive": dict(objective="return_tilt", max_weight=0.45,
                           rationale="Return-tilted allocation accepting higher volatility "
                                     "and concentration (up to 45% per name) to chase "
                                     "higher expected return."),
    }

    out = {}
    for name, cfg in profiles.items():
        w = _optimise(mu, cov, cfg["objective"], cfg["max_weight"], rf)
        ret, vol = _port_perf(w, mu, cov)
        sharpe = (ret - rf) / vol if vol > 0 else float("nan")
        alloc = {symbols[i]: round(float(w[i]), 4)
                 for i in range(len(symbols)) if w[i] > 1e-3}
        alloc = dict(sorted(alloc.items(), key=lambda kv: -kv[1]))
        out[name] = {
            "weights": alloc,
            "expected_annual_return": round(ret, 4),
            "expected_annual_volatility": round(vol, 4),
            "sharpe": round(float(sharpe), 4),
            "justification": cfg["rationale"],
        }
    return out


def sector_allocation(portfolio: dict, market) -> dict:
    """Aggregate a weight dict into sector exposures for explainability."""
    sectors = {}
    for sym, w in portfolio.get("weights", {}).items():
        sec = market.sector_of(sym)
        sectors[sec] = round(sectors.get(sec, 0.0) + w, 4)
    return dict(sorted(sectors.items(), key=lambda kv: -kv[1]))
