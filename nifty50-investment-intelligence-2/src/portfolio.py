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

The optimiser internals (`PROFILES`, `optimise_weights`, `annualised_stats`,
`clean_universe`) are reused by `src/backtest.py` for an out-of-sample,
walk-forward evaluation of the same allocations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252

# Per-profile optimisation config. Reused by the backtest so the live
# recommendation and the backtested strategy are *identical* by construction.
PROFILES = {
    "Conservative": dict(
        objective="minvar", max_weight=0.20,
        rationale="Minimum-variance allocation; caps single-name exposure at 20% "
                  "to prioritise capital preservation and low drawdown over upside."),
    "Balanced": dict(
        objective="sharpe", max_weight=0.30,
        rationale="Maximum-Sharpe (tangency) portfolio; best historical "
                  "risk-adjusted return, diversified across names."),
    "Aggressive": dict(
        objective="return_tilt", max_weight=0.45,
        rationale="Return-tilted allocation accepting higher volatility and "
                  "concentration (up to 45% per name) to chase higher expected return."),
}


def clean_universe(
    close_wide: pd.DataFrame,
    min_history: int = TRADING_DAYS,
    coverage: float = 0.95,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Build a usable price panel from a ragged close-price matrix.

    The dataset carries old *renamed* tickers (TELCO→TATAMOTORS, INFOSYSTCH→INFY,
    …) that trade in non-overlapping eras, so a naive ``dropna(axis=1, how="any")``
    — or aligning to a single common window — collapses to nothing. Instead we
    select a coherent cross-section by *coverage*:

      1. optionally restrict to the most recent ``lookback_days`` rows,
      2. keep only symbols present for at least ``coverage`` of that window
         (this drops delisted old tickers and barely-listed names),
      3. forward-fill incidental gaps, drop residual NaN rows, and require at
         least ``min_history`` remaining observations.

    With the full history this yields the ~20 continuously-listed blue chips
    (a 20-year panel); with ``lookback_days`` set it yields a broader recent
    cross-section.
    """
    cw = close_wide.sort_index()
    if lookback_days is not None:
        cw = cw.iloc[-lookback_days:]

    col_cov = cw.notna().mean()
    keep = col_cov[col_cov >= coverage].index
    cw = cw.loc[:, keep]
    if cw.shape[1] == 0:
        return cw

    cw = cw.ffill(limit=5).dropna(axis=0, how="any")
    if len(cw) < min_history:
        return cw.iloc[0:0]   # signal "not enough" to callers
    return cw


def annualised_stats(close_wide: pd.DataFrame):
    """Annualised mean-return vector and covariance matrix from a price panel."""
    rets = close_wide.pct_change().dropna()
    return stats_from_returns(rets)


def stats_from_returns(rets: pd.DataFrame):
    """Annualised (mu, cov, symbols) from a daily-return panel."""
    mu = rets.mean().to_numpy() * TRADING_DAYS
    cov = rets.cov().to_numpy() * TRADING_DAYS
    return mu, cov, list(rets.columns)


def _port_perf(w, mu, cov):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol


def optimise_weights(mu, cov, objective: str, max_weight: float = 0.35,
                     rf: float = 0.06) -> np.ndarray:
    """Solve one long-only, fully-invested allocation for the given objective."""
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
    total = w.sum()
    return w / total if total > 0 else w0


def construct_portfolios(
    close_wide: pd.DataFrame,
    rf: float = 0.06,
    lookback_days: int | None = None,
) -> dict:
    """Return allocations + stats + justification for the three profiles."""
    close_wide = clean_universe(close_wide, lookback_days=lookback_days)
    if close_wide.shape[1] == 0:
        raise ValueError("No symbols have enough history to build a portfolio.")
    mu, cov, symbols = annualised_stats(close_wide)

    out = {}
    for name, cfg in PROFILES.items():
        w = optimise_weights(mu, cov, cfg["objective"], cfg["max_weight"], rf)
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
            "n_universe": len(symbols),
        }
    return out


def sector_allocation(portfolio: dict, market) -> dict:
    """Aggregate a weight dict into sector exposures for explainability."""
    sectors = {}
    for sym, w in portfolio.get("weights", {}).items():
        sec = market.sector_of(sym)
        sectors[sec] = round(sectors.get(sec, 0.0) + w, 4)
    return dict(sorted(sectors.items(), key=lambda kv: -kv[1]))
