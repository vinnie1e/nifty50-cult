"""
Walk-forward portfolio backtest.

`construct_portfolios` reports the *in-sample* expected return/vol/Sharpe of an
allocation estimated on the whole history. That is the right thing to recommend
today, but it says nothing about how the strategy would actually have performed.
This module answers that question honestly:

  * Estimate each profile's weights on a TRAILING window only (no lookahead).
  * Hold those weights, letting them drift, until the next rebalance date.
  * Record realised daily portfolio returns out-of-sample.
  * Compare against an equal-weight benchmark on the same investable universe.

The result is an equity curve + realised risk metrics per profile — the evidence
that turns a recommendation into a decision-support claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .portfolio import PROFILES, clean_universe, optimise_weights, stats_from_returns
from . import risk as riskmod


def backtest_portfolios(
    close_wide: pd.DataFrame,
    rf: float = 0.06,
    lookback: int = 252,
    rebalance: int = 63,
    min_history: int = 252,
) -> dict:
    """Walk-forward backtest of all profiles + an equal-weight benchmark.

    Args:
        close_wide:  date-indexed wide close-price matrix.
        rf:          annual risk-free rate (for Sharpe/Sortino).
        lookback:    trailing trading days used to estimate mu/cov at each rebalance.
        rebalance:   rebalance cadence in trading days (63 ~ quarterly).
        min_history: minimum observations for a name to enter the universe.

    Returns a dict with per-strategy realised metrics, the equity curves, and the
    backtest configuration.
    """
    cw = clean_universe(close_wide, min_history=min_history)
    rets = cw.pct_change().dropna()
    if len(rets) <= lookback + rebalance:
        raise ValueError("Not enough overlapping history to backtest.")

    n = rets.shape[1]
    strategies = list(PROFILES) + ["EqualWeight"]
    daily = {s: [] for s in strategies}
    out_dates = []

    current = None  # {strategy: drifting weight vector}
    for i in range(lookback, len(rets)):
        # rebalance using ONLY information available before day i
        if current is None or (i - lookback) % rebalance == 0:
            mu, cov, _ = stats_from_returns(rets.iloc[i - lookback:i])
            current = {
                name: optimise_weights(mu, cov, cfg["objective"], cfg["max_weight"], rf)
                for name, cfg in PROFILES.items()
            }
            current["EqualWeight"] = np.repeat(1 / n, n)

        r = rets.iloc[i].to_numpy()
        out_dates.append(rets.index[i])
        for s in strategies:
            w = current[s]
            daily[s].append(float(w @ r))
            grown = w * (1.0 + r)            # let weights drift with returns
            total = grown.sum()
            current[s] = grown / total if total > 0 else w

    idx = pd.DatetimeIndex(out_dates)
    metrics, curves = {}, {}
    for s in strategies:
        series = pd.Series(daily[s], index=idx)
        equity = (1.0 + series).cumprod()
        metrics[s] = {
            "annualised_return": round(riskmod.annualised_return(series), 4),
            "annualised_volatility": round(riskmod.annualised_volatility(series), 4),
            "sharpe": round(riskmod.sharpe_ratio(series, rf), 4),
            "sortino": round(riskmod.sortino_ratio(series, rf), 4),
            "max_drawdown": round(riskmod.max_drawdown(equity), 4),
            "calmar": round(riskmod.calmar_ratio(equity), 4),
            "total_return": round(float(equity.iloc[-1] - 1.0), 4),
        }
        curves[s] = equity

    return {
        "metrics": metrics,
        "equity_curves": curves,
        "config": {
            "lookback": lookback, "rebalance": rebalance,
            "n_universe": n, "oos_start": str(idx.min().date()),
            "oos_end": str(idx.max().date()), "oos_days": len(idx),
            "rf": rf,
        },
    }
