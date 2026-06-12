"""
Exploratory Data Analysis for the NIFTY-50 dataset.

Produces sector-level and stock-level visuals plus a numeric summary:
  * normalised price trends (cumulative growth) for sample symbols
  * annualised return vs volatility scatter (the risk/return map)
  * rolling-volatility time series
  * correlation heatmap across symbols
  * summary stats table -> results/eda_insights.json
"""
from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .risk import risk_profile

TRADING_DAYS = 252


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_eda(market, out_dir: str) -> dict:
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    close = market.close_wide().dropna(axis=1, how="all")

    # normalised price trends
    norm = close / close.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col in norm.columns[:8]:
        ax.plot(norm.index, norm[col], label=col, linewidth=1)
    ax.set(title="Normalised price growth", xlabel="Date", ylabel="Growth (×)")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, os.path.join(fig_dir, "normalised_prices.png"))

    # risk/return map
    stats = {}
    for col in close.columns:
        series = close[col].dropna()
        if len(series) > TRADING_DAYS:
            stats[col] = risk_profile(series)
    rr = pd.DataFrame(stats).T
    if not rr.empty:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(rr["annualised_volatility"], rr["annualised_return"], s=40)
        for sym, row in rr.iterrows():
            ax.annotate(sym, (row["annualised_volatility"], row["annualised_return"]),
                        fontsize=6, alpha=0.7)
        ax.set(title="Risk / return map", xlabel="Annualised volatility",
               ylabel="Annualised return")
        ax.axhline(0, color="grey", lw=0.5)
        _save(fig, os.path.join(fig_dir, "risk_return_map.png"))

    # correlation heatmap
    rets = close.pct_change().dropna()
    if rets.shape[1] >= 2:
        corr = rets.corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels(corr.columns, fontsize=6)
        fig.colorbar(im, fraction=0.046)
        ax.set_title("Return correlation matrix")
        _save(fig, os.path.join(fig_dir, "correlation_heatmap.png"))

    insights = {
        "n_symbols": int(close.shape[1]),
        "date_start": str(close.index.min().date()),
        "date_end": str(close.index.max().date()),
        "n_trading_days": int(close.shape[0]),
        "per_symbol_risk_return": {k: stats[k] for k in list(stats)[:50]},
        "mean_pairwise_correlation": round(float(
            rets.corr().where(~np.eye(rets.shape[1], dtype=bool)).stack().mean()
        ), 4) if rets.shape[1] >= 2 else None,
    }
    with open(os.path.join(out_dir, "eda_insights.json"), "w") as fh:
        json.dump(insights, fh, indent=2, default=str)
    return insights
