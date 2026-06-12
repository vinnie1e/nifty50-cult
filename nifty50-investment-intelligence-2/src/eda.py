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


INK = "#1a1611"
GREEN = "#1d6b4c"
GRID = "#d9cfbd"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, color=INK, pad=8, loc="left", weight="bold")
    ax.set_xlabel(xlabel, fontsize=8, color=INK)
    ax.set_ylabel(ylabel, fontsize=8, color=INK)
    ax.tick_params(labelsize=7, colors=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.6)


def run_eda(market, out_dir: str) -> dict:
    from .portfolio import clean_universe  # coverage-aware subset

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    close = market.close_wide().dropna(axis=1, how="all")

    # A coherent, continuously-listed subset for the visuals (avoids old-ticker
    # clutter and non-overlapping eras). Falls back to all names if too few.
    clean = clean_universe(close, coverage=0.90)
    panel = clean if clean.shape[1] >= 5 else close.dropna(axis=1, how="any")

    # ---- normalised price trends (representative long-history names) ----
    norm = panel / panel.iloc[0]
    show = list(norm.columns[:9])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for col in show:
        ax.plot(norm.index, norm[col], label=col, linewidth=1.1)
    _style(ax, "Normalised price growth — growth of 1 unit", "", "Growth (×)")
    ax.legend(fontsize=6.5, ncol=3, frameon=False)
    _save(fig, os.path.join(fig_dir, "normalised_prices.png"))

    # ---- risk/return map (all names; annotate the clean subset only) ----
    stats = {}
    for col in close.columns:
        series = close[col].dropna()
        if len(series) > TRADING_DAYS:
            stats[col] = risk_profile(series)
    rr = pd.DataFrame(stats).T
    if not rr.empty:
        fig, ax = plt.subplots(figsize=(6.6, 5))
        ax.scatter(rr["annualised_volatility"], rr["annualised_return"],
                   s=26, c=GREEN, alpha=0.55, edgecolor="none")
        for sym in panel.columns:
            if sym in rr.index:
                row = rr.loc[sym]
                ax.annotate(sym, (row["annualised_volatility"], row["annualised_return"]),
                            fontsize=5.6, alpha=0.85, color=INK)
        _style(ax, "Risk / return map — annualised", "Annualised volatility",
               "Annualised return")
        ax.axhline(0, color="#9b2c2c", lw=0.6)
        _save(fig, os.path.join(fig_dir, "risk_return_map.png"))

    # ---- correlation (pairwise — robust to non-overlapping history) ----
    rets_all = close.pct_change()
    corr_all = rets_all.corr(min_periods=TRADING_DAYS)
    mean_corr = float(
        corr_all.where(~np.eye(len(corr_all), dtype=bool)).stack().mean())

    hm = panel.pct_change().corr()
    if hm.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(6, 5.2))
        im = ax.imshow(hm.values, cmap="BrBG", vmin=-1, vmax=1)
        ax.set_xticks(range(len(hm)))
        ax.set_xticklabels(hm.columns, rotation=90, fontsize=5.4, color=INK)
        ax.set_yticks(range(len(hm)))
        ax.set_yticklabels(hm.columns, fontsize=5.4, color=INK)
        cb = fig.colorbar(im, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6)
        ax.set_title("Return correlation matrix — continuously-listed names",
                     fontsize=10, color=INK, loc="left", pad=8, weight="bold")
        _save(fig, os.path.join(fig_dir, "correlation_heatmap.png"))

    insights = {
        "n_symbols": int(close.shape[1]),
        "n_analysis_universe": int(panel.shape[1]),
        "date_start": str(close.index.min().date()),
        "date_end": str(close.index.max().date()),
        "n_trading_days": int(close.shape[0]),
        "per_symbol_risk_return": {k: stats[k] for k in list(stats)[:50]},
        "mean_pairwise_correlation": round(mean_corr, 4) if mean_corr == mean_corr else None,
    }
    with open(os.path.join(out_dir, "eda_insights.json"), "w") as fh:
        json.dump(insights, fh, indent=2, default=str)
    return insights
