"""
Explainability + anomaly detection.

Explainability:
  * Permutation / impurity feature importances from the predictor models, so a
    user can see WHICH indicators drove a forecast (e.g. "RSI and 20-day
    momentum were the strongest signals").

Anomaly detection (optional task):
  * Flags volatility spikes, extreme drawdown days, and unusual volume using
    robust z-scores on rolling statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def feature_importance(model, feature_columns) -> dict:
    """Return sorted {feature: importance} for a tree-based sklearn model."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        pairs = sorted(zip(feature_columns, imp), key=lambda kv: -kv[1])
        return {k: round(float(v), 4) for k, v in pairs}
    return {}


def explain_forecast(model, feature_columns, top_n: int = 5) -> str:
    imp = feature_importance(model, feature_columns)
    top = list(imp.items())[:top_n]
    if not top:
        return "Model does not expose feature importances."
    drivers = ", ".join(f"{name} ({w:.0%})" for name, w in top)
    return f"Top signals driving this forecast: {drivers}."


def detect_anomalies(df: pd.DataFrame, vol_z=3.0, vol_window=20) -> pd.DataFrame:
    """Return rows flagged as anomalous with the reason(s)."""
    df = df.sort_values("date").copy()
    ret = df["close"].pct_change()
    roll_std = ret.rolling(vol_window).std()
    roll_mean = ret.rolling(vol_window).mean()

    # robust z-score of daily return
    z = (ret - roll_mean) / roll_std
    df["return_z"] = z

    flags = []
    for _, row in df.iterrows():
        reasons = []
        if pd.notna(row["return_z"]) and abs(row["return_z"]) >= vol_z:
            reasons.append(f"volatility spike (z={row['return_z']:.1f})")
        flags.append("; ".join(reasons))
    df["anomaly"] = flags

    # unusual volume
    if "volume" in df.columns:
        vmean = df["volume"].rolling(vol_window).mean()
        vstd = df["volume"].rolling(vol_window).std()
        vol_z_score = (df["volume"] - vmean) / vstd
        df.loc[vol_z_score >= vol_z, "anomaly"] = (
            df.loc[vol_z_score >= vol_z, "anomaly"].astype(str)
            + " | unusual volume"
        )

    return df[df["anomaly"].astype(bool) & (df["anomaly"] != "")][
        ["date", "close", "return_z", "anomaly"]
    ].reset_index(drop=True)
