"""
Explainability + anomaly detection.

Explainability (three complementary lenses):
  * Impurity importances — fast, global, model-internal.
  * Permutation importances — model-agnostic, measured on held-out targets, so
    they reflect predictive value rather than tree-split frequency. This is the
    honest headline measure.
  * Per-prediction explanation — translates a single forecast into the few
    indicators that drove it (SHAP if installed, impurity fallback otherwise).

Anomaly detection (optional task) flags the three event types named in the brief
— volatility spikes, extreme drawdowns, and unusual trading volume — using
rolling robust z-scores derived only from the provided dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def feature_importance(model, feature_columns) -> dict:
    """Return sorted {feature: importance} for a tree-based sklearn model (impurity)."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        pairs = sorted(zip(feature_columns, imp), key=lambda kv: -kv[1])
        return {k: round(float(v), 4) for k, v in pairs}
    return {}


def permutation_feature_importance(model, X, y, n_repeats: int = 10,
                                   random_state: int = 42) -> dict:
    """Model-agnostic importance: drop in score when each feature is shuffled."""
    try:
        r = permutation_importance(model, X, y, n_repeats=n_repeats,
                                   random_state=random_state, n_jobs=1)
    except Exception:
        return {}
    pairs = sorted(zip(list(X.columns), r.importances_mean), key=lambda kv: -kv[1])
    return {k: round(float(v), 5) for k, v in pairs}


def explain_forecast(model, feature_columns, top_n: int = 5) -> str:
    imp = feature_importance(model, feature_columns)
    top = list(imp.items())[:top_n]
    if not top:
        return "Model does not expose feature importances."
    drivers = ", ".join(f"{name} ({w:.0%})" for name, w in top)
    return f"Top signals driving this forecast: {drivers}."


def explain_prediction(model, x_row, feature_columns, top_n: int = 5) -> dict:
    """Explain ONE forecast: signed per-feature contributions.

    Uses SHAP (TreeExplainer) when available for exact local attributions;
    otherwise falls back to impurity importances (unsigned). Returns the method
    used so the report can state it honestly.
    """
    x = np.asarray(x_row, dtype=float).reshape(1, -1)
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        vals = np.asarray(explainer.shap_values(x)).reshape(-1)
        pairs = sorted(zip(feature_columns, vals), key=lambda kv: -abs(kv[1]))[:top_n]
        return {"method": "shap",
                "contributions": {k: round(float(v), 5) for k, v in pairs}}
    except Exception:
        imp = feature_importance(model, feature_columns)
        pairs = list(imp.items())[:top_n]
        return {"method": "impurity",
                "contributions": {k: float(v) for k, v in pairs}}


def detect_anomalies(df: pd.DataFrame, vol_z: float = 3.0, vol_window: int = 20,
                     drawdown_threshold: float = -0.10) -> pd.DataFrame:
    """Flag volatility spikes, extreme drawdowns, and unusual volume.

    * volatility spike — daily return > `vol_z` rolling-σ from its rolling mean.
    * extreme drawdown — running peak-to-trough decline past `drawdown_threshold`.
    * unusual volume   — volume > `vol_z` rolling-σ above its rolling mean.
    """
    df = df.sort_values("date").copy()
    ret = df["close"].pct_change()
    df["return_z"] = (ret - ret.rolling(vol_window).mean()) / ret.rolling(vol_window).std()

    # running drawdown from the cumulative peak
    cum = (1 + ret.fillna(0)).cumprod()
    df["drawdown"] = cum / cum.cummax() - 1.0

    if "volume" in df.columns:
        vmean = df["volume"].rolling(vol_window).mean()
        vstd = df["volume"].rolling(vol_window).std()
        volume_z = (df["volume"] - vmean) / vstd
    else:
        volume_z = pd.Series(np.nan, index=df.index)

    reasons = []
    for rz, dd, vz in zip(df["return_z"], df["drawdown"], volume_z):
        parts = []
        if pd.notna(rz) and abs(rz) >= vol_z:
            parts.append(f"volatility spike (z={rz:.1f})")
        if pd.notna(dd) and dd <= drawdown_threshold:
            parts.append(f"extreme drawdown ({dd:.0%})")
        if pd.notna(vz) and vz >= vol_z:
            parts.append("unusual volume")
        reasons.append("; ".join(parts))
    df["anomaly"] = reasons

    flagged = df[df["anomaly"] != ""]
    return flagged[["date", "close", "return_z", "drawdown", "anomaly"]].reset_index(drop=True)
