"""
Stock Predictor Engine.

Two complementary tasks, both validated with a TIME-SERIES split (no shuffling —
we never train on the future):

  * Return regression  -> predict forward N-day return.  Metrics: MAE, RMSE, R².
  * Direction classification -> predict up/down.          Metric: Directional Accuracy.

Models are deliberately simple and robust (gradient-boosted trees / ridge),
because on noisy financial data, interpretable + regularised beats fragile-deep.
A naive baseline (predict 0 return / predict majority class) is always reported
so the model has to earn its keep.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from .features import build_supervised


def _directional_accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def train_return_predictor(df: pd.DataFrame, horizon: int = 5, n_splits: int = 5):
    """Forward-return regression with walk-forward CV. Returns metrics + fitted model."""
    X, y_ret, _, dates, cols = build_supervised(df, horizon=horizon)
    if len(X) < 100:
        raise ValueError("Not enough rows after feature construction.")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, r2s, dir_accs, base_maes = [], [], [], [], []
    for tr, te in tscv.split(X):
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            random_state=42)
        model.fit(X.iloc[tr], y_ret.iloc[tr])
        pred = model.predict(X.iloc[te])
        true = y_ret.iloc[te].to_numpy()
        maes.append(mean_absolute_error(true, pred))
        rmses.append(np.sqrt(mean_squared_error(true, pred)))
        r2s.append(r2_score(true, pred))
        dir_accs.append(_directional_accuracy(true, pred))
        base_maes.append(mean_absolute_error(true, np.zeros_like(true)))  # predict-0

    # final model on all data for downstream use
    final = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        random_state=42).fit(X, y_ret)

    metrics = {
        "task": "return_regression",
        "horizon_days": horizon,
        "MAE": float(np.mean(maes)),
        "RMSE": float(np.mean(rmses)),
        "R2": float(np.mean(r2s)),
        "DirectionalAccuracy": float(np.mean(dir_accs)),
        "baseline_MAE_predict_zero": float(np.mean(base_maes)),
        "n_samples": int(len(X)),
    }
    return final, metrics, cols


def train_direction_classifier(df: pd.DataFrame, horizon: int = 5, n_splits: int = 5):
    """Up/down classification with walk-forward CV."""
    X, _, y_dir, dates, cols = build_supervised(df, horizon=horizon)
    if len(X) < 100:
        raise ValueError("Not enough rows after feature construction.")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, base_accs = [], []
    for tr, te in tscv.split(X):
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            random_state=42)
        clf.fit(X.iloc[tr], y_dir.iloc[tr])
        pred = clf.predict(X.iloc[te])
        true = y_dir.iloc[te].to_numpy()
        accs.append(float(np.mean(pred == true)))
        majority = int(round(y_dir.iloc[tr].mean()))
        base_accs.append(float(np.mean(true == majority)))

    final = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        random_state=42).fit(X, y_dir)
    metrics = {
        "task": "direction_classification",
        "horizon_days": horizon,
        "Accuracy": float(np.mean(accs)),
        "baseline_majority_accuracy": float(np.mean(base_accs)),
        "n_samples": int(len(X)),
    }
    return final, metrics, cols


def predict_symbol(market, symbol: str, horizon: int = 5):
    """Convenience: train both predictors for one symbol and return their metrics."""
    df = market.for_symbol(symbol)
    reg_model, reg_metrics, cols = train_return_predictor(df, horizon)
    clf_model, clf_metrics, _ = train_direction_classifier(df, horizon)
    return {
        "symbol": symbol,
        "regression": reg_metrics,
        "classification": clf_metrics,
        "models": {"regressor": reg_model, "classifier": clf_model},
        "feature_columns": cols,
    }
