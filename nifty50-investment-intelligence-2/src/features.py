"""
Feature engineering: technical indicators derived from OHLCV data.

All indicators are computed from the provided dataset only (no external feeds),
satisfying the competition constraints. Functions take a single-symbol,
date-sorted DataFrame and return it with indicator columns appended.

Implemented: returns, log-returns, SMA, EMA, RSI, MACD (+signal/hist),
Bollinger Bands (+%b, bandwidth), rolling volatility, momentum, ATR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()
    return df


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, window=20, n_std=2):
    mid = sma(series, window)
    std = series.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (series - lower) / (upper - lower)
    bandwidth = (upper - lower) / mid
    return mid, upper, lower, pct_b, bandwidth


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append the full indicator set to a single-symbol DataFrame."""
    df = add_returns(df)
    c = df["close"]
    for w in (5, 10, 20, 50, 200):
        df[f"sma_{w}"] = sma(c, w)
    for s in (12, 26):
        df[f"ema_{s}"] = ema(c, s)
    df["rsi_14"] = rsi(c, 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(c)
    df["bb_mid"], df["bb_upper"], df["bb_lower"], df["bb_pctb"], df["bb_bw"] = bollinger(c)
    df["volatility_20"] = df["return"].rolling(20).std() * np.sqrt(252)
    df["momentum_10"] = c.pct_change(10)
    df["momentum_20"] = c.pct_change(20)
    if {"high", "low"}.issubset(df.columns):
        df["atr_14"] = atr(df, 14)
    df["volume_ma_20"] = df["volume"].rolling(20).mean() if "volume" in df else np.nan
    return df


# columns used as model inputs (after add_all_indicators)
FEATURE_COLUMNS = [
    "return", "log_return", "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_12", "ema_26", "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_pctb", "bb_bw", "volatility_20", "momentum_10", "momentum_20", "atr_14",
]


def build_supervised(df: pd.DataFrame, horizon: int = 5):
    """Create a supervised frame: features at t, target = forward return over `horizon`.

    Returns X, y_return (continuous), y_direction (1 if up else 0), and the
    aligned index dates.
    """
    df = add_all_indicators(df)
    fwd_return = df["close"].shift(-horizon) / df["close"] - 1.0
    df = df.assign(fwd_return=fwd_return)
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    data = df.dropna(subset=cols + ["fwd_return"]).reset_index(drop=True)
    X = data[cols]
    y_return = data["fwd_return"]
    y_direction = (y_return > 0).astype(int)
    return X, y_return, y_direction, data["date"], cols


def build_volatility_supervised(df: pd.DataFrame, horizon: int = 10):
    """Supervised frame for volatility forecasting.

    Target = annualised realised volatility over the NEXT `horizon` days. The
    persistence baseline (trailing `horizon`-day realised vol) is returned too so
    the forecaster can be judged against "tomorrow looks like today".
    """
    df = add_all_indicators(df)
    ann = np.sqrt(252)
    trailing_vol = df["return"].rolling(horizon).std() * ann      # info up to t
    fwd_vol = trailing_vol.shift(-horizon)                        # realised over t+1..t+h
    df = df.assign(_fwd_vol=fwd_vol, _persist_vol=trailing_vol)
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    data = df.dropna(subset=cols + ["_fwd_vol", "_persist_vol"]).reset_index(drop=True)
    X = data[cols]
    y_vol = data["_fwd_vol"]
    baseline = data["_persist_vol"]
    return X, y_vol, baseline, data["date"], cols
