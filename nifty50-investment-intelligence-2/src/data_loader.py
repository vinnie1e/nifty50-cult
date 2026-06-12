"""
Data loading for the NIFTY-50 stock-market dataset.

Expected layout (from the Kaggle dataset 'rohanrao/nifty50-stock-market-data'):

    data/
      RELIANCE.csv  TCS.csv  INFY.csv  ...     <- one CSV per symbol
      stock_metadata.csv                        <- symbol, company, sector, industry

Each per-symbol CSV has daily OHLCV-style columns. Column names vary slightly
between sources, so we normalise them to:

    date, symbol, open, high, low, close, volume, turnover

`synthetic_prices()` fabricates a small multi-stock panel (geometric brownian
motion with per-sector drift) so the whole pipeline can be smoke-tested without
the download.
"""
from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# canonical name -> possible source column names
_COLMAP = {
    "date": ["Date", "date"],
    "open": ["Open", "open"],
    "high": ["High", "high"],
    "low": ["Low", "low"],
    "close": ["Close", "close"],
    "volume": ["Volume", "volume", "Traded Volume"],
    "turnover": ["Turnover", "turnover"],
    "symbol": ["Symbol", "symbol"],
}


@dataclass
class MarketData:
    prices: pd.DataFrame                 # long: date, symbol, open..turnover
    metadata: pd.DataFrame | None = None # symbol, company, sector, industry
    _close_wide: pd.DataFrame = field(default=None, repr=False)

    @property
    def symbols(self):
        return sorted(self.prices["symbol"].unique())

    def close_wide(self) -> pd.DataFrame:
        """Date-indexed wide matrix of close prices (one column per symbol)."""
        if self._close_wide is None:
            self._close_wide = (
                self.prices.pivot_table(index="date", columns="symbol", values="close")
                .sort_index()
            )
        return self._close_wide

    def sector_of(self, symbol: str) -> str:
        if self.metadata is None:
            return "Unknown"
        row = self.metadata.loc[self.metadata["symbol"] == symbol]
        return "Unknown" if row.empty else str(row.iloc[0].get("sector", "Unknown"))

    def for_symbol(self, symbol: str) -> pd.DataFrame:
        df = self.prices[self.prices["symbol"] == symbol].sort_values("date")
        return df.reset_index(drop=True)


def _adjust_splits(df: pd.DataFrame, jump: float = 0.40) -> pd.DataFrame:
    """Back-adjust prices for stock splits / bonus issues.

    The Kaggle OHLCV is *unadjusted*, so a 1:2 split shows up as a spurious ~-50%
    one-day "return". For NIFTY-50 large caps a genuine >40% single-day move is
    essentially nonexistent, so we treat such jumps as corporate actions and
    rescale all earlier prices onto the post-action scale (volume rescaled
    inversely). This keeps returns, volatility, and the predictor honest.
    """
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    factor = np.ones(n)
    cum = 1.0
    for i in range(n - 1, 0, -1):
        if close[i - 1] > 0:
            ratio = close[i] / close[i - 1]
            if ratio > 1 + jump or ratio < 1 - jump:
                cum *= ratio
        factor[i - 1] = cum
    for col in ("open", "high", "low", "close"):
        if col in df:
            df[col] = df[col].to_numpy(dtype=float) * factor
    if "volume" in df:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["volume"] = df["volume"].to_numpy(dtype=float) / np.where(factor == 0, 1, factor)
    return df


def _normalise_columns(df: pd.DataFrame, symbol_fallback: str) -> pd.DataFrame:
    out = {}
    for canon, options in _COLMAP.items():
        for opt in options:
            if opt in df.columns:
                out[canon] = df[opt]
                break
    res = pd.DataFrame(out)
    if "symbol" not in res:
        res["symbol"] = symbol_fallback
    res["date"] = pd.to_datetime(res["date"])
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        if c in res:
            res[c] = pd.to_numeric(res[c], errors="coerce")
        else:
            res[c] = np.nan
    return res.dropna(subset=["close"]).reset_index(drop=True)


def load_market_data(data_dir: str, symbols: list[str] | None = None,
                     adjust_splits: bool = True) -> MarketData:
    csvs = glob.glob(os.path.join(data_dir, "*.csv"))
    meta_path = None
    frames = []
    for path in csvs:
        name = os.path.splitext(os.path.basename(path))[0]
        low = name.lower()
        if low in ("stock_metadata", "metadata"):
            meta_path = path
            continue
        # skip combined / index files (e.g. NIFTY50_all.csv) — they repeat every
        # symbol and would double-count against the per-symbol CSVs.
        if low.startswith("nifty"):
            continue
        if symbols and name.upper() not in {s.upper() for s in symbols}:
            continue
        raw = pd.read_csv(path)
        frame = _normalise_columns(raw, symbol_fallback=name.upper())
        if adjust_splits and len(frame) > 1:
            frame = _adjust_splits(frame)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No per-symbol CSVs found in {data_dir!r}. Download the dataset from "
            "Kaggle (rohanrao/nifty50-stock-market-data) — see README."
        )

    prices = pd.concat(frames, ignore_index=True)
    metadata = None
    if meta_path:
        m = pd.read_csv(meta_path)
        m.columns = [c.strip().lower() for c in m.columns]
        rename = {}
        for c in m.columns:
            if "symbol" in c:
                rename[c] = "symbol"
            elif "company" in c or "name" in c:
                rename[c] = "company"
            elif "sector" in c:
                rename[c] = "sector"
            elif "industry" in c:
                rename[c] = "industry"
        metadata = m.rename(columns=rename)
        # the NIFTY-50 metadata ships an NSE macro-sector under "Industry" and no
        # explicit "sector" column — use it so sector analysis works.
        if "sector" not in metadata.columns and "industry" in metadata.columns:
            metadata["sector"] = metadata["industry"].astype(str).str.title()
    return MarketData(prices=prices, metadata=metadata)


def synthetic_prices(
    n_days: int = 1500,
    sectors: dict | None = None,
    seed: int = 0,
) -> MarketData:
    """Generate a small multi-sector price panel via geometric Brownian motion."""
    rng = np.random.default_rng(seed)
    if sectors is None:
        sectors = {
            "Banking": ["HDFCBANK", "ICICIBANK"],
            "IT": ["TCS", "INFY"],
            "Energy": ["RELIANCE", "ONGC"],
            "Pharma": ["SUNPHARMA", "CIPLA"],
        }
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    rows = []
    meta_rows = []
    for sector, syms in sectors.items():
        sector_drift = rng.normal(0.0004, 0.0002)
        for sym in syms:
            mu = sector_drift + rng.normal(0, 0.0002)
            vol = rng.uniform(0.012, 0.025)
            price = rng.uniform(200, 2000)
            closes = []
            for _ in range(n_days):
                price *= np.exp(rng.normal(mu, vol))
                closes.append(price)
            closes = np.array(closes)
            highs = closes * (1 + np.abs(rng.normal(0, 0.005, n_days)))
            lows = closes * (1 - np.abs(rng.normal(0, 0.005, n_days)))
            opens = np.r_[closes[0], closes[:-1]]
            vols = rng.integers(1e5, 5e6, n_days)
            for i in range(n_days):
                rows.append((dates[i], sym, opens[i], highs[i], lows[i],
                             closes[i], int(vols[i]), float(closes[i] * vols[i])))
            meta_rows.append((sym, f"{sym} Ltd", sector, sector))
    prices = pd.DataFrame(
        rows, columns=["date", "symbol", "open", "high", "low",
                       "close", "volume", "turnover"])
    metadata = pd.DataFrame(
        meta_rows, columns=["symbol", "company", "sector", "industry"])
    return MarketData(prices=prices, metadata=metadata)


if __name__ == "__main__":
    md = synthetic_prices()
    print(f"symbols={md.symbols}")
    print(md.close_wide().tail())
