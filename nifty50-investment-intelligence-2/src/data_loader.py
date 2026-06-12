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


def load_market_data(data_dir: str, symbols: list[str] | None = None) -> MarketData:
    csvs = glob.glob(os.path.join(data_dir, "*.csv"))
    meta_path = None
    frames = []
    for path in csvs:
        name = os.path.splitext(os.path.basename(path))[0]
        if name.lower() in ("stock_metadata", "metadata", "nifty50"):
            meta_path = path
            continue
        if symbols and name.upper() not in {s.upper() for s in symbols}:
            continue
        raw = pd.read_csv(path)
        frames.append(_normalise_columns(raw, symbol_fallback=name.upper()))

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
