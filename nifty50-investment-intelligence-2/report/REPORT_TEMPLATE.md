# NIFTY-50 Investment Intelligence Platform
### Technical Report — Open Projects 2026 (AI/ML)

**Team:** Rukdimax ·

**Team Members** -

S R Nivedhitha - 22411030 - Geophysical Technology - 4th Year

Somil Agrawal - Chemical Engineering - 3rd Year
           
**Repo:** [‹GitHub URL›](https://github.com/vinnie1e/nifty50-cult.git)

---

## 1. Exploratory Data Analysis
- Coverage: ‹n_symbols› symbols, ‹n_trading_days› trading days
  (‹date_start› → ‹date_end›).
- **Price trends** *(Fig: normalised_prices.png)* — long-run cumulative growth by
  symbol/sector.
- **Risk/return map** *(Fig: risk_return_map.png)* — annualised return vs
  volatility; identify the efficient names.
- **Correlation structure** *(Fig: correlation_heatmap.png)* — mean pairwise
  correlation ‹mean_pairwise_correlation›; discuss diversification potential.
- Sector observations: which sectors led/lagged, and volatility clustering during
  known market events (2008, 2020 COVID crash).

## 2. Feature Engineering
Indicators derived purely from provided OHLCV data (`src/features.py`):
returns/log-returns, SMA(5/10/20/50/200), EMA(12/26), RSI(14),
MACD(+signal/hist), Bollinger Bands (+%b, bandwidth), rolling volatility(20),
momentum(10/20), ATR(14), volume MA. Rationale: trend (MAs), momentum (RSI,
MACD, momentum), and volatility (Bollinger, ATR, rolling σ) each capture a
distinct market regime signal.

## 3. Methodology
- **Predictor.** Two tasks: forward ‹horizon›-day **return regression** and
  **up/down direction classification**, both with gradient-boosted trees.
- **Validation.** Walk-forward `TimeSeriesSplit` (‹n_splits› folds) — no future
  leakage, no shuffling. Compared against naive baselines (predict-zero /
  predict-majority).
- **Portfolio.** Mean–variance optimisation (long-only, SLSQP, per-name caps).
- **Risk.** Standard historical risk measures on daily returns.

## 4. Model Architecture
`GradientBoostingRegressor` / `GradientBoostingClassifier`
(n_estimators=200, depth=3, lr=0.05, subsample=0.8). Inputs: ‹k› technical
features. Justify the choice: robust to noise, handles non-linear indicator
interactions, exposes feature importances for explainability.

## 5. Experimental Results — Predictor

| Symbol | RMSE ↓ | MAE ↓ | R² | Directional Acc. ↑ | Baseline DirAcc |
|--------|-------:|------:|---:|-------------------:|----------------:|
| ‹SYM›  | ‹› | ‹› | ‹› | ‹› | ‹› |
| ...    |   |   |   |   |   |

*(From `predictions` in `intelligence_report.json`.)*
**Interpretation.** Stock return is near-random-walk; honest directional accuracy
modestly above 50% and beating the predict-zero MAE baseline is the realistic
bar. State clearly where the model adds value and where it doesn't.

## 6. Portfolio Construction Logic

| Profile | E[return] | Volatility | Sharpe | Top holdings |
|---------|----------:|-----------:|-------:|--------------|
| Conservative | ‹› | ‹› | ‹› | ‹› |
| Balanced     | ‹› | ‹› | ‹› | ‹› |
| Aggressive   | ‹› | ‹› | ‹› | ‹› |

Explain the objective per profile (min-variance / max-Sharpe / return-tilt),
the per-name caps, and the resulting **sector allocations** (from
`sector_allocation`). Justify each with the quantitative stats above.

## 7. Risk Assessment Methodology
Per-stock and per-portfolio: annualised volatility, Sharpe, Sortino, maximum
drawdown, Calmar. Present a table for the recommended portfolios and discuss
drawdown behaviour during stress periods (e.g. March 2020).

## 8. Explainability Techniques
Feature importances per forecast (`feature_importance`/`explain_forecast`):
report the top signals (e.g. RSI, 20-day momentum, MACD hist) and translate them
into plain-language reasons a user would trust. Optionally include the anomaly
detector's flagged events.

## 9. Key Insights
- ‹Which sectors offered the best risk-adjusted return historically?›
- ‹How much does diversification reduce portfolio volatility vs single names?›
- ‹Where did the predictor genuinely help vs where the market was efficient?›
- ‹What stress events did the anomaly detector correctly surface?›

## 10. Limitations & Future Work
Survivorship bias in index membership, regime shifts, transaction costs not
modelled, no live data by design. Future: LSTM/temporal models, regime-aware
allocation, backtested rebalancing, deployment as a cloud service.
