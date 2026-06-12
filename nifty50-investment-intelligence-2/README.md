# NIFTY-50 Investment Intelligence Platform

Open Projects 2026 · AI / ML  ·  Team rukdimaxx Somil Agrawal (23112099) · S R Nivedhitha (22411030)


An AI-powered **decision-support** platform built on the NIFTY-50 historical
market dataset (Jan 2000 – Apr 2021). It goes beyond price prediction to deliver
practical investment intelligence: a **stock predictor engine**, a **portfolio
construction module** for different investor profiles, a **risk-assessment
module**, explainability, and anomaly detection — all served through an
interactive **Streamlit prototype**.

> Built using only the organiser-provided dataset. No live feeds, financial
> APIs, news, or alternative data are used (per the competition constraints).

---

## Deliverables map

| Deliverable | Where |
|-------------|-------|
| Working prototype | `app.py` (Streamlit) |
| Technical report (PDF) | `report/NIFTY50_Investment_Intelligence_Report.pdf` — generated from results by `report/build_report.py` |
| Source code | `src/` |
| Model files | `results/models/*.joblib` (representative set committed; full set regenerates — see below) |
| README | this file |

## What it does

| Module | File | Output |
|--------|------|--------|
| Data loading & normalisation | `src/data_loader.py` | tidy price panel + metadata |
| Technical indicators | `src/features.py` | SMA, EMA, RSI, MACD, Bollinger, ATR, momentum, volatility |
| EDA | `src/eda.py` | risk/return map, correlation heatmap, trends |
| Stock predictor engine | `src/predictor.py` | return regression + direction classification + **volatility forecast** (MAE/RMSE/R²/Directional Accuracy, each vs a baseline) |
| Portfolio construction | `src/portfolio.py` | Conservative / Balanced / Aggressive allocations |
| Portfolio backtest | `src/backtest.py` | **walk-forward out-of-sample** equity curves + realised risk vs equal-weight |
| Risk assessment | `src/risk.py` | volatility, Sharpe, Sortino, max drawdown, Calmar |
| Explainability & anomalies | `src/explain.py` | impurity **+ permutation** importances, per-prediction (SHAP) explanation, vol-spike / extreme-drawdown / unusual-volume anomalies |
| Orchestrator | `src/main.py` | runs everything → `results/intelligence_report.json` + `results/models/*.joblib` |

---

## 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

## 2. Dependency Installation

```bash
pip install -r requirements.txt
```

## 3. Get the data

Download from Kaggle and unzip into `data/`:

> https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data

```
data/
  RELIANCE.csv  TCS.csv  INFY.csv  HDFCBANK.csv  ...   <- per-symbol OHLCV
  stock_metadata.csv                                    <- symbol, company, sector, industry
```

The loader auto-normalises slightly different column names and reads
`stock_metadata.csv` for sector mappings.

## 4. Running the Application

**Interactive prototype:**
```bash
streamlit run app.py
```
(If `data/` is empty, the app launches on synthetic data so it always runs.)

**Headless full pipeline (generates the report JSON + figures):**
```bash
python -m src.main \
    --data-dir data \
    --symbols RELIANCE TCS INFY HDFCBANK ICICIBANK ITC SBIN \
    --horizon 5 --rf 0.06 --out results
```

**Verify install without the dataset:**
```bash
python -m src.main --synthetic
```

## 5. Reproducing Results

- All randomness is seeded (`--seed`, default 42).
- Predictor validation uses a **walk-forward `TimeSeriesSplit`** — no training on
  future data, no shuffling.
- The exact command, args, and library versions are saved into
  `results/intelligence_report.json` under `"config"`.
- Re-run the command in §4 to regenerate every number and figure.

---

## Methodology notes (for graders)

- **Predictor.** Gradient-boosted trees on technical-indicator features, evaluated
  with time-series cross-validation against naive baselines (predict-zero /
  predict-majority / persistence). Stock-direction prediction is intrinsically
  hard; we report honest, baseline-relative performance rather than over-fit
  point estimates. A **volatility forecaster** targets forward realised volatility
  and is benchmarked against a persistence baseline.
- **Portfolio.** Mean–variance optimisation (SLSQP, long-only, per-name caps):
  minimum-variance (Conservative), max-Sharpe tangency (Balanced), return-tilted
  (Aggressive). Each allocation ships with a justification and sector breakdown.
  The investable universe is built with a coverage-aware `clean_universe` so
  staggered listing dates don't silently shrink it.
- **Backtest.** Every recommendation is validated **out-of-sample**: weights are
  re-estimated quarterly on a trailing window only (no lookahead), held with
  drift between rebalances, and compared to an equal-weight benchmark — turning a
  recommendation into evidence.
- **Risk.** Annualised volatility, Sharpe, Sortino, maximum drawdown, Calmar.
- **Explainability.** Impurity **and** permutation importances (the latter
  model-agnostic, measured on held-out targets), plus a per-prediction SHAP
  attribution (impurity fallback if SHAP isn't installed), translate each
  forecast into the indicators that drove it.
- **Reproducibility.** All randomness seeded; final models persisted to
  `results/models/*.joblib` with a feature manifest.

## Repo layout
```
nifty50-investment-intelligence-2/
├── README.md
├── requirements.txt
├── app.py                      <- Streamlit working prototype (Deliverable 1)
├── src/
│   ├── data_loader.py          <- load + split-adjust OHLCV
│   ├── features.py             <- technical indicators
│   ├── eda.py                  <- EDA figures + insights
│   ├── predictor.py            <- return / direction / volatility models
│   ├── portfolio.py            <- mean-variance allocations
│   ├── backtest.py             <- walk-forward out-of-sample backtest
│   ├── risk.py                 <- Sharpe/Sortino/drawdown/Calmar
│   ├── explain.py              <- importances + anomaly detection
│   └── main.py                 <- end-to-end pipeline
├── report/
│   ├── NIFTY50_Investment_Intelligence_Report.pdf   <- Deliverable 2
│   ├── build_report.py         <- builds the report HTML from results
│   └── render_pdf.sh           <- HTML -> PDF (headless Chrome)
└── results/models/             <- Model Files (representative set; rest reproducible)
```
> The pipeline also writes `results/intelligence_report.json` and
> `results/figures/` — these are regenerated by `python -m src.main` and are not
> tracked (only the deliverables live in the repo).

## Regenerating the report
The PDF is built from the pipeline output — no manual editing:
```bash
python -m src.main --data-dir data --out results   # produces results/
python report/build_report.py --results results --data data --out report
bash report/render_pdf.sh                            # -> report/*.pdf
```

## Model files
The full pipeline persists one regressor, classifier and volatility model per
symbol to `results/models/*.joblib` (≈50 MB). A **representative subset** is
committed to keep the repo lean; the complete set is regenerated by re-running
`python -m src.main`. All training is seeded, so the regenerated models are
identical.
