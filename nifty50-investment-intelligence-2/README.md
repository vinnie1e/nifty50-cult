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
| Technical report | `report/REPORT_TEMPLATE.md` (fill with real results → export PDF) |
| Source code | `src/` |
| README | this file |

## What it does

| Module | File | Output |
|--------|------|--------|
| Data loading & normalisation | `src/data_loader.py` | tidy price panel + metadata |
| Technical indicators | `src/features.py` | SMA, EMA, RSI, MACD, Bollinger, ATR, momentum, volatility |
| EDA | `src/eda.py` | risk/return map, correlation heatmap, trends |
| Stock predictor engine | `src/predictor.py` | return regression + direction classification (MAE/RMSE/R²/Directional Accuracy) |
| Portfolio construction | `src/portfolio.py` | Conservative / Balanced / Aggressive allocations |
| Risk assessment | `src/risk.py` | volatility, Sharpe, Sortino, max drawdown, Calmar |
| Explainability & anomalies | `src/explain.py` | feature importances + flagged unusual events |
| Orchestrator | `src/main.py` | runs everything → `results/intelligence_report.json` |

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
  predict-majority). Stock-direction prediction is intrinsically hard; we report
  honest, baseline-relative performance rather than over-fit point estimates.
- **Portfolio.** Mean–variance optimisation (SLSQP, long-only, per-name caps):
  minimum-variance (Conservative), max-Sharpe tangency (Balanced), return-tilted
  (Aggressive). Each allocation ships with a justification and sector breakdown.
- **Risk.** Annualised volatility, Sharpe, Sortino, maximum drawdown, Calmar.
- **Explainability.** Feature importances translate each forecast into the
  indicators that drove it.

## Repo layout
```
nifty50-investment-intelligence/
├── README.md
├── requirements.txt
├── app.py                      <- Streamlit working prototype (Deliverable 1)
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── eda.py
│   ├── predictor.py
│   ├── portfolio.py
│   ├── risk.py
│   ├── explain.py
│   └── main.py
├── report/REPORT_TEMPLATE.md   <- 12-page technical report scaffold
└── results/                    <- generated artifacts
```

## A note on report numbers
The report template has `‹…›` placeholders for every data-dependent figure. Run
the pipeline on the real dataset, then pull values from
`results/intelligence_report.json` and figures from `results/figures/`.
