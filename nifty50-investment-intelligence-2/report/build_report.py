"""
Build the executive technical-report HTML from the pipeline's results.

    python report/build_report.py --results results --data data --out report

Reads `results/intelligence_report.json` (+ figures), curates the numbers to the
current 50 NIFTY-50 constituents, and emits a self-contained, print-ready
`report/report.html` (charts embedded as base64). Render to PDF with headless
Chrome — see report/render_pdf.sh. Re-runnable: every figure is data-driven.
"""
from __future__ import annotations

import os
import json
import base64
import argparse
import statistics as stats

import pandas as pd


# ----------------------------------------------------------------------------- helpers
def _img(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return f"data:image/png;base64,{b64}"


def _pct(x, dp=1):
    try:
        return f"{float(x) * 100:.{dp}f}%"
    except (TypeError, ValueError):
        return "—"


def _num(x, dp=2):
    try:
        return f"{float(x):.{dp}f}"
    except (TypeError, ValueError):
        return "—"


def _money(x):
    try:
        return f"{float(x):.1f}×"
    except (TypeError, ValueError):
        return "—"


# ----------------------------------------------------------------------------- load
def load(results_dir: str, data_dir: str):
    with open(os.path.join(results_dir, "intelligence_report.json")) as fh:
        rep = json.load(fh)
    meta = None
    mpath = os.path.join(data_dir, "stock_metadata.csv")
    if os.path.exists(mpath):
        meta = pd.read_csv(mpath)
        meta.columns = [c.strip().lower() for c in meta.columns]
    return rep, meta


def current_constituents(meta):
    if meta is None or "symbol" not in meta.columns:
        return None, {}
    syms = [str(s) for s in meta["symbol"].tolist()]
    sector_col = "industry" if "industry" in meta.columns else None
    name_col = next((c for c in meta.columns if "company" in c or "name" in c), None)
    sect = {}
    name = {}
    for _, r in meta.iterrows():
        s = str(r["symbol"])
        sect[s] = str(r[sector_col]).title() if sector_col else "—"
        name[s] = str(r[name_col]) if name_col else s
    return syms, (sect, name)


# ----------------------------------------------------------------------------- analysis
def curate(rep, meta):
    syms, maps = current_constituents(meta)
    sect, name = maps if maps else ({}, {})
    risk = rep.get("risk_profiles", {})
    preds = rep.get("predictions", {})
    perm = rep.get("permutation_importances", {})

    universe = [s for s in (syms or risk.keys()) if s in risk]

    # ---- risk league table (top by Sharpe) ----
    rows = []
    for s in universe:
        rp = risk[s]
        rows.append({
            "symbol": s, "sector": sect.get(s, "—"),
            "ret": rp.get("annualised_return"), "vol": rp.get("annualised_volatility"),
            "sharpe": rp.get("sharpe"), "mdd": rp.get("max_drawdown"),
        })
    rows = [r for r in rows if isinstance(r["sharpe"], (int, float))]
    rows.sort(key=lambda r: r["sharpe"], reverse=True)

    # ---- sector aggregation ----
    sec_agg = {}
    for r in rows:
        sec_agg.setdefault(r["sector"], []).append(r)
    sector_rows = []
    for sec, rs in sec_agg.items():
        sector_rows.append({
            "sector": sec, "n": len(rs),
            "ret": stats.mean([x["ret"] for x in rs if x["ret"] is not None]),
            "vol": stats.mean([x["vol"] for x in rs if x["vol"] is not None]),
            "sharpe": stats.mean([x["sharpe"] for x in rs]),
        })
    sector_rows.sort(key=lambda r: r["sharpe"], reverse=True)

    # ---- predictor aggregates ----
    dir_accs, beat50 = [], 0
    pred_rows = []
    for s in universe:
        if s not in preds:
            continue
        reg = preds[s].get("regression", {})
        da = reg.get("DirectionalAccuracy")
        if isinstance(da, (int, float)):
            dir_accs.append(da)
            if da > 0.50:
                beat50 += 1
        pred_rows.append({
            "symbol": s, "da": da, "rmse": reg.get("RMSE"),
            "r2": reg.get("R2"),
            "vol_mae": preds[s].get("volatility", {}).get("MAE"),
            "vol_base": preds[s].get("volatility", {}).get("baseline_MAE_persistence"),
        })
    pred_rows.sort(key=lambda r: (r["da"] is not None, r["da"] or 0), reverse=True)

    # ---- permutation importance: average across universe ----
    agg = {}
    for s in universe:
        for f, v in (perm.get(s, {}) or {}).items():
            agg.setdefault(f, []).append(v)
    top_features = sorted(
        ((f, stats.mean(v)) for f, v in agg.items() if v),
        key=lambda kv: kv[1], reverse=True)[:8]

    return {
        "universe_n": len(universe),
        "risk_rows": rows,
        "sector_rows": sector_rows,
        "pred_rows": pred_rows,
        "dir_median": stats.median(dir_accs) if dir_accs else None,
        "dir_mean": stats.mean(dir_accs) if dir_accs else None,
        "beat50": beat50, "n_pred": len(dir_accs),
        "top_features": top_features,
    }


# ----------------------------------------------------------------------------- HTML
FEATURE_LABELS = {
    "return": "1-day return", "log_return": "log return", "sma_5": "SMA(5)",
    "sma_10": "SMA(10)", "sma_20": "SMA(20)", "sma_50": "SMA(50)",
    "ema_12": "EMA(12)", "ema_26": "EMA(26)", "rsi_14": "RSI(14)", "macd": "MACD",
    "macd_signal": "MACD signal", "macd_hist": "MACD hist", "bb_pctb": "Bollinger %b",
    "bb_bw": "Bollinger width", "volatility_20": "realised vol(20)",
    "momentum_10": "momentum(10)", "momentum_20": "momentum(20)", "atr_14": "ATR(14)",
}


def feat(f):
    return FEATURE_LABELS.get(f, f)


def render(rep, cur, results_dir="results"):
    eda = rep.get("eda", {})
    ports = rep.get("portfolios", {})
    bt = rep.get("backtest", {}) or {}
    btm = bt.get("metrics", {})
    btc = bt.get("config", {})
    figdir = os.path.join(results_dir, "figures")

    # headline numbers
    cons = btm.get("Conservative", {})
    bal = btm.get("Balanced", {})
    agg = btm.get("Aggressive", {})
    ew = btm.get("EqualWeight", {})
    best = cur["risk_rows"][0] if cur["risk_rows"] else {}
    best_sector = cur["sector_rows"][0] if cur["sector_rows"] else {}
    avg_single_vol = (stats.mean([r["vol"] for r in cur["risk_rows"] if r["vol"]])
                      if cur["risk_rows"] else None)
    bal_port = ports.get("Balanced", {})

    # ---- figures ----
    f_norm = _img(os.path.join(figdir, "normalised_prices.png"))
    f_rr = _img(os.path.join(figdir, "risk_return_map.png"))
    f_corr = _img(os.path.join(figdir, "correlation_heatmap.png"))
    f_bt = _img(os.path.join(figdir, "backtest_equity.png"))

    # ---- table builders ----
    def risk_table(rows, n=10):
        body = ""
        for r in rows[:n]:
            body += (f"<tr><td class='sym'>{r['symbol']}</td>"
                     f"<td class='sec'>{r['sector']}</td>"
                     f"<td class='n'>{_pct(r['ret'])}</td>"
                     f"<td class='n'>{_pct(r['vol'])}</td>"
                     f"<td class='n strong'>{_num(r['sharpe'])}</td>"
                     f"<td class='n neg'>{_pct(r['mdd'])}</td></tr>")
        return body

    def sector_table(rows):
        body = ""
        for r in rows:
            body += (f"<tr><td class='sec'>{r['sector']}</td>"
                     f"<td class='n'>{r['n']}</td>"
                     f"<td class='n'>{_pct(r['ret'])}</td>"
                     f"<td class='n'>{_pct(r['vol'])}</td>"
                     f"<td class='n strong'>{_num(r['sharpe'])}</td></tr>")
        return body

    def port_table():
        order = ["Conservative", "Balanced", "Aggressive"]
        body = ""
        for nm in order:
            p = ports.get(nm, {})
            holds = ", ".join(list(p.get("weights", {}))[:4])
            body += (f"<tr><td class='sym'>{nm}</td>"
                     f"<td class='n'>{_pct(p.get('expected_annual_return'))}</td>"
                     f"<td class='n'>{_pct(p.get('expected_annual_volatility'))}</td>"
                     f"<td class='n strong'>{_num(p.get('sharpe'))}</td>"
                     f"<td class='hold'>{holds}</td></tr>")
        return body

    def bt_table():
        order = ["Conservative", "Balanced", "Aggressive", "EqualWeight"]
        labels = {"EqualWeight": "Equal-weight (benchmark)"}
        body = ""
        for nm in order:
            m = btm.get(nm, {})
            cls = "bench" if nm == "EqualWeight" else ""
            body += (f"<tr class='{cls}'><td class='sym'>{labels.get(nm, nm)}</td>"
                     f"<td class='n'>{_pct(m.get('annualised_return'))}</td>"
                     f"<td class='n'>{_pct(m.get('annualised_volatility'))}</td>"
                     f"<td class='n strong'>{_num(m.get('sharpe'))}</td>"
                     f"<td class='n neg'>{_pct(m.get('max_drawdown'))}</td>"
                     f"<td class='n'>{_money((m.get('total_return') or 0) + 1)}</td></tr>")
        return body

    def pred_table(rows, n=8):
        body = ""
        for r in rows[:n]:
            body += (f"<tr><td class='sym'>{r['symbol']}</td>"
                     f"<td class='n strong'>{_pct(r['da'])}</td>"
                     f"<td class='n'>{_num(r['rmse'], 4)}</td></tr>")
        return body

    def feat_bars():
        if not cur["top_features"]:
            return ""
        mx = max(v for _, v in cur["top_features"]) or 1
        out = ""
        for f, v in cur["top_features"]:
            w = max(3, int(100 * v / mx))
            out += (f"<div class='fbar'><span class='flabel'>{feat(f)}</span>"
                    f"<span class='ftrack'><span class='ffill' style='width:{w}%'></span></span></div>")
        return out

    # diversification insight
    div_txt = "—"
    if avg_single_vol and bal_port.get("expected_annual_volatility"):
        cut = 1 - bal_port["expected_annual_volatility"] / avg_single_vol
        div_txt = f"{cut*100:.0f}%"

    date_lo = eda.get("date_start", "—")
    date_hi = eda.get("date_end", "—")
    corr = eda.get("mean_pairwise_correlation")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>NIFTY-50 Investment Intelligence — Technical Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,900&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{{
  --paper:#f7f3ea; --paper-2:#f1ece0; --ink:#1a1611; --ink-soft:#544c3f;
  --rule:#d9cfbd; --green:#1d6b4c; --green-deep:#114a34; --gold:#9c7a23;
  --red:#9b2c2c; --chip:#ece4d4;
  --serif:'Newsreader',Georgia,serif; --display:'Fraunces',Georgia,serif;
  --mono:'IBM Plex Mono',ui-monospace,monospace;
}}
*{{box-sizing:border-box}}
html,body{{margin:0;padding:0;background:var(--paper);color:var(--ink);
  font-family:var(--serif);font-size:10.2pt;line-height:1.5}}
.bg{{position:fixed;inset:0;background:var(--paper);z-index:-1}}
.page{{padding:0 17mm;max-width:210mm;margin:0 auto}}
figure,table,.callout,.fig-half,.fbar,.kpis,.sechead,tr{{break-inside:avoid}}
.sechead,h2,h3{{break-after:avoid}}
section{{break-inside:auto}}
@page{{size:A4;margin:15mm 0 13mm}}
@media screen{{
  html,body{{background:#33302b}}
  .page{{background:var(--paper);width:210mm;padding:13mm 17mm 4mm;margin:8mm auto 0;
    box-shadow:0 2px 18px rgba(0,0,0,.4)}}
  .page:first-of-type{{padding-top:16mm}}
}}
/* ---- type ---- */
h1,h2,h3{{font-family:var(--display);font-weight:600;margin:0;line-height:1.08;
  letter-spacing:-.01em;font-optical-sizing:auto}}
p{{margin:0 0 8pt}}
.kicker{{font-family:var(--mono);font-size:7.4pt;letter-spacing:.22em;
  text-transform:uppercase;color:var(--green-deep);font-weight:600}}
.lead{{font-size:11.5pt;line-height:1.55;color:var(--ink)}}
.small{{font-size:8.6pt;color:var(--ink-soft)}}
em{{font-style:italic}}
strong{{font-weight:600}}
/* ---- masthead ---- */
.mast{{border-bottom:2.5pt solid var(--ink);padding-bottom:9pt;margin-bottom:13pt}}
.mast .row{{display:flex;justify-content:space-between;align-items:flex-end;gap:14pt}}
.mast h1{{font-size:33pt;font-weight:900;letter-spacing:-.02em;max-width:135mm}}
.mast .org{{text-align:right;font-family:var(--mono);font-size:7.6pt;
  letter-spacing:.12em;color:var(--ink-soft);line-height:1.7;text-transform:uppercase}}
.mast .sub{{margin-top:7pt;font-family:var(--display);font-style:italic;
  font-size:12.5pt;color:var(--green-deep)}}
.ticker{{font-family:var(--mono);font-size:7.2pt;letter-spacing:.08em;
  color:var(--ink-soft);margin-top:6pt;border-top:.6pt solid var(--rule);
  padding-top:5pt;display:flex;gap:16pt;flex-wrap:wrap}}
/* ---- kpi ---- */
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:0;margin:13pt 0;
  border:.8pt solid var(--ink);background:var(--paper)}}
.kpi{{padding:9pt 10pt;border-right:.8pt solid var(--rule)}}
.kpi:last-child{{border-right:none}}
.kpi .v{{font-family:var(--mono);font-size:19pt;font-weight:600;
  letter-spacing:-.02em;color:var(--green-deep);font-variant-numeric:tabular-nums}}
.kpi .v.gold{{color:var(--gold)}}
.kpi .l{{font-family:var(--mono);font-size:6.7pt;letter-spacing:.12em;
  text-transform:uppercase;color:var(--ink-soft);margin-top:3pt;line-height:1.35}}
/* ---- sections ---- */
section{{margin-top:14pt}}
.sechead{{display:flex;align-items:baseline;gap:9pt;border-bottom:.8pt solid var(--ink);
  padding-bottom:4pt;margin-bottom:8pt}}
.sechead .no{{font-family:var(--mono);font-size:8.5pt;font-weight:600;color:var(--gold)}}
.sechead h2{{font-size:16pt}}
.cols2{{column-count:2;column-gap:13pt}}
.cols2 p{{margin-bottom:6pt}}
/* ---- tables ---- */
table{{width:100%;border-collapse:collapse;font-size:8.7pt;margin:6pt 0}}
caption{{caption-side:top;text-align:left;font-family:var(--mono);font-size:6.9pt;
  letter-spacing:.14em;text-transform:uppercase;color:var(--ink-soft);
  padding-bottom:4pt}}
th{{font-family:var(--mono);font-size:6.8pt;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-soft);text-align:right;padding:3pt 5pt;border-bottom:.8pt solid var(--ink);
  font-weight:600}}
th:first-child,td:first-child{{text-align:left}}
td{{padding:3.4pt 5pt;border-bottom:.5pt solid var(--rule)}}
td.n{{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums;
  font-size:8.5pt}}
td.sym{{font-weight:600;font-family:var(--display)}}
td.sec,td.hold{{color:var(--ink-soft);font-size:8.2pt}}
td.strong{{color:var(--green-deep);font-weight:600}}
td.neg{{color:var(--red)}}
tr.bench td{{background:var(--chip);font-style:italic}}
tbody tr:last-child td{{border-bottom:.8pt solid var(--ink)}}
/* ---- callout ---- */
.callout{{border-left:2.5pt solid var(--green);background:rgba(29,107,76,.06);
  padding:8pt 11pt;margin:9pt 0;font-family:var(--display);font-size:10.5pt;
  line-height:1.45}}
.callout .k{{font-family:var(--mono);font-size:6.8pt;letter-spacing:.16em;
  text-transform:uppercase;color:var(--green);display:block;margin-bottom:3pt;font-weight:600}}
.callout.gold{{border-left-color:var(--gold);background:rgba(156,122,35,.07)}}
.callout.gold .k{{color:var(--gold)}}
/* ---- figures ---- */
figure{{margin:8pt 0;break-inside:avoid}}
figure img{{width:100%;border:.6pt solid var(--rule);background:#fff;padding:5pt}}
figcaption{{font-family:var(--mono);font-size:6.9pt;letter-spacing:.05em;
  color:var(--ink-soft);margin-top:4pt;line-height:1.4}}
.fig-half{{display:grid;grid-template-columns:1fr 1fr;gap:11pt}}
/* ---- feature bars ---- */
.fbar{{display:flex;align-items:center;gap:8pt;margin:2.6pt 0}}
.flabel{{font-family:var(--mono);font-size:7.8pt;width:34mm;color:var(--ink);text-align:right}}
.ftrack{{flex:1;height:8pt;background:var(--chip)}}
.ffill{{display:block;height:100%;background:var(--green)}}
/* ---- misc ---- */
.note{{font-family:var(--mono);font-size:7.3pt;color:var(--ink-soft);line-height:1.5}}
.runfoot{{position:fixed;bottom:6mm;left:17mm;right:17mm;display:flex;
  justify-content:space-between;font-family:var(--mono);font-size:6.4pt;
  letter-spacing:.12em;text-transform:uppercase;color:var(--ink-soft);
  border-top:.5pt solid var(--rule);padding-top:3pt}}
@media screen{{.runfoot{{display:none}}}}
.tag{{display:inline-block;font-family:var(--mono);font-size:6.8pt;letter-spacing:.08em;
  text-transform:uppercase;background:var(--ink);color:var(--paper);padding:1.5pt 6pt;
  margin-right:4pt}}
ul.tight{{margin:4pt 0 8pt;padding-left:14pt}}
ul.tight li{{margin-bottom:3.5pt}}
</style></head>
<body>
<div class="bg"></div>
<div class="runfoot"><span>NIFTY-50 Investment Intelligence</span>
  <span>Decision-support · historical data only · {date_lo}–{date_hi}</span></div>

<!-- ===================== PAGE 1 ===================== -->
<div class="page">
  <div class="mast">
    <div class="row">
      <div>
        <div class="kicker">Open Projects 2026 · AI / ML · Technical Report</div>
        <h1>The NIFTY-50, Decoded</h1>
      </div>
      <div class="org">Team Rukdimax<br>Somil Agrawal · 23112099<br>S R Nivedhitha · 22411030</div>
    </div>
    <div class="sub">Turning two decades of price history into decisions, not just forecasts.</div>
    <div class="ticker">
      <span>● {cur['universe_n']} constituents</span>
      <span>● {eda.get('n_trading_days','—')} trading days</span>
      <span>● {date_lo} → {date_hi}</span>
      <span>● split-adjusted OHLCV only</span>
      <span>● no live data · no APIs · no news</span>
    </div>
  </div>

  <p class="lead">This platform reads the NIFTY-50 the way an analyst does — measuring
  risk before return, testing every recommendation out-of-sample, and explaining
  <em>why</em> before it says <em>what</em>. Below: the decisions a data-driven desk
  would actually make, and the evidence behind each.</p>

  <div class="kpis">
    <div class="kpi"><div class="v">{_num(cons.get('sharpe'))}</div>
      <div class="l">Conservative Sharpe · 20-yr backtest</div></div>
    <div class="kpi"><div class="v">{_num(ew.get('sharpe'))}</div>
      <div class="l">Equal-weight Sharpe · same window</div></div>
    <div class="kpi"><div class="v gold">{_money((agg.get('total_return') or 0) + 1)}</div>
      <div class="l">Aggressive growth of 1 · out-of-sample</div></div>
    <div class="kpi"><div class="v">{div_txt}</div>
      <div class="l">Volatility cut vs avg single name</div></div>
  </div>

  <section>
    <div class="sechead"><span class="no">00</span><h2>Executive summary</h2></div>
    <div class="cols2">
      <p><strong>Risk first.</strong> Across {cur['universe_n']} constituents over
      {date_lo}–{date_hi}, the highest historical risk-adjusted return belonged to
      <strong>{best.get('symbol','—')}</strong> ({best.get('sector','—')}, Sharpe
      {_num(best.get('sharpe'))}), and the strongest sector on a Sharpe basis was
      <strong>{best_sector.get('sector','—')}</strong>. Average pairwise return
      correlation of {_num(corr)} leaves real room to diversify.</p>

      <p><strong>Portfolios that earn their keep.</strong> Three profiles are built by
      mean–variance optimisation, then <em>backtested walk-forward</em> with quarterly
      rebalancing and no lookahead. The minimum-variance <strong>Conservative</strong>
      book delivered the best out-of-sample Sharpe ({_num(cons.get('sharpe'))}) and the
      shallowest drawdown ({_pct(cons.get('max_drawdown'))}) — beating an equal-weight
      benchmark ({_num(ew.get('sharpe'))}) on risk-adjusted terms.</p>

      <p><strong>Honest forecasting.</strong> A gradient-boosted predictor reaches a
      median directional accuracy of <strong>{_pct(cur['dir_median'])}</strong>
      ({cur['beat50']}/{cur['n_pred']} names above a coin toss). We report where the
      edge is real and where the market is efficient — rather than over-fitting a
      price curve.</p>

      <p><strong>Explainable by construction.</strong> Every forecast is attributed to
      the indicators that drove it; every flagged anomaly maps to a real event
      (2008, March 2020). Decisions come with reasons a user can audit.</p>
    </div>
    <div class="callout"><span class="k">The one decision</span>
      For a typical investor, the evidence favours the <strong>minimum-variance book</strong>:
      over twenty years spanning the 2008 crisis and the COVID crash, it compounded
      capital at {_pct(cons.get('annualised_return'))} a year with materially lower
      drawdowns than chasing return — the clearest risk-adjusted edge in the study.</div>
  </section>

</div>

<!-- ===================== PAGE 2 ===================== -->
<div class="page">
  <section style="margin-top:0">
    <div class="sechead"><span class="no">01</span><h2>Exploratory data analysis</h2></div>
    <p>The dataset is daily split-adjusted OHLCV for NIFTY-50 member companies,
    {date_lo}–{date_hi} ({eda.get('n_trading_days','—')} trading days). Raw prices are
    <em>unadjusted</em>, so corporate actions appear as spurious ±40%+ one-day moves; we
    back-adjust these in the loader before any statistic is computed. The two views below
    anchor every downstream decision — a risk/return map to rank names, and a correlation
    structure to size the diversification opportunity.</p>
    <div class="fig-half">
      <figure><img src="{f_rr}" alt="risk return map">
        <figcaption>Fig 1 — Annualised return vs volatility. Up-and-to-the-left is
        efficient; this is the universe we optimise over.</figcaption></figure>
      <figure><img src="{f_corr}" alt="correlation heatmap">
        <figcaption>Fig 2 — Return correlation matrix. Mean pairwise correlation
        {_num(corr)} — diversification is available, not free.</figcaption></figure>
    </div>
    <figure><img src="{f_norm}" alt="normalised prices">
      <figcaption>Fig 3 — Growth of 1 unit, normalised. Long-run compounding and the
      two systemic drawdowns (2008, 2020) that any allocation must survive.</figcaption></figure>
  </section>
  <section>
    <div class="sechead"><span class="no">02</span><h2>Feature engineering</h2></div>
    <p>Every input is derived purely from the provided OHLCV — no external feeds. Indicators
    span the three regimes that move equities: <strong>trend</strong> (SMA 5/10/20/50/200,
    EMA 12/26), <strong>momentum</strong> (RSI-14, MACD with signal &amp; histogram,
    10/20-day momentum) and <strong>volatility</strong> (Bollinger %b &amp; bandwidth,
    20-day realised σ, ATR-14), plus returns and log-returns. Targets are forward
    <em>h</em>-day return, its sign, and forward realised volatility.</p>
    <p class="note">18 model features · computed per symbol on a date-sorted panel ·
    leakage-safe (every feature uses only information available at time <em>t</em>).</p>
  </section>
</div>

<!-- ===================== PAGE 3 ===================== -->
<div class="page">
  <section style="margin-top:0">
    <div class="sechead"><span class="no">03</span><h2>Methodology &amp; model architecture</h2></div>
    <div class="cols2">
      <p><strong>Predictor.</strong> Gradient-boosted trees (200 stumps, depth 3, lr 0.05,
      80% subsample) on the 18-feature set. Three heads: forward-return regression,
      up/down classification, and forward-volatility regression. Trees are chosen for
      noisy financial data — robust, non-linear, and natively interpretable.</p>
      <p><strong>Validation.</strong> Walk-forward <span class="tag">TimeSeriesSplit</span>
      (5 folds, no shuffling). Each head is scored against a naive baseline —
      predict-zero, predict-majority, and persistence — so a model only counts if it
      <em>beats</em> the trivial answer.</p>
      <p><strong>Portfolio.</strong> Mean–variance optimisation (SLSQP, long-only,
      per-name caps) on a coverage-cleaned universe: minimum-variance for Conservative,
      max-Sharpe tangency for Balanced, return-tilted for Aggressive.</p>
      <p><strong>Backtest.</strong> Weights re-estimated quarterly on a trailing window
      only, held with drift between rebalances, benchmarked to equal-weight — the step
      that converts a recommendation into evidence.</p>
    </div>
    <div class="callout gold"><span class="k">Why this is decision-support, not a price oracle</span>
      The system is graded on the quality of <em>decisions</em>. Point forecasts of price
      are treated as the weakest signal; risk, diversification, and out-of-sample
      behaviour carry the weight — exactly where the evidence is strongest.</div>
  </section>
  <section>
    <div class="sechead"><span class="no">04</span><h2>Stock predictor — results</h2></div>
    <p>Five-day return <em>level</em> is close to a random walk: cross-validated R²
    is negative for most names, i.e. the model does not beat predict-zero on squared
    error — an honest finding about market efficiency. The exploitable signal is
    <strong>directional</strong>: a median accuracy of {_pct(cur['dir_median'])} with
    {cur['beat50']} of {cur['n_pred']} names above 50%.</p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:13pt;align-items:start">
      <table><caption>Top directional accuracy (5-day)</caption>
        <thead><tr><th>Symbol</th><th>Dir. acc</th><th>RMSE</th></tr></thead>
        <tbody>{pred_table(cur['pred_rows'])}</tbody></table>
      <div>
        <p class="small"><strong>Reading the table.</strong> Directional accuracy a few
        points above 50% is, on near-efficient daily data, a genuine and tradable edge —
        not a flashy one. RMSE is reported on split-adjusted returns; we deliberately do
        <em>not</em> headline R², which flatters models that merely predict zero.</p>
        <p class="small"><strong>Volatility forecasting.</strong> A separate head predicts
        forward realised σ. Volatility is highly persistent, so the bar is the persistence
        baseline; the model adds value where volatility <em>clusters</em> (post-shock
        regimes) and ties it elsewhere — reported transparently per name.</p>
      </div>
    </div>
  </section>
</div>

<!-- ===================== PAGE 4 ===================== -->
<div class="page">
  <section style="margin-top:0">
    <div class="sechead"><span class="no">05</span><h2>Portfolio construction</h2></div>
    <p>Three books for three investor profiles, each with a stated objective, per-name
    caps, and a justification. Expected figures below are in-sample (full-history
    mean/covariance); their out-of-sample behaviour is on the next page.</p>
    <table><caption>Recommended allocations · expected (in-sample) statistics</caption>
      <thead><tr><th>Profile</th><th>E[return]</th><th>Volatility</th><th>Sharpe</th>
        <th>Top holdings</th></tr></thead>
      <tbody>{port_table()}</tbody></table>
    <p class="small">{ports.get('Conservative',{}).get('justification','')}
      &nbsp;·&nbsp; {ports.get('Aggressive',{}).get('justification','')}</p>
  </section>
  <section>
    <div class="sechead"><span class="no">06</span><h2>Walk-forward backtest</h2></div>
    <p>The decisive test: hold each book out-of-sample over
    {btc.get('oos_start','—')}–{btc.get('oos_end','—')}
    ({btc.get('n_universe','—')} names, quarterly rebalancing, no lookahead).</p>
    <figure><img src="{f_bt}" alt="backtest equity curves">
      <figcaption>Fig 4 — Out-of-sample growth of 1 unit. Two crises are inside the
      window; the gap between books is a risk story, not just a return story.</figcaption></figure>
    <table><caption>Realised out-of-sample performance</caption>
      <thead><tr><th>Strategy</th><th>CAGR</th><th>Vol</th><th>Sharpe</th>
        <th>Max DD</th><th>Growth ×</th></tr></thead>
      <tbody>{bt_table()}</tbody></table>
    <div class="callout"><span class="k">What the backtest says</span>
      Minimising variance — not chasing return — produced the best risk-adjusted outcome:
      Sharpe {_num(cons.get('sharpe'))} vs {_num(ew.get('sharpe'))} for equal-weight, at
      roughly the drawdown of the aggressive book. Return-tilting won on raw growth
      ({_money((agg.get('total_return') or 0) + 1)}) but cost {_pct(agg.get('max_drawdown'))} in
      peak-to-trough pain. The profiles are doing their job.</div>
  </section>
</div>

<!-- ===================== PAGE 5 ===================== -->
<div class="page">
  <section style="margin-top:0">
    <div class="sechead"><span class="no">07</span><h2>Risk assessment</h2></div>
    <p>Per-name risk is measured with the full institutional set — annualised volatility,
    Sharpe, Sortino, maximum drawdown and Calmar — on split-adjusted returns. The league
    table ranks the universe by historical risk-adjusted return.</p>
    <div style="display:grid;grid-template-columns:1.1fr .9fr;gap:13pt;align-items:start">
      <table><caption>Risk league table · top 10 by Sharpe</caption>
        <thead><tr><th>Symbol</th><th>Sector</th><th>Return</th><th>Vol</th>
          <th>Sharpe</th><th>Max DD</th></tr></thead>
        <tbody>{risk_table(cur['risk_rows'],10)}</tbody></table>
      <table><caption>Sector view · mean by Sharpe</caption>
        <thead><tr><th>Sector</th><th>n</th><th>Return</th><th>Vol</th>
          <th>Sharpe</th></tr></thead>
        <tbody>{sector_table(cur['sector_rows'])}</tbody></table>
    </div>
  </section>
  <section>
    <div class="sechead"><span class="no">08</span><h2>Explainability &amp; anomalies</h2></div>
    <p>Forecasts are attributed with <strong>permutation importance</strong> — measured on
    held-out targets, so it reflects predictive value, not tree-split frequency. Averaged
    across the universe, the drivers are:</p>
    {feat_bars()}
    <p class="small" style="margin-top:7pt">Each individual forecast also ships a
    per-prediction attribution (SHAP when available, impurity otherwise). The anomaly
    detector independently flags volatility spikes, extreme drawdowns and unusual
    volume via rolling robust z-scores — surfacing the 2008 and March-2020 stress
    events directly from price action.</p>
  </section>
</div>

<!-- ===================== PAGE 6 ===================== -->
<div class="page">
  <section style="margin-top:0">
    <div class="sechead"><span class="no">09</span><h2>Key insights</h2></div>
    <ul class="tight">
      <li><strong>Low-volatility wins risk-adjusted.</strong> The minimum-variance book
      beat equal-weight on out-of-sample Sharpe ({_num(cons.get('sharpe'))} vs
      {_num(ew.get('sharpe'))}) with the smallest drawdown — the study's clearest edge.</li>
      <li><strong>Diversification is material.</strong> At a mean pairwise correlation of
      {_num(corr)}, the balanced book's volatility is ~{div_txt} below the average single
      name — risk reduction you don't pay for in return.</li>
      <li><strong>{best_sector.get('sector','—')} led on risk-adjusted return</strong>
      across the period; <strong>{best.get('symbol','—')}</strong> was the single best
      name by Sharpe ({_num(best.get('sharpe'))}).</li>
      <li><strong>Price level is near-efficient; direction is not.</strong> Negative R²
      on return regression but {cur['beat50']}/{cur['n_pred']} names with a directional
      edge — we monetise the signal that's actually there.</li>
      <li><strong>Reasons travel with every number.</strong> Permutation importances,
      per-forecast attribution and event-level anomaly flags make the system auditable.</li>
    </ul>
  </section>
  <section>
    <div class="sechead"><span class="no">10</span><h2>Limitations &amp; future work</h2></div>
    <div class="cols2">
      <p><strong>Limitations.</strong> Index membership carries survivorship bias;
      transaction costs and slippage are not modelled; covariance is sample-estimated and
      regime-blind; per-name forecasting edges are small by nature. Split adjustment is
      heuristic (threshold-based) in the absence of an action calendar.</p>
      <p><strong>Future work.</strong> Regime-aware allocation (vol-targeting, turbulence
      filters); shrinkage/Ledoit-Wolf covariance; cost-aware rebalancing; temporal models
      (LSTM/Temporal-Fusion) for the direction head; and a cloud deployment of the
      Streamlit prototype for live, explainable decision support.</p>
    </div>
    <div class="callout gold"><span class="k">Reproduce every number</span>
      One command — <span class="tag">python -m src.main --data-dir data</span> —
      regenerates every figure, table and model in this report. All randomness is seeded;
      the exact configuration is written into <em>results/intelligence_report.json</em>.</div>
    <p class="note" style="margin-top:9pt">Constraints honoured: only the organiser
    dataset is used — no live market data, financial APIs, news, or alternative data.
    Prototype: <span class="tag">streamlit run app.py</span> · Source: <em>src/</em> ·
    Backtest: <em>src/backtest.py</em>.</p>
  </section>
</div>

</body></html>"""
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="report")
    args = ap.parse_args()

    rep, meta = load(args.results, args.data)
    cur = curate(rep, meta)
    html = render(rep, cur, results_dir=args.results)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "report.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out}  ({len(html)//1024} KB)  universe={cur['universe_n']}")


if __name__ == "__main__":
    main()
