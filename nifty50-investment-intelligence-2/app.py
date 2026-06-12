"""
Working Prototype — Investment Intelligence Dashboard (Streamlit).

Run with:
    streamlit run app.py

It reuses the same analytical modules as the headless pipeline (src/*), so the
UI and the report come from identical logic. If the real dataset isn't present,
it falls back to synthetic data so the prototype always launches.

Tabs:
  1. Overview & EDA       — price trends, risk/return map
  2. Stock Predictor      — per-symbol forecast metrics + driving signals
  3. Portfolio Builder    — allocations for Conservative/Balanced/Aggressive
  4. Risk & Anomalies     — risk profile + flagged unusual events
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_market_data, synthetic_prices
from src.features import add_all_indicators
from src.predictor import predict_symbol
from src.portfolio import construct_portfolios, sector_allocation
from src.backtest import backtest_portfolios
from src.risk import risk_profile
from src.explain import (
    feature_importance, permutation_feature_importance, explain_forecast,
    detect_anomalies,
)

st.set_page_config(page_title="NIFTY-50 Investment Intelligence", layout="wide")


@st.cache_data(show_spinner=False)
def get_market(data_dir: str):
    if os.path.isdir(data_dir) and any(f.endswith(".csv") for f in os.listdir(data_dir)):
        try:
            return load_market_data(data_dir), "real"
        except Exception:
            pass
    return synthetic_prices(), "synthetic"


def main():
    st.title("📈 NIFTY-50 Investment Intelligence")
    st.caption("Decision-support platform — analysis from historical data only.")

    data_dir = st.sidebar.text_input("Data directory", "data")
    market, source = get_market(data_dir)
    if source == "synthetic":
        st.sidebar.warning("Using SYNTHETIC data (real dataset not found in data dir).")
    symbols = market.symbols
    rf = st.sidebar.slider("Risk-free rate", 0.0, 0.10, 0.06, 0.005)
    horizon = st.sidebar.slider("Forecast horizon (days)", 1, 20, 5)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview & EDA", "Stock Predictor", "Portfolio Builder", "Risk & Anomalies"])

    # ---- Tab 1: EDA ----
    with tab1:
        st.subheader("Normalised price growth")
        close = market.close_wide().dropna(axis=1, how="all")
        norm = close / close.iloc[0]
        st.line_chart(norm)

        st.subheader("Risk / return map")
        rows = []
        for sym in close.columns:
            s = close[sym].dropna()
            if len(s) > 252:
                rp = risk_profile(s, rf)
                rows.append({"symbol": sym, **rp})
        rr = pd.DataFrame(rows)
        if not rr.empty:
            st.scatter_chart(rr, x="annualised_volatility", y="annualised_return")
            st.dataframe(rr, use_container_width=True)

    # ---- Tab 2: Predictor ----
    with tab2:
        sym = st.selectbox("Symbol", symbols, key="pred_sym")
        if st.button("Run forecast", key="run_pred"):
            with st.spinner("Training walk-forward models..."):
                res = predict_symbol(market, sym, horizon=horizon)
            c1, c2, c3, c4 = st.columns(4)
            reg = res["regression"]
            c1.metric("RMSE", f"{reg['RMSE']:.4f}")
            c2.metric("MAE", f"{reg['MAE']:.4f}")
            c3.metric("R²", f"{reg['R2']:.3f}")
            c4.metric("Directional Acc.", f"{reg['DirectionalAccuracy']:.1%}")
            st.caption(
                f"Baseline (predict-zero) MAE = {reg['baseline_MAE_predict_zero']:.4f} · "
                f"direction baseline (majority) = "
                f"{res['classification']['baseline_majority_accuracy']:.1%}")
            if "volatility" in res:
                vol = res["volatility"]
                st.markdown(
                    f"**Volatility forecast ({vol['horizon_days']}d):** "
                    f"MAE {vol['MAE']:.4f} vs persistence baseline "
                    f"{vol['baseline_MAE_persistence']:.4f} (lower = better).")
            st.info(explain_forecast(res["models"]["regressor"], res["feature_columns"]))
            st.caption("Permutation importance (model-agnostic, held-out):")
            perm = permutation_feature_importance(
                res["models"]["regressor"], res["_train"]["X"], res["_train"]["y"])
            st.bar_chart(pd.Series(perm).head(10))

    # ---- Tab 3: Portfolio ----
    with tab3:
        st.subheader("Recommended allocations by investor profile")
        if st.button("Build portfolios", key="build_port"):
            with st.spinner("Optimising..."):
                ports = construct_portfolios(market.close_wide(), rf=rf)
            cols = st.columns(3)
            for col, (name, port) in zip(cols, ports.items()):
                with col:
                    st.markdown(f"### {name}")
                    st.metric("Exp. return", f"{port['expected_annual_return']:.1%}")
                    st.metric("Volatility", f"{port['expected_annual_volatility']:.1%}")
                    st.metric("Sharpe", f"{port['sharpe']:.2f}")
                    st.bar_chart(pd.Series(port["weights"]))
                    st.caption(port["justification"])
                    sec = sector_allocation(port, market)
                    if sec:
                        st.write("**Sector mix:**", sec)

            st.divider()
            st.subheader("Walk-forward backtest (out-of-sample)")
            st.caption("Weights re-estimated quarterly on a trailing window only — "
                       "no lookahead. Benchmarked against equal-weight.")
            with st.spinner("Backtesting..."):
                try:
                    bt = backtest_portfolios(market.close_wide(), rf=rf)
                    curves = pd.DataFrame({k: v for k, v in bt["equity_curves"].items()})
                    st.line_chart(curves)
                    st.dataframe(pd.DataFrame(bt["metrics"]).T, use_container_width=True)
                    cfg = bt["config"]
                    st.caption(f"OOS window {cfg['oos_start']} → {cfg['oos_end']} "
                               f"({cfg['oos_days']} days, {cfg['n_universe']} names).")
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Backtest unavailable: {e}")

    # ---- Tab 4: Risk & anomalies ----
    with tab4:
        sym = st.selectbox("Symbol", symbols, key="risk_sym")
        df = market.for_symbol(sym)
        rp = risk_profile(df["close"], rf)
        cols = st.columns(6)
        for col, (k, v) in zip(cols, rp.items()):
            col.metric(k, f"{v}")
        st.subheader("Detected anomalies")
        anomalies = detect_anomalies(df)
        if anomalies.empty:
            st.success("No significant anomalies flagged.")
        else:
            st.dataframe(anomalies, use_container_width=True)
        st.subheader("Price with indicators")
        ind = add_all_indicators(df)
        st.line_chart(ind.set_index("date")[["close", "sma_20", "sma_50"]])


if __name__ == "__main__":
    main()
