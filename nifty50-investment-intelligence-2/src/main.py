"""
End-to-end investment-intelligence pipeline (headless).

    python -m src.main --data-dir data --symbols RELIANCE TCS INFY HDFCBANK ICICIBANK \
        --horizon 5 --out results

Use --synthetic to run on generated data (no download needed). The Streamlit
prototype (app.py) reuses these same modules for the interactive UI.

Outputs (under --out):
    intelligence_report.json   every metric, importance, portfolio, backtest
    eda_insights.json          EDA summary
    figures/*.png              EDA + backtest charts
    models/*.joblib            persisted final models (Deliverable 3: "Model Files")
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import platform

import numpy as np
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_loader import load_market_data, synthetic_prices
from .eda import run_eda
from .predictor import predict_symbol
from .portfolio import construct_portfolios, sector_allocation
from .backtest import backtest_portfolios
from .risk import risk_profile
from .explain import feature_importance, permutation_feature_importance, detect_anomalies


def _save_backtest_figure(curves: dict, path: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, equity in curves.items():
        ax.plot(equity.index, equity.values, label=name, linewidth=1.3)
    ax.set(title="Out-of-sample equity curves (walk-forward backtest)",
           xlabel="Date", ylabel="Growth of 1 unit")
    ax.legend(fontsize=8)
    ax.axhline(1.0, color="grey", lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="NIFTY-50 investment intelligence")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--symbols", nargs="*", default=None,
                   help="subset of symbols; default = all in data dir")
    p.add_argument("--horizon", type=int, default=5, help="return forecast horizon (days)")
    p.add_argument("--vol-horizon", type=int, default=10, help="volatility forecast horizon")
    p.add_argument("--rf", type=float, default=0.06, help="annual risk-free rate")
    p.add_argument("--out", default="results")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    model_dir = os.path.join(args.out, "models")
    os.makedirs(model_dir, exist_ok=True)
    fig_dir = os.path.join(args.out, "figures")
    t0 = time.time()

    print("[1/6] Loading market data ...")
    market = (synthetic_prices(seed=args.seed) if args.synthetic
              else load_market_data(args.data_dir, symbols=args.symbols))
    symbols = market.symbols if args.symbols is None else [
        s for s in args.symbols if s in market.symbols]
    print(f"      {len(market.symbols)} symbols loaded; analysing {len(symbols)}")

    print("[2/6] EDA ...")
    eda = run_eda(market, args.out)
    print(f"      {eda['n_symbols']} symbols, {eda['n_trading_days']} trading days")

    print("[3/6] Risk profiles ...")
    risk = {}
    for sym in symbols:
        series = market.for_symbol(sym)["close"]
        if len(series) > 252:
            risk[sym] = risk_profile(series, rf=args.rf)

    print("[4/6] Stock predictor engine (return / direction / volatility) ...")
    predictions, importances, perm_importances = {}, {}, {}
    feature_cols = None
    for sym in symbols:
        try:
            res = predict_symbol(market, sym, horizon=args.horizon,
                                 vol_horizon=args.vol_horizon)
            feature_cols = res["feature_columns"]
            predictions[sym] = {
                "regression": res["regression"],
                "classification": res["classification"],
            }
            if "volatility" in res:
                predictions[sym]["volatility"] = res["volatility"]
            importances[sym] = feature_importance(
                res["models"]["regressor"], res["feature_columns"])
            perm_importances[sym] = permutation_feature_importance(
                res["models"]["regressor"], res["_train"]["X"], res["_train"]["y"])
            # persist final models
            for kind, model in res["models"].items():
                joblib.dump(model, os.path.join(model_dir, f"{sym}_{kind}.joblib"))
            r = res["regression"]
            print(f"      {sym:12s} RMSE={r['RMSE']:.4f} R2={r['R2']:.3f} "
                  f"DirAcc={r['DirectionalAccuracy']:.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"      {sym}: skipped ({e})")
    if feature_cols is not None:
        joblib.dump({"feature_columns": feature_cols, "horizon": args.horizon,
                     "vol_horizon": args.vol_horizon},
                    os.path.join(model_dir, "_manifest.joblib"))

    print("[5/6] Portfolio construction (recommendation + walk-forward backtest) ...")
    portfolios = construct_portfolios(market.close_wide(), rf=args.rf)
    for name, port in portfolios.items():
        port["sector_allocation"] = sector_allocation(port, market)
        print(f"      {name:12s} E[ret]={port['expected_annual_return']:.3f} "
              f"vol={port['expected_annual_volatility']:.3f} "
              f"Sharpe={port['sharpe']:.3f}")

    backtest = None
    try:
        backtest = backtest_portfolios(market.close_wide(), rf=args.rf)
        os.makedirs(fig_dir, exist_ok=True)
        _save_backtest_figure(backtest["equity_curves"],
                              os.path.join(fig_dir, "backtest_equity.png"))
        for name, m in backtest["metrics"].items():
            print(f"      [OOS] {name:12s} CAGR={m['annualised_return']:.3f} "
                  f"Sharpe={m['sharpe']:.3f} MaxDD={m['max_drawdown']:.3f}")
        backtest_serialisable = {
            "metrics": backtest["metrics"], "config": backtest["config"]}
    except Exception as e:  # noqa: BLE001
        print(f"      backtest skipped ({e})")
        backtest_serialisable = None

    print("[6/6] Anomaly detection ...")
    anomalies = {}
    for sym in symbols[:10]:
        df = market.for_symbol(sym)
        flagged = detect_anomalies(df)
        anomalies[sym] = flagged.head(20).to_dict(orient="records")

    bundle = {
        "eda": eda,
        "risk_profiles": risk,
        "predictions": predictions,
        "feature_importances": importances,
        "permutation_importances": perm_importances,
        "portfolios": portfolios,
        "backtest": backtest_serialisable,
        "anomalies": anomalies,
        "config": {
            "argv": sys.argv, "args": vars(args),
            "python": platform.python_version(), "numpy": np.__version__,
            "elapsed_sec": round(time.time() - t0, 1),
        },
    }
    with open(os.path.join(args.out, "intelligence_report.json"), "w") as fh:
        json.dump(bundle, fh, indent=2, default=str)

    print(f"\nDone in {bundle['config']['elapsed_sec']}s. "
          f"See {args.out}/intelligence_report.json, {args.out}/figures/, "
          f"and {args.out}/models/.")


if __name__ == "__main__":
    main()
