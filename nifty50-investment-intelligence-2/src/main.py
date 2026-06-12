"""
End-to-end investment-intelligence pipeline (headless).

    python -m src.main --data-dir data --symbols RELIANCE TCS INFY HDFCBANK ICICIBANK \
        --horizon 5 --out results

Use --synthetic to run on generated data (no download needed). The Streamlit
prototype (app.py) reuses these same modules for the interactive UI.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import platform

import numpy as np

from .data_loader import load_market_data, synthetic_prices
from .eda import run_eda
from .predictor import predict_symbol
from .portfolio import construct_portfolios, sector_allocation
from .risk import risk_profile
from .explain import feature_importance, detect_anomalies


def main(argv=None):
    p = argparse.ArgumentParser(description="NIFTY-50 investment intelligence")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--symbols", nargs="*", default=None,
                   help="subset of symbols; default = all in data dir")
    p.add_argument("--horizon", type=int, default=5, help="forecast horizon (days)")
    p.add_argument("--rf", type=float, default=0.06, help="annual risk-free rate")
    p.add_argument("--out", default="results")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    print("[1/5] Loading market data ...")
    market = (synthetic_prices(seed=args.seed) if args.synthetic
              else load_market_data(args.data_dir, symbols=args.symbols))
    symbols = market.symbols if args.symbols is None else [
        s for s in args.symbols if s in market.symbols]
    print(f"      {len(market.symbols)} symbols loaded; analysing {len(symbols)}")

    print("[2/5] EDA ...")
    eda = run_eda(market, args.out)
    print(f"      {eda['n_symbols']} symbols, {eda['n_trading_days']} trading days")

    print("[3/5] Risk profiles ...")
    risk = {}
    for sym in symbols:
        series = market.for_symbol(sym)["close"]
        if len(series) > 252:
            risk[sym] = risk_profile(series, rf=args.rf)

    print("[4/5] Stock predictor engine ...")
    predictions = {}
    importances = {}
    for sym in symbols:
        try:
            res = predict_symbol(market, sym, horizon=args.horizon)
            predictions[sym] = {
                "regression": res["regression"],
                "classification": res["classification"],
            }
            importances[sym] = feature_importance(
                res["models"]["regressor"], res["feature_columns"])
            r = res["regression"]
            print(f"      {sym:12s} RMSE={r['RMSE']:.4f} R2={r['R2']:.3f} "
                  f"DirAcc={r['DirectionalAccuracy']:.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"      {sym}: skipped ({e})")

    print("[5/5] Portfolio construction + anomalies ...")
    portfolios = construct_portfolios(market.close_wide(), rf=args.rf)
    for name, port in portfolios.items():
        port["sector_allocation"] = sector_allocation(port, market)
        print(f"      {name:12s} E[ret]={port['expected_annual_return']:.3f} "
              f"vol={port['expected_annual_volatility']:.3f} "
              f"Sharpe={port['sharpe']:.3f}")

    anomalies = {}
    for sym in symbols[:10]:
        df = market.for_symbol(sym)
        flagged = detect_anomalies(df)
        anomalies[sym] = flagged.head(20).to_dict(orient="records")

    # write everything
    bundle = {
        "eda": eda,
        "risk_profiles": risk,
        "predictions": predictions,
        "feature_importances": importances,
        "portfolios": portfolios,
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
          f"See {args.out}/intelligence_report.json and {args.out}/figures/.")


if __name__ == "__main__":
    main()
