"""
forecast_backtest.py

Backtest SARIMA forecasting on your historical daily_revenue.csv.

Saves:
 - data/outputs/forecast_backtest_results.json   (MAE/RMSE and info)
 - data/outputs/forecast_backtest_comparison.csv (date, actual, forecast)

Usage:
    python src/forecast_backtest.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings

# statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def load_series():
    p = PROC / "daily_revenue.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run preprocessing first.")
    df = pd.read_csv(p, parse_dates=["date"])
    df = df.sort_values("date").set_index("date").asfreq("D")
    # fill small gaps with 0 (or you can forward-fill)
    df["total_revenue"] = df["total_revenue"].fillna(0)
    return df["total_revenue"]

def single_holdout_backtest(y, horizon=30, sarima_order=(1,1,1), seasonal_order=(1,1,1,7)):
    """
    Train on all data except last `horizon` days, forecast horizon days, compute MAE/RMSE.
    Returns forecast series (indexed by date) and metrics dict.
    """
    train = y[:-horizon]
    test = y[-horizon:]
    print(f"Training on {len(train)} days, testing on {len(test)} days ({test.index.min().date()} -> {test.index.max().date()})")

    model = SARIMAX(train, order=sarima_order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=horizon).predicted_mean
    # ensure index aligns with test index
    fc.index = test.index
    mae = float(np.mean(np.abs(test.values - fc.values)))
    rmse = float(np.sqrt(np.mean((test.values - fc.values)**2)))
    return fc, {"mae": mae, "rmse": rmse, "train_size": len(train), "horizon": horizon}

def rolling_backtest(y, horizon=30, n_splits=5, sarima_order=(1,1,1), seasonal_order=(1,1,1,7)):
    """
    Rolling-origin backtest: run n_splits where each split expands training window and forecasts `horizon`.
    Aggregates MAE/RMSE across splits.
    """
    results = []
    total_len = len(y)
    # compute starting points so last split tests the final horizon
    # pick equally spaced train end positions
    min_train = total_len - horizon - (n_splits-1)*horizon
    if min_train < 30:
        raise ValueError("Not enough data for rolling backtest. Reduce n_splits or horizon.")
    train_ends = [min_train + i*horizon for i in range(n_splits)]
    for te in train_ends:
        train = y[:te]
        test = y[te:te+horizon]
        model = SARIMAX(train, order=sarima_order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=horizon).predicted_mean
        fc.index = test.index
        mae = float(np.mean(np.abs(test.values - fc.values)))
        rmse = float(np.sqrt(np.mean((test.values - fc.values)**2)))
        results.append({"train_end": str(train.index[-1].date()), "mae": mae, "rmse": rmse})
        print(f"Split train_end={train.index[-1].date()} mae={mae:.2f} rmse={rmse:.2f}")
    # aggregate
    maes = [r["mae"] for r in results]
    rmses = [r["rmse"] for r in results]
    agg = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
           "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
           "n_splits": len(results)}
    return results, agg

def main():
    y = load_series()
    horizon = 30
    sarima_order = (1,1,1)
    seasonal_order = (1,1,1,7)

    # Single holdout
    fc, metrics = single_holdout_backtest(y, horizon=horizon, sarima_order=sarima_order, seasonal_order=seasonal_order)
    print("Single holdout metrics:", metrics)

    # Save comparison CSV
    comp = pd.DataFrame({"date": fc.index, "actual": y.loc[fc.index].values, "forecast": fc.values})
    comp_file = OUT / f"forecast_backtest_comparison_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv"
    comp.to_csv(comp_file, index=False)
    print("Saved comparison to", comp_file)

    # Rolling backtest (optional; do it if you have enough data)
    try:
        rolling_results, rolling_agg = rolling_backtest(y, horizon=horizon, n_splits=3,
                                                        sarima_order=sarima_order, seasonal_order=seasonal_order)
        print("Rolling aggregate:", rolling_agg)
    except Exception as e:
        rolling_results, rolling_agg = None, {"error": str(e)}
        print("Skipping rolling backtest:", e)

    # Save metrics JSON
    out = {
        "single_holdout": metrics,
        "rolling": {"results": rolling_results, "aggregate": rolling_agg}
    }
    out_file = OUT / f"forecast_backtest_results_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved backtest results to", out_file)

if __name__ == "__main__":
    main()
