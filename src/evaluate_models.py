# src/evaluate_models.py
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, classification_report,accuracy_score, precision_score, recall_score
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def eval_forecast():
    """
    Prefer backtest comparison CSV (already aligned).
    Fallback order:
      1) data/outputs/forecast_backtest_comparison_*.csv
      2) data/processed/revenue_forecast_sarima.csv (intersection with daily_revenue)
      3) data/outputs/revenue_forecast_sarima.csv (intersection)
    """
    # 1) prefer forecast_backtest_comparison
    comp_files = sorted(OUT.glob("forecast_backtest_comparison_*.csv"))
    if comp_files:
        comp = pd.read_csv(comp_files[-1], parse_dates=["date"]).set_index("date")
        if "actual" in comp.columns and "forecast" in comp.columns:
            y_true = comp["actual"].values
            y_pred = comp["forecast"].values
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            return {"mae": mae, "rmse": rmse, "n_days": int(len(comp)), "source": str(comp_files[-1].resolve())}
        else:
            return {"error": "comparison file found but missing 'actual'/'forecast' columns", "file": str(comp_files[-1].resolve())}

    # helper to try processed / outputs forecast files
    def try_forecast_file(fc_path):
        if not fc_path.exists():
            return {"found": False}
        daily = pd.read_csv(PROC / "daily_revenue.csv", parse_dates=["date"]).set_index("date")
        fc = pd.read_csv(fc_path, parse_dates=["date"]).set_index("date")
        common = daily.index.intersection(fc.index)
        if len(common) >= 7:
            y_true = daily.loc[common, "total_revenue"].values
            y_pred = fc.loc[common, "forecast_revenue"].values if "forecast_revenue" in fc.columns else fc.iloc[:len(common),0].values
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            return {"found": True, "mae": mae, "rmse": rmse, "n_days": int(len(common)), "source": str(fc_path.resolve())}
        else:
            return {"found": True, "error": "No overlapping dates between actuals and this forecast file", "file": str(fc_path.resolve())}

    # 2) try processed
    proc_fc = PROC / "revenue_forecast_sarima.csv"
    res = try_forecast_file(proc_fc)
    if res.get("found"):
        if "mae" in res:
            return {"mae": res["mae"], "rmse": res["rmse"], "n_days": res["n_days"], "source": res["source"]}
        else:
            # continue to fallback
            pass

    # 3) try outputs
    out_fc = OUT / "revenue_forecast_sarima.csv"
    res2 = try_forecast_file(out_fc)
    if res2.get("found") and "mae" in res2:
        return {"mae": res2["mae"], "rmse": res2["rmse"], "n_days": res2["n_days"], "source": res2["source"]}

    # if we reach here, nothing usable found
    return {"error": "No aligned forecast found. Run forecast_backtest.py to generate a comparison CSV for evaluation."}

def eval_churn():
    p = PROC / "customer_churn_predictions.csv"
    if not p.exists():
        return {"error": "customer_churn_predictions.csv missing"}

    df = pd.read_csv(p)

    if "churn" not in df.columns or "churn_probability" not in df.columns:
        return {"error": "columns 'churn' and/or 'churn_probability' missing"}

    # actual and predicted probabilities
    y_true = df["churn"].fillna(0).astype(int)
    y_prob = df["churn_probability"].fillna(0)

    # convert probabilities → predicted label (threshold = 0.5)
    y_pred = (y_prob >= 0.5).astype(int)

    # compute metrics
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = None

    try:
        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
    except Exception:
        accuracy = precision = recall = None

    return {
        "roc_auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "n_customers": int(len(df))
    }


def eval_sentiment():
    p = PROC / "review_sentiments.csv"
    if not p.exists():
        return {"error": "review_sentiments.csv missing"}
    df = pd.read_csv(p)
    if "sentiment_pred" not in df.columns or "sentiment_prob" not in df.columns or "review_score" not in df.columns:
        return {"error": "required columns missing in review_sentiments.csv"}
    df["true_label"] = df["review_score"].apply(lambda x: 1 if x in [4,5] else 0)
    y_true = df["true_label"].fillna(0).astype(int)
    y_prob = df["sentiment_prob"].fillna(0)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = None
    preds = (y_prob >= 0.5).astype(int)
    report = classification_report(y_true, preds, output_dict=True, zero_division=0)
    return {"roc_auc": auc, "report": report, "n_reviews": int(len(df))}

def eval_segmentation():
    p = PROC / "customer_segments.csv"
    if not p.exists():
        return {"error": "customer_segments.csv missing"}
    df = pd.read_csv(p)
    if "cluster" not in df.columns:
        return {"error": "'cluster' column missing"}
    feat_cols = [c for c in ["orders_count","total_spent","recency_days"] if c in df.columns]
    if len(feat_cols) < 2:
        sizes = df["cluster"].value_counts().to_dict()
        return {"silhouette": None, "cluster_sizes": sizes}
    X = df[feat_cols].fillna(0).values
    try:
        sil = float(silhouette_score(X, df["cluster"]))
    except Exception:
        sil = None
    sizes = df["cluster"].value_counts().to_dict()
    return {"silhouette": sil, "cluster_sizes": sizes, "n_customers": int(len(df))}

def main():
    out = {}
    out["forecast"] = eval_forecast()
    out["churn"] = eval_churn()
    out["sentiment"] = eval_sentiment()
    out["segmentation"] = eval_segmentation()
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_file = OUT / f"model_metrics_{ts}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved metrics to", out_file)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
