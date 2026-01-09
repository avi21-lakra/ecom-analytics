import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "processed"

def load_csv(name):
    return pd.read_csv(DATA / name)

def total_revenue():
    df = load_csv("olist_full_orders.csv")
    return float(df["revenue"].sum())

def total_orders():
    df = load_csv("olist_full_orders.csv")
    return int(df["order_id"].nunique())

def highest_selling_product():
    df = load_csv("olist_full_orders.csv")
    top = (
        df.groupby("product_category_name")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(1)
    )
    return {
        "product": top.index[0],
        "revenue": float(top.values[0])
    }

def churn_rate(threshold=0.7):
    df = load_csv("customer_churn_predictions.csv")
    return round((df["churn_probability"] >= threshold).mean(), 4)

def sentiment_ratio():
    df = load_csv("review_sentiments.csv")
    return {
        "positive": int((df["sentiment_pred"] == 1).sum()),
        "negative": int((df["sentiment_pred"] == 0).sum())
    }
def highest_sales_month():
    df = load_csv("olist_full_orders.csv")

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"]
    )

    monthly = (
        df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    top_month = monthly.index[0].strftime("%Y-%m")
    top_revenue = float(monthly.iloc[0])

    return {
        "month": top_month,
        "revenue": top_revenue
    }

def compute_metrics():
    return {
        "total_revenue": total_revenue(),
        "total_orders": total_orders(),
        "highest_selling_product": highest_selling_product(),
        "highest_sales_month": highest_sales_month(),   # 👈 ADD THIS
        "churn_rate": churn_rate(),
        "sentiment": sentiment_ratio()
    }

