import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

def run_preprocess():
    # load tables (adjust filenames if needed)
    orders = pd.read_csv(RAW / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp","order_delivered_customer_date","order_estimated_delivery_date"], low_memory=False)
    items  = pd.read_csv(RAW / "olist_order_items_dataset.csv", low_memory=False)
    payments = pd.read_csv(RAW / "olist_order_payments_dataset.csv", low_memory=False)
    customers = pd.read_csv(RAW / "olist_customers_dataset.csv", low_memory=False)
    products = pd.read_csv(RAW / "olist_products_dataset.csv", low_memory=False)

    # merge to line-item level
    df = orders.merge(items, on="order_id", how="left") \
               .merge(payments, on="order_id", how="left") \
               .merge(customers, on="customer_id", how="left") \
               .merge(products, on="product_id", how="left")

    # revenue per line
    df["revenue"] = df["price"].fillna(0) + df["freight_value"].fillna(0)

    # delivery and delay days (coerce bad dates)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"], errors="coerce")
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce")
    df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df["delay_days"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days

    # drop rows missing purchase timestamp
    df = df[~df["order_purchase_timestamp"].isna()].copy()

    # save full merged table
    df.to_csv(PROC / "olist_full_orders.csv", index=False)

    # daily aggregates
    daily = df.groupby(df["order_purchase_timestamp"].dt.date).agg(
        total_revenue=("revenue","sum"),
        orders_count=("order_id","nunique"),
        avg_order_value=("revenue","mean")
    ).reset_index().rename(columns={"order_purchase_timestamp":"date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily.to_csv(PROC / "daily_revenue.csv", index=False)

    # customer metrics (RFM-ish)
    cust = df.groupby("customer_id").agg(
    orders_count=("order_id", "nunique"),
    total_spent=("revenue", "sum"),
    last_order=("order_purchase_timestamp", "max")
    ).reset_index()

# Compute max date inside dataset
    max_date = df["order_purchase_timestamp"].max()

# Correct recency calculation (recency = days since last purchase AS OF last dataset date)
    cust["recency_days"] = (max_date - cust["last_order"]).dt.days

    cust.to_csv(PROC / "customer_metrics.csv", index=False)

    print("✅ Preprocessing finished. Files written to data/processed/")

if __name__ == "__main__":
    run_preprocess()
'@ | Out-File -Encoding utf8 src\preprocess.py'