import sqlite3
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
DB_PATH = Path(__file__).parent / "olist.db"

# Connect to SQLite
conn = sqlite3.connect(DB_PATH)

# Load CSVs into SQL tables
pd.read_csv(DATA_DIR / "olist_full_orders.csv").to_sql(
    "orders", conn, if_exists="replace", index=False
)

pd.read_csv(DATA_DIR / "review_sentiments.csv").to_sql(
    "reviews", conn, if_exists="replace", index=False
)

pd.read_csv(DATA_DIR / "customer_churn_predictions.csv").to_sql(
    "churn", conn, if_exists="replace", index=False
)

conn.close()
print("✅ SQLite database created successfully")

