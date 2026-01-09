import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"

def run_segmentation():
    print("📌 Loading customer_metrics.csv...")
    df = pd.read_csv(PROC / "customer_metrics.csv")

    X = df[["orders_count", "total_spent", "recency_days"]].fillna(0)

    print("📌 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("📌 Running KMeans...")
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    df.to_csv(PROC / "customer_segments.csv", index=False)

    print("✅ Customer segments saved!")

if __name__ == "__main__":
    run_segmentation()
