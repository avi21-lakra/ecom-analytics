"""
Item-to-item co-purchase recommender.

Approach:
 - Build product co-occurrence counts from orders (order_items)
 - For each product, rank other products by co-occurrence count (popularity among co-purchases)
 - Output top-N recommendations per product

Output:
 - data/processed/product_recommendations.csv
   columns: product_id, recommended_products (semicolon-separated product_ids)
"""
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import itertools
import csv

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

def run_recommender(top_k=10):
    # prefer processed order items if available, else raw
    cand1 = PROC / "olist_order_items_dataset.csv"
    cand2 = RAW / "olist_order_items_dataset.csv"
    if cand1.exists():
        items_path = cand1
    elif cand2.exists():
        items_path = cand2
    else:
        # fallback: try to extract from merged full orders
        merged = PROC / "olist_full_orders.csv"
        if merged.exists():
            print("ℹ️ using olist_full_orders.csv to derive order items")
            df = pd.read_csv(merged, low_memory=False)
            # ensure product_id and order_id exist
            if 'product_id' not in df.columns or 'order_id' not in df.columns:
                print("❌ Cannot find product_id/order_id in merged file.")
                return
            order_items = df[['order_id','product_id']].dropna().astype(str)
        else:
            print("❌ No order_items file found in data/raw or data/processed, and no merged file present.")
            return
    if 'items_path' in locals():
        print(f"📌 Loading order items from {items_path}")
        order_items = pd.read_csv(items_path, low_memory=False, usecols=['order_id','product_id']).dropna().astype(str)

    # group products per order
    print("📌 Grouping products per order...")
    order_groups = order_items.groupby('order_id')['product_id'].apply(list)

    # compute co-occurrence counts (product -> Counter of other products)
    print("📌 Computing co-occurrence counts (memory-efficient pair counting)...")
    co_counts = defaultdict(Counter)
    for products in order_groups:
        # unique products per order (avoid double counting same product in same order)
        unique_products = list(dict.fromkeys(products))
        # for every ordered pair
        for a, b in itertools.permutations(unique_products, 2):
            co_counts[a][b] += 1

    # build recommendations: for each product, top-k by co-occurrence count
    print("📌 Building top-k recommendations...")
    rows = []
    for product, counter in co_counts.items():
        most = [pid for pid, cnt in counter.most_common(top_k)]
        rows.append((product, ";".join(most)))

    # also include products that never co-occurred (cold-start): recommend top popular products
    print("📌 Computing product popularity fallback...")
    prod_counts = order_items['product_id'].value_counts()
    popular_products = [str(p) for p in prod_counts.index.tolist()]

    known_prods = set(p for p,_ in rows)
    # include missing products
    for p in prod_counts.index.astype(str):
        if p not in known_prods:
            # top-K popular excluding itself
            recs = [q for q in popular_products if q != str(p)][:top_k]
            rows.append((str(p), ";".join(recs)))

    # save to CSV
    out_path = PROC / "product_recommendations.csv"
    print(f"📌 Saving recommendations to {out_path} ...")
    with open(out_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["product_id","recommended_products"])
        for product, recs in rows:
            writer.writerow([product, recs])

    print("✅ Recommendations saved.")

if __name__ == "__main__":
    run_recommender(top_k=10)
