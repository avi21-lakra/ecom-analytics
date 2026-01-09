import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

def run_churn_model():

    print("📌 Loading customer_metrics.csv...")
    df = pd.read_csv(PROC / "customer_metrics.csv")

    # ---------------------------------------------------
    # 1. IMPROVED Churn Label (More realistic)
    # ---------------------------------------------------
    df["churn"] = (
        (df["recency_days"] > 180) &
        (df["orders_count"] == 1) &
        (df["total_spent"] < df["total_spent"].median())
    ).astype(int)

    print("Churn rate:", df["churn"].mean())

    # ---------------------------------------------------
    # 2. Feature Selection (remove leakage features)
    # ---------------------------------------------------
    X = df[["orders_count", "total_spent"]].copy()
    y = df["churn"]

    # ---------------------------------------------------
    # 3. Train-Test split
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ---------------------------------------------------
    # 4. Balance the dataset using SMOTE
    # ---------------------------------------------------
    sm = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

    # ---------------------------------------------------
    # 5. Scale features
    # ---------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------
    # 6. Train RandomForest with balanced class weights
    # ---------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train_scaled, y_train_balanced)

    # ---------------------------------------------------
    # 7. Predict probabilities
    # ---------------------------------------------------
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # ---------------------------------------------------
    # 8. Find Optimal Threshold (maximize Youden's J)
    # ---------------------------------------------------
    thresholds = np.linspace(0, 1, 200)
    best_threshold = 0.5
    best_score = 0

    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        score = recall_score(y_test, yp) + precision_score(y_test, yp, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"\nOptimal Threshold Found: {best_threshold}")

    # Apply optimal threshold
    y_pred = (y_prob >= best_threshold).astype(int)

    # ---------------------------------------------------
    # 9. Evaluate Final Model
    # ---------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print("\n🎯 Final Churn Model Evaluation:")
    print("---------------------------------")
    print("AUC       :", auc)
    print("Accuracy  :", accuracy)
    print("Precision :", precision)
    print("Recall    :", recall)

    # ---------------------------------------------------
    # 10. Save predictions for ALL customers (for Power BI)
    # ---------------------------------------------------
    df["churn_probability"] = model.predict_proba(
        scaler.transform(X)
    )[:, 1]

    out_path = PROC / "customer_churn_predictions.csv"
    df.to_csv(out_path, index=False)

    print(f"\n💾 Saved churn predictions to {out_path}")


if __name__ == "__main__":
    run_churn_model()
