"""
Sentiment model (TF-IDF + Logistic Regression)

Output:
 - data/processed/review_sentiments.csv  (columns: review_id, order_id, review_score, sentiment_label, sentiment_prob)
 
Binary label logic:
 - Positive (1) -> review_score in [4,5]
 - Not positive (0) -> review_score in [1,2,3]
"""
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import re

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

def clean_text(s):
    if pd.isna(s):
        return ""
    # basic cleaning
    s = str(s).lower()
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def run_sentiment():
    reviews_path = RAW / "olist_order_reviews_dataset.csv"
    if not reviews_path.exists():
        print(f"❌ Expected reviews file not found: {reviews_path}")
        return

    print("📌 Loading reviews...")
    reviews = pd.read_csv(reviews_path, low_memory=False)

    # Ensure necessary columns
    if 'review_id' not in reviews.columns or 'review_score' not in reviews.columns:
        print("❌ reviews file missing required columns ('review_id', 'review_score'). Inspect file.")
        return

    # Use review_comment_message if available
    text_col = 'review_comment_message'
    if text_col not in reviews.columns:
        print("⚠️ review_comment_message not found. Creating empty text column.")
        reviews[text_col] = ""

    reviews['text_clean'] = reviews[text_col].astype(str).apply(clean_text)
    # create binary label: positive (4-5) => 1, else 0
    reviews['sentiment_label'] = reviews['review_score'].apply(lambda x: 1 if x in [4,5] else 0)

    # Keep rows with some text (but allow empty — model will still train)
    X = reviews['text_clean'].fillna("")
    y = reviews['sentiment_label']

    # Simple train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("📌 Building TF-IDF + Logistic Regression pipeline...")
    pipe = make_pipeline(
        TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english'),
        LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    )

    print("📌 Training...")
    pipe.fit(X_train, y_train)

    print("📌 Evaluating on test set...")
    probs = pipe.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = None
    print(classification_report(y_test, preds, digits=4))
    print("ROC-AUC:", auc)

    # Predict probabilities on full dataset
    print("📌 Predicting sentiment probabilities for all reviews...")
    reviews['sentiment_prob'] = pipe.predict_proba(reviews['text_clean'].fillna(""))[:,1]
    reviews['sentiment_pred'] = (reviews['sentiment_prob'] >= 0.5).astype(int)

    out_path = PROC / "review_sentiments.csv"
    reviews_out = reviews[['review_id','order_id','review_score','review_comment_message','sentiment_pred','sentiment_prob']]
    reviews_out.to_csv(out_path, index=False)
    print(f"✅ Saved review sentiments to {out_path}")

    # Save model for later use
    model_path = PROC / "sentiment_model_pipeline.joblib"
    joblib.dump(pipe, model_path)
    print(f"✅ Sentiment model pipeline saved to {model_path}")

if __name__ == "__main__":
    run_sentiment()
