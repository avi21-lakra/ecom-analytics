from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import os
import httpx
from pydantic import BaseModel
from src.ai.intent import detect_intent
from src.ai.metrics import compute_metrics
from src.ai.llm import call_deepseek
from src.ai.router import route_intent
from typing import List, Dict
from src.ai.sql_agent import build_sql_prompt
from src.ai.sql_executor import run_sql






# =====================================================
# APP CONFIG
# =====================================================

app = FastAPI(
    title="E-Commerce Business Analytics API",
    description="""
    Backend API for an end-to-end E-Commerce Analytics project.

    Modules:
    • Sales Analytics
    • Revenue Forecasting
    • Customer Segmentation
    • Churn Prediction
    • Review Sentiment Analysis
    • Product Recommendation System

    Built using FastAPI for scalability and easy integration.
    """,
    version="1.0.0"
)

# Allow frontend tools (Streamlit / Power BI / Browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# =====================================================
# AI HELPER — DeepSeek R1 (via OpenRouter)
# =====================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError("❌ OPENROUTER_API_KEY not set")


async def call_deepseek(prompt: str):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Ecommerce Analytics AI"
    }

    payload = {
        "model": "deepseek/deepseek-r1",   # ✅ FIXED MODEL
        "messages": [
            {
                "role": "system",
                "content": "You are a senior business analytics expert."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.4,
        "max_tokens": 200
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, json=payload)

        # 🔍 DEBUG (optional but helpful)
        if response.status_code != 200:
            print("OpenRouter error:", response.text)
            response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]



# =====================================================
# PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

# =====================================================
# HELPER
# =====================================================

def load_csv(file_name: str) -> pd.DataFrame:
    path = DATA / file_name
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{file_name} not found in processed data"
        )
    return pd.read_csv(path)
def get_product_names():
    df = load_csv("olist_full_orders.csv")
    return df[["product_id", "product_category_name"]].drop_duplicates()



def clean_llm_code(code: str):
    if "```" in code:
        code = code.split("```")[1]
    return code.strip()



# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/", tags=["Health"])
def health():
    return {
        "status": "running",
        "message": "E-Commerce Analytics API is live",
        "backend": "FastAPI"
    }

# =====================================================
# SALES ANALYTICS
# =====================================================

@app.get("/sales/summary", tags=["Sales"])
def sales_summary():
    df = load_csv("olist_full_orders.csv")

    total_revenue = float(df["revenue"].sum())
    total_orders = int(df["order_id"].nunique())
    total_customers = int(df["customer_unique_id"].nunique())
    aov = round(total_revenue / total_orders, 2)

    return {
        "total_revenue": round(total_revenue, 2),
        "total_orders": total_orders,
        "total_customers": total_customers,
        "average_order_value": aov
    }

@app.get("/sales/daily", tags=["Sales"])
def daily_sales():
    df = load_csv("daily_revenue.csv")
    return df.to_dict(orient="records")
# =====================================================
# AI INSIGHTS — SALES
# ==================================================

# =====================================================
# FORECASTING
# =====================================================

@app.get("/forecast/revenue", tags=["Forecasting"])
def revenue_forecast(days: int = 30):
    df = load_csv("revenue_forecast_sarima.csv")
    return df.tail(days).to_dict(orient="records")

# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

@app.get("/customers/segments", tags=["Customers"])
def customer_segments():
    df = load_csv("customer_segments.csv")

    segments = (
        df.groupby("cluster")
        .size()
        .reset_index(name="customers")
        .to_dict(orient="records")
    )

    return {
        "total_customers": int(df.shape[0]),
        "segments": segments
    }

# =====================================================
# CHURN ANALYSIS
# =====================================================

@app.get("/customers/churn/summary", tags=["Customers"])
def churn_summary(threshold: float = 0.7):
    df = load_csv("customer_churn_predictions.csv")

    total_customers = len(df)
    high_risk = df[df["churn_probability"] >= threshold]

    churn_rate = len(high_risk) / total_customers if total_customers > 0 else 0

    return {
        "churn_rate": round(churn_rate, 4),
        "high_risk_customers": int(len(high_risk)),
        "threshold_used": threshold
    }


# =====================================================
# high risk 
# =====================================================


@app.get("/customers/churn/high-risk", tags=["Customers"])
def high_risk_customers(threshold: float = 0.7):
    df = load_csv("customer_churn_predictions.csv")

    if "churn_probability" not in df.columns:
        return []

    risky = df[df["churn_probability"] >= threshold]

    if risky.empty:
        return []

    result = (
        risky[["customer_id", "churn_probability"]]
        .sort_values(by="churn_probability", ascending=False)
        .head(50)
        .reset_index(drop=True)
    )

    return result.to_dict(orient="records")

# =====================================================
# REVIEW SENTIMENT
# =====================================================

@app.get("/reviews/sentiment/summary", tags=["Reviews"])
def sentiment_summary():
    df = load_csv("review_sentiments.csv")

    counts = df["sentiment_pred"].value_counts().to_dict()

    return {
        "total_reviews": int(df.shape[0]),
        "positive_reviews": int(counts.get(1, 0)),
        "negative_reviews": int(counts.get(0, 0))
    }

@app.get("/reviews/negative", tags=["Reviews"])
def negative_reviews(limit: int = 20):
    df = load_csv("review_sentiments.csv")

    neg = df[df["sentiment_pred"] == 0]

    return neg[
        ["review_id", "review_score", "sentiment_prob"]
    ].sort_values(
        by="sentiment_prob",
        ascending=False
    ).head(limit).to_dict(orient="records")

# =====================================================
# PRODUCT RECOMMENDATION
# =====================================================

@app.get("/products/recommendations", tags=["Products"])
def product_recommendations(product_id: str):

    rec_df = load_csv("product_recommendations.csv")
    product_df = load_csv("olist_full_orders.csv")[
        ["product_id", "product_category_name"]
    ].drop_duplicates()

    # 🔧 CLEAN IDS
    rec_df["product_id"] = rec_df["product_id"].astype(str).str.strip()
    product_df["product_id"] = product_df["product_id"].astype(str).str.strip()
    product_id = product_id.strip()

    # 🔍 FIND ROW
    row = rec_df[rec_df["product_id"] == product_id]

    if row.empty:
        return []

    recommended_ids = (
        row.iloc[0]["recommended_products"]
        .split(";")
    )

    result = (
        pd.DataFrame({"product_id": recommended_ids})
        .merge(product_df, on="product_id", how="left")
        .rename(columns={"product_category_name": "product_name"})
    )

    return result.to_dict(orient="records")
# =====================================================
# AI INSIGHTS (RULE + READY FOR LLM)
# =====================================================

@app.post("/ai/insight/sales", tags=["AI"])
async def ai_sales_insight():
    df = load_csv("olist_full_orders.csv")

    total_revenue = df["revenue"].sum()
    total_orders = df["order_id"].nunique()
    aov = total_revenue / total_orders

    prompt = f"""
You are analyzing an e-commerce business.

Metrics:
- Total Revenue: R$ {total_revenue:,.2f}
- Total Orders: {total_orders}
- Average Order Value: R$ {aov:.2f}

Generate a short, professional business insight (2–3 lines).
Mention strengths, risks, and one recommendation.
"""

    insight = await call_deepseek(prompt)
    return {"insight": insight}


@app.post("/ai/insight/forecast", tags=["AI"])
async def ai_forecast_insight():
    df = load_csv("revenue_forecast_sarima.csv").tail(14)

    values = df["forecast_revenue"].tolist()

    prompt = f"""
Revenue forecast values for next days:
{values}

Analyze trend, risk, and provide 1 recommendation.
"""

    insight = await call_deepseek(prompt)
    return {"insight": insight}


@app.post("/ai/insight/churn", tags=["AI"])
async def ai_churn_insight():
    df = load_csv("customer_churn_predictions.csv")
    churn_rate = (df["churn_probability"] >= 0.7).mean()

    prompt = f"""
Customer churn rate is {churn_rate:.2%}.
Analyze churn severity and suggest retention strategies.
"""

    insight = await call_deepseek(prompt)
    return {"insight": insight}


@app.post("/ai/insight/sentiment", tags=["AI"])
async def ai_sentiment_insight():
    df = load_csv("review_sentiments.csv")
    positive_ratio = (df["sentiment_pred"] == 1).mean()

    prompt = f"""
Customer positive sentiment ratio is {positive_ratio:.2%}.
Analyze customer satisfaction and suggest improvements.
"""

    insight = await call_deepseek(prompt)
    return {"insight": insight}



#---ai chat --

class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []


@app.post("/ai/chat", tags=["AI Chatbot"])
async def business_chat(req: ChatRequest):

    # 🔹 FIX 4: Resolve ambiguity
    q = req.question.lower()

    if "highest selling" in q or "top selling" in q:
        req.question += " by revenue"

    if "least selling" in q or "lowest selling" in q:
        req.question += " by revenue"

    # Step 1: Build SQL prompt
    sql_prompt = build_sql_prompt(req.question)

    # Step 2: LLM generates SQL
    sql_query = await call_deepseek(sql_prompt)

    if not sql_query or "select" not in sql_query.lower():
        return {
            "answer": "AI could not generate a valid SQL query. Try rephrasing.",
            "raw_sql": sql_query
        }

    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    # Step 3: Execute SQL
    result = run_sql(sql_query)

    # Step 4: Explain result
    explain_prompt = f"""
You are a senior business analyst.

User question:
{req.question}

SQL result:
{result}

Explain clearly in business language.
Provide insights and recommendations.
"""

    answer = await call_deepseek(explain_prompt)

    return {
        "answer": answer,
        "data": result
    }
