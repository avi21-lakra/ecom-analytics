from fastapi import APIRouter
from pydantic import BaseModel
from src.ai.metrics import compute_metrics

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.post("/ai/chat", tags=["AI Chatbot"])
async def business_chat(req: ChatRequest):
    """
    Business-aware chatbot using real analytics data
    """

    metrics = compute_metrics()
    q = req.question.lower()

    if "total revenue" in q:
        answer = f"Total revenue is R$ {metrics['total_revenue']:,.2f}."

    elif "highest selling" in q or "top product" in q:
        p = metrics["highest_selling_product"]
        answer = (
            f"The highest selling product category is "
            f"'{p['product']}' with revenue R$ {p['revenue']:,.2f}."
        )

    elif "churn" in q:
        answer = f"Current churn rate is {metrics['churn_rate']*100:.2f}%."

    elif "sentiment" in q or "reviews" in q:
        s = metrics["sentiment"]
        answer = (
            f"There are {s['positive']} positive and "
            f"{s['negative']} negative reviews."
        )

    else:
        answer = (
            "I can answer questions about revenue, churn, sentiment, "
            "top products, and customer performance."
        )

    return {
        "question": req.question,
        "answer": answer
    }
