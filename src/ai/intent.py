def detect_intent(question: str):
    q = question.lower()

    if "total revenue" in q:
        return "total_revenue"

    if "highest" in q and "product" in q:
        return "highest_selling_product"

    if "least" in q:
        return "least_selling_product"

    if "churn" in q:
        return "churn_rate"

    if "sentiment" in q or "review" in q:
        return "sentiment"

    return "general_insight"
