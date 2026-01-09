import streamlit as st
import requests
import pandas as pd


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =====================================================
# CONFIG
# =====================================================
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="E-Commerce Business Analytics",
    layout="wide"
)

# =====================================================
# STYLES (SUBTLE + VIBRANT)
# =====================================================
st.markdown("""
<style>
.kpi-card {
    background: linear-gradient(135deg, #ffffff, #eef2ff);
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.08);
    border-left: 6px solid #6366f1;
    text-align: center;
}
.kpi-title {
    font-size: 14px;
    color: #6b7280;
    font-weight: 600;
}
.kpi-value {
    font-size: 28px;
    color: #4f46e5;
    font-weight: 800;
}
.insight-box {
    background-color: #ecfeff;
    border-left: 6px solid #06b6d4;
    padding: 16px 20px;
    border-radius: 12px;
    margin-top: 20px;
    color: #0f172a;
    font-size: 15px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# AI INSIGHT HELPER (AUTO, CACHED)
# =====================================================
@st.cache_data(ttl=300)
def get_ai_insight(endpoint: str):
    response = requests.post(f"{API_URL}{endpoint}")

    if response.status_code != 200:
        return "AI insight is currently unavailable."

    data = response.json()
    return data.get("insight", "No insight generated.")


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("Dashboard Menu")

page = st.sidebar.radio(
    "Select Section",
    [
        "Sales Overview",
        "Revenue Forecast",
        "Customer Churn Analysis",
        "Sentiment & Recommendations",
        "AI Business Chatbot"

    ]
)

# =====================================================
# SALES OVERVIEW
# =====================================================
if page == "Sales Overview":
    st.title("Sales Overview")

    data = requests.get(f"{API_URL}/sales/summary").json()

    st.markdown("### Key Business KPIs")
    c1, c2, c3, c4 = st.columns(4)

    kpis = [
        (" Total Revenue", f"R$ {data['total_revenue']:,}"),
        (" Total Orders", data["total_orders"]),
        (" Customers", data["total_customers"]),
        (" Avg Order Value", f"R$ {data['average_order_value']}")
    ]

    for col, (title, value) in zip([c1, c2, c3, c4], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---------- AUTO BUSINESS INSIGHT ----------
    revenue = data["total_revenue"]
    aov = data["average_order_value"]

    if revenue > 1_000_000 and aov > 150:
        insight = " Strong revenue and high AOV indicate healthy customer spending."
    elif revenue > 1_000_000:
        insight = " Revenue is high; focus on increasing order value through bundling."
    else:
        insight = " Revenue is moderate; marketing optimization is recommended."

    st.markdown(
        f"<div class='insight-box'> <b>Business Insight:</b> {insight}</div>",
        unsafe_allow_html=True
    )

    # ---------- AI INSIGHT ----------
    st.markdown("###  AI Sales Insight")
    with st.spinner("Analyzing sales performance..."):
        st.info(get_ai_insight("/ai/insight/sales"))

    # ---------- TREND ----------
    st.markdown("---")
    st.subheader(" Daily Revenue Trend")
    df = pd.DataFrame(requests.get(f"{API_URL}/sales/daily").json())
    df["date"] = pd.to_datetime(df["date"])
    st.line_chart(df.set_index("date")["total_revenue"])

# =====================================================
# REVENUE FORECAST
# =====================================================
elif page == "Revenue Forecast":
    st.title(" Revenue Forecast")

    days = st.slider("Forecast horizon (days)", 7, 60, 30)
    df = pd.DataFrame(
        requests.get(f"{API_URL}/forecast/revenue", params={"days": days}).json()
    )
    df["date"] = pd.to_datetime(df["date"])
    st.line_chart(df.set_index("date")["forecast_revenue"])

    
# =====================================================
# CUSTOMER CHURN (STEP 4)
# =====================================================
elif page == "Customer Churn Analysis":
    st.title(" Customer Churn Analysis")

    threshold = st.slider("Select churn risk threshold", 0.5, 0.9, 0.7, 0.05)

    summary = requests.get(
        f"{API_URL}/customers/churn/summary",
        params={"threshold": threshold}
    ).json()

    risky = requests.get(
        f"{API_URL}/customers/churn/high-risk",
        params={"threshold": threshold}
    ).json()

    tab1, tab2 = st.tabs([" Overview", "🚨 High-Risk Customers"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        metrics = [
            (" Churn Rate", summary["churn_rate"]),
            (" High Risk Customers", summary["high_risk_customers"]),
            (" Threshold", summary["threshold_used"])
        ]

        for col, (title, value) in zip([c1, c2, c3], metrics):
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)

        if summary["churn_rate"] > 0.4:
            st.error("⚠ High churn risk detected!")
        else:
            st.success("✅ Churn under control")

        
    with tab2:
        if risky:
            st.dataframe(pd.DataFrame(risky), use_container_width=True)
        else:
            st.info("No high-risk customers at this threshold.")

# =====================================================
# SENTIMENT & RECOMMENDATION (STEP 5 & 6)
# =====================================================
elif page == "Sentiment & Recommendations":

    st.markdown("<h1> Sentiment & Recommendation</h1>", unsafe_allow_html=True)

    sentiment = requests.get(f"{API_URL}/reviews/sentiment/summary").json()

    c1, c2, c3 = st.columns(3)
    sentiments = [
        (" Total Reviews", sentiment["total_reviews"]),
        (" Positive Reviews", sentiment["positive_reviews"]),
        (" Negative Reviews", sentiment["negative_reviews"])
    ]

    for col, (title, value) in zip([c1, c2, c3], sentiments):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("###  AI Sentiment Insight")
    with st.spinner("Analyzing customer sentiment..."):
        st.info(get_ai_insight("/ai/insight/sentiment"))

    st.markdown("---")
    st.markdown("###  Product Recommendation System")

    product_id = st.text_input(
        "🔎 Enter Product ID",
        placeholder="Example: 90aaf7993470e411207dabf5b998b6"
    )

    if product_id:
        with st.spinner("Finding similar products..."):
            rec = requests.get(
                f"{API_URL}/products/recommendations",
                params={"product_id": product_id}
            ).json()

        if rec:
            st.success("✅ Recommended Products")
            st.dataframe(pd.DataFrame(rec), use_container_width=True)
        else:
            st.warning(" No similar products found.")

# =====================================================
# AI Business Chatbot 
# =====================================================


elif page == "AI Business Chatbot":

    st.title(" AI Business Chatbot")
  #  st.caption("Ask anything about revenue, sales, churn, sentiment, products")

    # --- Session state ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    def normalize_question(q: str) -> str:
        q = q.lower().strip()

        if "highest selling product" in q or "highest sales" in q:
            return "highest selling category by revenue"

        if "lowest selling product" in q or "lowest sales" in q:
            return "lowest selling category by revenue"

        if "total revenue" in q:
            return "total revenue"

        if "sales in" in q:
            return q.replace("sales in", "total revenue in")

        return q

    # --- Input box ---
    user_input = st.text_input("Ask a business question")

    if st.button("Ask AI") and user_input.strip():
        # Save user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        # Call backend
        with st.spinner("Thinking..."):

            clean_question = normalize_question(user_input)

            res = requests.post(
                f"{API_URL}/ai/chat",
                json={
                    "question": clean_question,
                    "history": []
                }

            )



            if res.status_code == 200:
               response_json = res.json()
               answer = response_json.get("answer", "")
               data = response_json.get("data", [])

            else:
                answer = " AI service unavailable."

        # Save AI response (WITH DATA)
        st.session_state.chat_history.append({
            "role": "ai",
            "content": answer,
            "data": data   # 👈 THIS IS IMPORTANT
        })

    def render_auto_chart(data):
        if not data or not isinstance(data, list):
            return

        df = pd.DataFrame(data)

        if df.empty or df.shape[1] < 2:
            return

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        category_cols = df.select_dtypes(exclude="number").columns.tolist()

        if not numeric_cols or not category_cols:
            return

        x_col = category_cols[0]
        y_col = numeric_cols[0]

        st.markdown("### 📊 Data Visualization")

        # Heuristic: choose chart type
        if len(df) <= 5:
            st.bar_chart(df.set_index(x_col)[y_col])
        else:
            st.line_chart(df.set_index(x_col)[y_col])

    
    # --- Display chat ---
st.markdown("---")

for msg in st.session_state.chat_history:

    # USER MESSAGE
    if msg["role"] == "user":
        st.markdown(f" **You:** {msg['content']}")

    # AI MESSAGE
    elif msg["role"] == "ai":
        st.markdown(
            f"""
            <div style="
                background:#ecfeff;
                padding:15px;
                border-radius:10px;
                margin-bottom:10px;
                border-left:5px solid #06b6d4;
            ">
             <b>AI:</b><br>{msg['content']}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ✅ Render chart ONLY for AI messages
        if "data" in msg and msg["data"]:
            render_auto_chart(msg["data"])
