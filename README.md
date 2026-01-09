E-Commerce Business Analytics & AI Insights Platform

An end-to-end E-Commerce Analytics platform that transforms raw transactional data into actionable business insights using Data Analysis, Machine Learning, FastAPI, Streamlit, and AI-powered Chatbots.

This project is inspired by real-world e-commerce analytics use cases and demonstrates industry-grade data engineering, analytics, and AI integration.

🚀 Key Features
📈 Business Analytics

Sales KPIs (Revenue, Orders, Customers, AOV)

Daily revenue trends

Revenue forecasting using time-series models

Customer segmentation & churn analysis

🤖 AI-Powered Insights

AI-generated insights for:

Sales performance

Revenue forecasting

Customer churn

Review sentiment

SQL-based AI Business Chatbot that can answer:

Total revenue

Highest / lowest selling products

Monthly & daily sales

Churn & sentiment related queries

💬 AI Business Chatbot

Natural language → SQL → Insight

Auto-detects user intent

Generates SQL queries dynamically

Returns:

Business explanation

Insights & recommendations

Auto-generated charts from query results

🧠 Machine Learning Models

Revenue forecasting (SARIMA)

Customer churn prediction

Review sentiment classification

Product recommendation system

🏗️ Tech Stack
🔹 Backend

FastAPI – Scalable REST API

Pandas / NumPy – Data processing

SQLite / SQL Engine – AI SQL queries

DeepSeek / OpenRouter LLM – AI insights & chatbot

Uvicorn – ASGI server

🔹 Frontend

Streamlit – Interactive dashboard

Custom UI with KPI cards & charts

AI chatbot interface

🔹 Data Science & ML

Pandas, Scikit-learn

Time-series forecasting

NLP sentiment analysis

Recommendation systems

📂 Project Structure
ecom-olist-analytics/
│
├── src/
│   ├── api_app.py          # FastAPI backend
│   ├── ai/
│   │   ├── llm.py          # LLM client (DeepSeek / OpenRouter)
│   │   ├── sql_agent.py    # SQL-based AI agent
│   │   ├── prompt_builder.py
│   │   └── metrics.py      # Business metrics
│
├── frontend/
│   └── app.py              # Streamlit dashboard
│
├── data/
│   └── processed/          # Cleaned datasets
│
├── notebooks/              # EDA & ML notebooks
├── requirements.txt
└── README.md

🧠 AI Business Chatbot – How It Works

User asks a natural language question

LLM converts the question into SQL

SQL runs on business data

Results are:

Explained in business language

Visualized automatically

Enhanced with recommendations

Example queries:

“What is our total revenue?”

“Highest selling product category?”

“Sales in June 2018”

“Which category is underperforming?”

🖥️ How to Run Locally
1️⃣ Clone Repository
git clone https://github.com/USERNAME/ecom-olist-analytics.git
cd ecom-olist-analytics

2️⃣ Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Backend (FastAPI)
uvicorn src.api_app:app --reload


API Docs → http://127.0.0.1:8000/docs

5️⃣ Run Frontend (Streamlit)
streamlit run frontend/app.py