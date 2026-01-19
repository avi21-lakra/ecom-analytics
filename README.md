E-Commerce Business Analytics & AI Chatbot

An end-to-end data analytics platform that combines
Business Intelligence + Machine Learning + Generative AI
to deliver real-time insights for an e-commerce business.

This project goes beyond dashboards by enabling a natural-language AI chatbot that can answer any business question using live SQL queries and explain results in plain business language.

🚀 Key Features
📈 Business Analytics Dashboard

Total Revenue, Orders, Customers, AOV

Daily Revenue Trend Visualization

Revenue Forecasting (Time-Series)

Customer Churn Analysis

Review Sentiment Analysis

🤖 AI Business Chatbot (Major Highlight)

Ask questions like:

“What is our total revenue?”

“Highest selling product category?”

“Sales in March 2018”

“Lowest selling category by revenue”

“Churn risk summary”

➡️ AI converts your question into SQL, runs it on real data,
then explains results with business insights & recommendations.

📊 Auto-Generated Charts from AI

AI responses automatically generate bar / line charts

No manual coding needed

Works for any SQL result

🧠 AI Architecture (Industry-Grade)
User Question
      ↓
LLM (DeepSeek via OpenRouter)
      ↓
SQL Query Generation
      ↓
Database Execution (SQLite)
      ↓
Result Explanation (LLM)
      ↓
Auto Visualization (Streamlit)

🏗️ Tech Stack
Frontend

Streamlit

Interactive dashboards

AI Chat UI

Auto charts

Backend

FastAPI

REST APIs

AI endpoints

SQL execution engine

Data & ML

Pandas

SQL (SQLite)

Time-Series Forecasting

Churn Prediction

Sentiment Analysis

AI / LLM

DeepSeek (Free LLM)

OpenRouter API

Prompt Engineering

SQL-based AI Agent

Deployment

Backend: Render

Frontend: Streamlit Cloud

Version Control: Git + GitHub

📁 Project Structure
ecom-olist-analytics/
│
├── data/
│   └── processed/
│
├── src/
│   ├── api_app.py          # FastAPI backend
│   └── ai/
│       ├── llm.py          # LLM calls
│       ├── prompt_builder.py
│       ├── sql_runner.py
│
├── frontend/
│   └── app.py              # Streamlit UI
│
├── requirements.txt
├── README.md

⚙️ How to Run Locally
1️⃣ Clone Repository
git clone https://github.com/your-username/ecom-olist-analytics.git
cd ecom-olist-analytics

2️⃣ Create Virtual Environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set Environment Variable
OPENROUTER_API_KEY=your_api_key_here

5️⃣ Run Backend
uvicorn src.api_app:app --reload  

6️⃣ Run Frontend
streamlit run frontend/app.py

🌐 Live Demo

Backend API:https://ecom-analytics-wak7.onrender.com

Dashboard: https://your-streamlit-app.streamlit.app

(Free-tier deployments may take a few seconds to wake up)

📌 Business Value

✔ Converts raw data into decision-ready insights
✔ Removes dependency on analysts for ad-hoc questions
✔ Enables AI-driven decision making
✔ Scalable & production-ready architecture
