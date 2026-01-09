import httpx
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-r1:free"   # ✅ FREE + STABLE

async def call_deepseek(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Ecommerce Analytics Bot"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a professional business analyst."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4
    }

    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(OPENROUTER_URL, json=payload, headers=headers)

    data = res.json()

    # 🔍 DEBUG SAFETY
    if "choices" not in data or len(data["choices"]) == 0:
        return "⚠️ AI did not return a response."

    content = data["choices"][0]["message"].get("content", "").strip()

    if not content:
        return "⚠️ AI returned empty output. Try rephrasing the question."

    return content
