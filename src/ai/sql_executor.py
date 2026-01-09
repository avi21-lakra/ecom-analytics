import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "database" / "olist.db"

def run_sql(query: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if not query.lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed")

    cursor.execute(query)

    rows = cursor.fetchall()

    columns = [desc[0] for desc in cursor.description]
    conn.close()

    return [dict(zip(columns, row)) for row in rows]
