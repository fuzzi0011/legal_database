import sqlite3
from pathlib import Path

DB_PATH = Path("cases.db")

def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        court TEXT,
        title TEXT,
        citation TEXT,
        date TEXT,
        url TEXT UNIQUE,
        full_text TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_cases(cases):
    conn = get_connection()
    cur = conn.cursor()

    for c in cases:
        try:
            cur.execute("""
            INSERT OR IGNORE INTO cases
            (court, title, citation, date, url, full_text)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                c.get("court"),
                c.get("title"),
                c.get("citation"),
                c.get("date"),
                c.get("url"),
                c.get("full_text", "")
            ))
        except Exception:
            pass

    conn.commit()
    conn.close()
