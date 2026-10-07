import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent / "data.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS Statement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            submitted_at TEXT NOT NULL,
            status TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS Prediction (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            statement_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (statement_id) REFERENCES Statement(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS Explanation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            word TEXT NOT NULL,
            weight REAL NOT NULL,
            direction TEXT NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES Prediction(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS KnowledgeFact (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            statement_id INTEGER NOT NULL,
            claim TEXT,
            rating TEXT,
            publisher TEXT,
            url TEXT,
            FOREIGN KEY (statement_id) REFERENCES Statement(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS Feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            user_label TEXT NOT NULL,
            comment TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES Prediction(id) ON DELETE CASCADE
        );
        """)

def get_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
