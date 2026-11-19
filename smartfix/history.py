"""Q3 - Inspection history stored in SQLite (history.db at the project root)."""
import sqlite3
from datetime import datetime

import pandas as pd

from .config import ROOT

DB_PATH = ROOT / "history.db"


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS inspections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, filename TEXT, verdict TEXT,
        score REAL, defect_pct REAL, n_zones INTEGER, time_ms REAL)""")
    return c


def add(filename: str, r: dict) -> None:
    with _conn() as c:
        c.execute("INSERT INTO inspections (timestamp, filename, verdict, score, "
                  "defect_pct, n_zones, time_ms) VALUES (?,?,?,?,?,?,?)",
                  (datetime.now().isoformat(timespec="seconds"), filename,
                   r["verdict"], r["score"], r["defect_pct"], r["n_zones"], r["time_ms"]))


def load() -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql("SELECT * FROM inspections ORDER BY id DESC", c)


def clear() -> None:
    with _conn() as c:
        c.execute("DELETE FROM inspections")
