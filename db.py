import sqlite3
from pathlib import Path
from typing import Optional, List

DB_PATH = Path(__file__).parent / "set100_daily.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                date        TEXT PRIMARY KEY,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                prev_close  REAL,
                volume      INTEGER,
                source      TEXT DEFAULT 'yahoo',
                fetched_at  TEXT DEFAULT (datetime('now','localtime'))
            )
        """)
        conn.commit()


def upsert_prices(records: List[dict]):
    with get_conn() as conn:
        conn.executemany("""
            INSERT INTO daily_prices (date, open, high, low, close, prev_close, volume, source)
            VALUES (:date, :open, :high, :low, :close, :prev_close, :volume, :source)
            ON CONFLICT(date) DO UPDATE SET
                open       = excluded.open,
                high       = excluded.high,
                low        = excluded.low,
                close      = excluded.close,
                prev_close = excluded.prev_close,
                volume     = excluded.volume,
                source     = excluded.source,
                fetched_at = datetime('now','localtime')
        """, [_fill_defaults(r) for r in records])
        conn.commit()


def _fill_defaults(r: dict) -> dict:
    return {
        "date":       r["date"],
        "open":       r.get("open"),
        "high":       r.get("high"),
        "low":        r.get("low"),
        "close":      r.get("close"),
        "prev_close": r.get("prev_close"),
        "volume":     r.get("volume", 0),
        "source":     r.get("source", "yahoo"),
    }


def fetch_prices(limit: int = 30) -> List[dict]:
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM daily_prices ORDER BY date DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_price_by_date(date_str: str) -> Optional[dict]:
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM daily_prices WHERE date = ?", (date_str,)
        ).fetchone()
    return dict(row) if row else None


def latest_date() -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()
    return row[0] if row else None


def count() -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()
    return row[0] if row else 0
