"""SQLite history persistence for verification reports."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import PROJECT_ROOT, settings
from services.rule_engine import VerificationReport


SCHEMA_PATH = PROJECT_ROOT / "db" / "schema.sql"


def db_path() -> Path:
    """Return the configured history database path."""
    path = Path(settings.history_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def init_db() -> None:
    """Create history tables when needed."""
    with sqlite3.connect(db_path()) as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def save_history(report: VerificationReport) -> int:
    """Persist a verification report and return the row id."""
    init_db()
    payload = report.to_dict()
    with sqlite3.connect(db_path()) as conn:
        cursor = conn.execute(
            """
            INSERT INTO verification_history
            (created_at, raw_text, title, doi, verdict, score, report_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                report.user_citation.raw,
                report.user_citation.title,
                report.user_citation.doi,
                report.verdict,
                report.overall_score,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        return int(cursor.lastrowid)


def list_history(limit: int = 1000) -> pd.DataFrame:
    """Return recent history rows as a DataFrame."""
    init_db()
    with sqlite3.connect(db_path()) as conn:
        return pd.read_sql_query(
            """
            SELECT id, created_at, title, doi, verdict, score, raw_text, report_json
            FROM verification_history
            ORDER BY created_at DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


def history_summary() -> dict[str, Any]:
    """Return KPI summary for the dashboard.

    Uses ``GROUP BY verdict`` so the SQLite engine does the counting —
    previously this read 5,000 rows into a DataFrame just to call
    ``value_counts``, which was the dominant cost of every chat reply.
    """
    init_db()
    with sqlite3.connect(db_path()) as conn:
        cursor = conn.execute(
            "SELECT verdict, COUNT(*) FROM verification_history GROUP BY verdict"
        )
        counts = {row[0]: int(row[1]) for row in cursor.fetchall()}
    total = sum(counts.values())
    if total == 0:
        return {"total": 0, "real": 0, "suspicious": 0, "fake": 0, "real_rate": 0}
    real = counts.get("REAL", 0)
    suspicious = counts.get("SUSPICIOUS", 0)
    fake = counts.get("FAKE", 0)
    return {
        "total": total,
        "real": real,
        "suspicious": suspicious,
        "fake": fake,
        "real_rate": real / total,
    }


