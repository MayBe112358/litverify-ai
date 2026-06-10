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


# --------------------------------------------------------------------- #
# Chat session persistence — the fix for "conversations vanish on reload".
# Messages are already JSON-safe by construction: user turns store file
# *metadata* only (never bytes) and assistant payloads go through
# ``df_to_json_safe_records`` / ``fig.to_json``.
# --------------------------------------------------------------------- #
def save_chat_session(session: dict[str, Any]) -> None:
    """Insert or update one chat session (write-through on every message)."""
    init_db()
    with sqlite3.connect(db_path()) as conn:
        conn.execute(
            """
            INSERT INTO chat_sessions (id, title, created_at, updated_at, messages_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                updated_at = excluded.updated_at,
                messages_json = excluded.messages_json
            """,
            (
                session["id"],
                session.get("title") or "新对话",
                session.get("created_at") or datetime.now().isoformat(timespec="seconds"),
                datetime.now().isoformat(timespec="seconds"),
                json.dumps(session.get("messages") or [], ensure_ascii=False),
            ),
        )


def load_chat_sessions(limit: int = 30) -> list[dict[str, Any]]:
    """Return persisted sessions, most recently updated first."""
    init_db()
    with sqlite3.connect(db_path()) as conn:
        rows = conn.execute(
            """
            SELECT id, title, created_at, messages_json
            FROM chat_sessions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    sessions: list[dict[str, Any]] = []
    for sid, title, created_at, messages_json in rows:
        try:
            messages = json.loads(messages_json)
        except (TypeError, ValueError):
            messages = []
        sessions.append(
            {"id": sid, "title": title, "created_at": created_at, "messages": messages}
        )
    return sessions


def delete_chat_session(sid: str) -> None:
    """Remove one persisted session (the sidebar × button)."""
    init_db()
    with sqlite3.connect(db_path()) as conn:
        conn.execute("DELETE FROM chat_sessions WHERE id = ?", (sid,))


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


