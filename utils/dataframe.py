"""Tiny DataFrame helpers shared between the exporter and the agent router."""
from __future__ import annotations

import pandas as pd


def strip_report_json(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the heavy ``report_json`` column (if present) before downstream
    serialisation. Returns the same frame unchanged when the column isn't
    there, so callers don't need to branch."""
    return df.drop(columns=["report_json"], errors="ignore")


def df_to_json_safe_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-safe records (NaN → None) without going
    through ``to_json``/``json.loads`` — that round-trip was the dominant
    cost when sending 200+ batch rows back to the UI."""
    safe = strip_report_json(df)
    if safe.empty:
        return []
    return safe.astype(object).where(pd.notna(safe), None).to_dict("records")
