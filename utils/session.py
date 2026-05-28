"""Session state for the agent shell — supports multiple chat sessions.

Shape (everything lives under ``st.session_state``)::

    sessions:           dict[str, dict]    # id -> session record
    active_session_id:  str | None         # currently selected session
    pending_mode:       str                # tool chip selected for next send

    # Legacy single-shot keys still respected by services:
    deepseek_api_key, deepseek_chat_model, deepseek_vl_model,
    crossref_enabled, openalex_enabled, arxiv_enabled,
    real_threshold, suspicious_threshold,
    theme_name,
    last_verification_report
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import streamlit as st

from config.settings import settings


DEFAULTS: dict[str, Any] = {
    "theme_name": "浅色",
    "deepseek_api_key": "",
    "deepseek_chat_model": "",
    "deepseek_vl_model": "",
    "crossref_enabled": True,
    "openalex_enabled": True,
    "arxiv_enabled": True,
    "real_threshold": settings.real_threshold,
    "suspicious_threshold": settings.suspicious_threshold,
    "last_verification_report": None,
    "pending_mode": "chat",
}


def init_session() -> None:
    """Seed defaults + the first chat session."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state.get("deepseek_api_key") and settings.deepseek_api_key:
        st.session_state["deepseek_api_key"] = settings.deepseek_api_key
    if not st.session_state.get("deepseek_chat_model"):
        st.session_state["deepseek_chat_model"] = settings.chat_model
    if not st.session_state.get("deepseek_vl_model"):
        st.session_state["deepseek_vl_model"] = settings.vl_model

    if "sessions" not in st.session_state:
        st.session_state["sessions"] = {}
    if not st.session_state["sessions"]:
        new_session(activate=True)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_session(activate: bool = True) -> str:
    sid = uuid.uuid4().hex[:10]
    st.session_state["sessions"][sid] = {
        "id": sid,
        "title": "新对话",
        "created_at": _now_iso(),
        "messages": [],
    }
    if activate:
        st.session_state["active_session_id"] = sid
        st.session_state["pending_mode"] = "chat"
    return sid


def delete_session(sid: str) -> None:
    sessions: dict[str, dict] = st.session_state.get("sessions", {})
    sessions.pop(sid, None)
    if st.session_state.get("active_session_id") == sid:
        st.session_state["active_session_id"] = next(iter(sessions), None)
        if st.session_state["active_session_id"] is None:
            new_session(activate=True)


def switch_session(sid: str) -> None:
    if sid in st.session_state.get("sessions", {}):
        st.session_state["active_session_id"] = sid


def active_session() -> dict[str, Any]:
    sid = st.session_state.get("active_session_id")
    sessions: dict[str, dict] = st.session_state.get("sessions", {})
    if sid not in sessions:
        sid = new_session(activate=True)
    return sessions[sid]


def append_message(message: dict[str, Any]) -> None:
    sess = active_session()
    sess["messages"].append(message)
    # If still default-titled, derive a title from the first user message
    if sess["title"] == "新对话":
        first_user = next(
            (m for m in sess["messages"] if m.get("role") == "user" and m.get("text")),
            None,
        )
        if first_user:
            text = (first_user["text"] or "").strip().splitlines()[0]
            sess["title"] = (text[:24] + "…") if len(text) > 24 else (text or "新对话")
