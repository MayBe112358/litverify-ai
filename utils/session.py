"""Session state for the agent shell — supports multiple chat sessions.

Shape (everything lives under ``st.session_state``)::

    sessions:           dict[str, dict]    # id -> session record
    active_session_id:  str | None         # currently selected session

    # Single-shot keys read by services:
    deepseek_api_key, deepseek_chat_model,
    real_threshold, suspicious_threshold,
    theme_name

``st.session_state`` evaporates on every page reload / websocket reconnect /
server restart, so sessions are written through to SQLite on each appended
message and restored here on the next ``init_session``. Persistence is
best-effort: a DB hiccup never breaks the chat itself.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import streamlit as st

from config.settings import settings
from db.history import delete_chat_session, load_chat_sessions, save_chat_session


DEFAULTS: dict[str, Any] = {
    "theme_name": "浅色",
    "deepseek_api_key": "",
    "deepseek_chat_model": "",
    "real_threshold": settings.real_threshold,
    "suspicious_threshold": settings.suspicious_threshold,
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

    if "sessions" not in st.session_state:
        restored: dict[str, dict] = {}
        try:
            restored = {s["id"]: s for s in load_chat_sessions()}
        except Exception:  # noqa: BLE001 - persistence is best-effort
            restored = {}
        st.session_state["sessions"] = restored
        if restored:
            # Most recently updated session first (load order) — reopen it so
            # a reload/reconnect lands back in the conversation, not a blank page.
            st.session_state["active_session_id"] = next(iter(restored))
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
    return sid


def delete_session(sid: str) -> None:
    sessions: dict[str, dict] = st.session_state.get("sessions", {})
    sessions.pop(sid, None)
    try:
        delete_chat_session(sid)
    except Exception:  # noqa: BLE001 - persistence is best-effort
        pass
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
    try:
        save_chat_session(sess)
    except Exception:  # noqa: BLE001 - persistence is best-effort
        pass
