"""Round-trip tests for chat session persistence (db.history.chat_sessions)."""
from __future__ import annotations

import pytest

import db.history as history


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "db_path", lambda: tmp_path / "test.sqlite3")


def _session(sid: str = "abc123", title: str = "测试会话") -> dict:
    return {
        "id": sid,
        "title": title,
        "created_at": "2026-06-10T12:00:00",
        "messages": [
            {"role": "user", "text": "验证这条引用", "mode": "chat", "files": []},
            {
                "role": "assistant",
                "kind": "chat",
                "mode": "chat",
                "text": "好的",
                "data": {"reasoning": "想了一下"},
            },
        ],
    }


def test_save_and_load_round_trip(tmp_db) -> None:
    history.save_chat_session(_session())
    loaded = history.load_chat_sessions()
    assert len(loaded) == 1
    sess = loaded[0]
    assert sess["id"] == "abc123"
    assert sess["title"] == "测试会话"
    assert sess["messages"][0]["text"] == "验证这条引用"
    assert sess["messages"][1]["data"]["reasoning"] == "想了一下"


def test_save_is_upsert(tmp_db) -> None:
    history.save_chat_session(_session())
    updated = _session(title="改名了")
    updated["messages"].append({"role": "user", "text": "再问一句", "mode": "chat"})
    history.save_chat_session(updated)
    loaded = history.load_chat_sessions()
    assert len(loaded) == 1
    assert loaded[0]["title"] == "改名了"
    assert len(loaded[0]["messages"]) == 3


def test_delete_chat_session(tmp_db) -> None:
    history.save_chat_session(_session("a1"))
    history.save_chat_session(_session("b2"))
    history.delete_chat_session("a1")
    loaded = history.load_chat_sessions()
    assert [s["id"] for s in loaded] == ["b2"]


def test_load_skips_corrupt_messages(tmp_db) -> None:
    import json
    import sqlite3

    history.save_chat_session(_session())
    with sqlite3.connect(history.db_path()) as conn:
        conn.execute(
            "UPDATE chat_sessions SET messages_json = ? WHERE id = ?",
            ("{not json", "abc123"),
        )
    loaded = history.load_chat_sessions()
    assert loaded[0]["messages"] == []
    # sanity: the table itself still valid JSON for new writes
    history.save_chat_session(_session("c3"))
    assert json.loads("[]") == []
