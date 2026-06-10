CREATE TABLE IF NOT EXISTS verification_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    raw_text TEXT,
    title TEXT,
    doi TEXT,
    verdict TEXT NOT NULL,
    score REAL NOT NULL,
    report_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_verification_history_created
ON verification_history(created_at DESC);

-- Chat sessions persisted across page reloads / server restarts.
-- st.session_state is memory-only, so every conversation is written through
-- here on append and restored on the next init_session.
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    messages_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated
ON chat_sessions(updated_at DESC);
