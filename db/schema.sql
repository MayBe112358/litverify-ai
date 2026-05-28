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
