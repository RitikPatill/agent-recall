"""SQLite persistence layer."""
from __future__ import annotations

import os
import sqlite3

DB_PATH = os.environ.get("RECALL_DB_PATH", "memories.db")

_conn: sqlite3.Connection | None = None


def _make_conn(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id          TEXT PRIMARY KEY,
            text        TEXT NOT NULL,
            embedding   BLOB NOT NULL,
            metadata    TEXT NOT NULL DEFAULT '{}',
            created_at  TEXT NOT NULL
        )
        """
    )
    conn.commit()


def get_conn() -> sqlite3.Connection:
    global _conn, DB_PATH
    # Re-read env var each call so monkeypatching works in tests
    path = os.environ.get("RECALL_DB_PATH", "memories.db")
    if _conn is None:
        _conn = _make_conn(path)
    return _conn
