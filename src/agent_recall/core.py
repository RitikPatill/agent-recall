"""
Core public API for agent-recall.

These four functions form the complete interface for storing and retrieving
agent memories. All storage is local SQLite; embeddings are computed either
via OpenAI or offline TF-IDF depending on configuration.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime

import numpy as np

from agent_recall import db as _db
from agent_recall.embeddings import embed_one


def store(text: str, metadata: dict | None = None) -> str:
    """Embed and persist a memory snippet.

    Args:
        text: The text to store as a memory.
        metadata: Optional JSON-serialisable tags (e.g. {"session": "abc"}).

    Returns:
        The UUID string assigned to the new memory.
    """
    memory_id = str(uuid.uuid4())
    vec = embed_one(text)
    blob = vec.astype(np.float32).tobytes()
    meta_json = json.dumps(metadata or {})
    created_at = datetime.utcnow().isoformat()

    conn = _db.get_conn()
    conn.execute(
        "INSERT INTO memories (id, text, embedding, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        (memory_id, text, blob, meta_json, created_at),
    )
    conn.commit()
    return memory_id


def search(query: str, top_k: int = 5) -> list[dict]:
    """Return the k most semantically similar memories.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of results to return.

    Returns:
        List of dicts with keys: id, text, metadata, score.
    """
    query_vec = embed_one(query)
    query_norm = np.linalg.norm(query_vec)

    conn = _db.get_conn()
    rows = conn.execute(
        "SELECT id, text, embedding, metadata FROM memories"
    ).fetchall()

    scored: list[tuple[float, dict]] = []
    for row in rows:
        vec = np.frombuffer(row["embedding"], dtype=np.float32)
        if vec.shape != query_vec.shape:
            # Dimension mismatch due to TF-IDF corpus growth; skip stale entry
            continue
        vec_norm = np.linalg.norm(vec)
        if query_norm == 0.0 or vec_norm == 0.0:
            score = 0.0
        else:
            score = float(np.dot(query_vec, vec) / (query_norm * vec_norm))
        scored.append((score, {
            "id": row["id"],
            "text": row["text"],
            "metadata": json.loads(row["metadata"]),
            "score": score,
        }))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def forget(memory_id: str) -> None:
    """Hard-delete a memory by ID.

    Args:
        memory_id: UUID of the memory to delete.
    """
    conn = _db.get_conn()
    conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()


def wipe() -> None:
    """Delete ALL memories from the store. Irreversible."""
    conn = _db.get_conn()
    conn.execute("DELETE FROM memories")
    conn.commit()
