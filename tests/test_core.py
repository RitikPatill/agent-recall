"""Tests for core storage API using the offline TF-IDF backend."""
import os

os.environ["RECALL_BACKEND"] = "tfidf"

import pytest
import agent_recall.db as db_mod
import agent_recall.embeddings as emb_mod
from agent_recall.core import forget, search, store, wipe


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Give each test a fresh DB file and reset all cached state."""
    monkeypatch.setenv("RECALL_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(db_mod, "_conn", None, raising=False)
    # Reset TF-IDF corpus so each test starts clean
    monkeypatch.setattr(emb_mod, "_vectorizer", None, raising=False)
    monkeypatch.setattr(emb_mod, "_corpus", [], raising=False)
    yield


def test_store_returns_id():
    mem_id = store("hello")
    assert isinstance(mem_id, str)
    assert len(mem_id) > 0


def test_search_finds_stored_memory():
    store("the cat sat on the mat")
    results = search("cat mat", top_k=5)
    texts = [r["text"] for r in results]
    assert "the cat sat on the mat" in texts


def test_search_top_k():
    sentences = [
        "the quick brown fox",
        "jumps over the lazy dog",
        "hello world",
        "python programming",
        "machine learning models",
        "neural networks are powerful",
        "data science pipeline",
        "natural language processing",
        "vector embeddings store",
        "sqlite database query",
    ]
    for s in sentences:
        store(s)
    results = search("fox dog", top_k=3)
    assert len(results) <= 3


def test_search_score_range():
    store("the sky is blue")
    store("water is wet")
    results = search("sky water", top_k=5)
    for r in results:
        assert -1.0 <= r["score"] <= 1.0


def test_forget_removes_memory():
    mem_id = store("unique phrase about penguins")
    forget(mem_id)
    results = search("unique phrase about penguins", top_k=5)
    ids = [r["id"] for r in results]
    assert mem_id not in ids


def test_wipe_clears_all():
    store("memory one")
    store("memory two")
    store("memory three")
    wipe()
    results = search("memory", top_k=10)
    assert results == []


def test_metadata_roundtrip():
    mem_id = store("session tagged memory", metadata={"session": "x", "turn": 1})
    results = search("session tagged memory", top_k=5)
    match = next((r for r in results if r["id"] == mem_id), None)
    assert match is not None
    assert match["metadata"] == {"session": "x", "turn": 1}
