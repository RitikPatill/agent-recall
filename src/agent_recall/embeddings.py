"""Embedding backends: OpenAI and offline TF-IDF.

WARNING (TF-IDF backend): The TF-IDF vectorizer is refit on the growing
corpus every time embed() is called. This means vectors produced before a
refit are **not comparable** to vectors produced after one — if new texts
expand the vocabulary, existing stored blobs become stale. This is
acceptable for the offline/test use case but unsuitable for production.
Use the OpenAI backend for a stable embedding space.
"""
from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
# RECALL_BACKEND: "openai" | "tfidf" | unset (auto-detect)


def _use_openai() -> bool:
    backend = os.environ.get("RECALL_BACKEND", "").lower()
    if backend == "tfidf":
        return False
    if backend == "openai":
        return True
    # auto-detect: use OpenAI if key is present
    return bool(os.environ.get("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# TF-IDF state
# ---------------------------------------------------------------------------
_vectorizer: object | None = None  # TfidfVectorizer instance
_corpus: list[str] = []


def _embed_tfidf(texts: list[str]) -> np.ndarray:
    """Fit/refit TF-IDF on cumulative corpus and return dense float32 matrix."""
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]

    global _vectorizer, _corpus

    # Add any texts not yet in corpus
    new_texts = [t for t in texts if t not in _corpus]
    _corpus.extend(new_texts)

    if len(_corpus) == 0:
        raise ValueError("Cannot embed with empty corpus.")

    # Refit on the full corpus
    _vectorizer = TfidfVectorizer()
    _vectorizer.fit(_corpus)  # type: ignore[union-attr]

    matrix = _vectorizer.transform(texts).toarray()  # type: ignore[union-attr]
    return matrix.astype(np.float32)


def _embed_openai(texts: list[str]) -> np.ndarray:
    import openai  # type: ignore[import-untyped]

    client = openai.OpenAI()
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([e.embedding for e in response.data], dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed(texts: list[str]) -> np.ndarray:
    """Return float32 array of shape (len(texts), dim)."""
    if _use_openai():
        return _embed_openai(texts)
    return _embed_tfidf(texts)


def embed_one(text: str) -> np.ndarray:
    """Convenience wrapper: embed a single string, return 1-D array."""
    return embed([text])[0]
