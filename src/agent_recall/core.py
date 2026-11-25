"""
Core public API for agent-recall (implemented in M2).

These four functions form the complete interface for storing and retrieving
agent memories. All storage is local SQLite; embeddings are computed either
via OpenAI or offline TF-IDF depending on configuration.
"""


def store(text: str, metadata: dict | None = None) -> str:
    """Embed and persist a memory snippet.

    Args:
        text: The text to store as a memory.
        metadata: Optional JSON-serialisable tags (e.g. {"session": "abc"}).

    Returns:
        The UUID string assigned to the new memory.
    """
    ...


def search(query: str, top_k: int = 5) -> list[dict]:
    """Return the k most semantically similar memories.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of results to return.

    Returns:
        List of dicts with keys: id, text, metadata, score.
    """
    ...


def forget(memory_id: str) -> None:
    """Hard-delete a memory by ID.

    Args:
        memory_id: UUID of the memory to delete.
    """
    ...


def wipe() -> None:
    """Delete ALL memories from the store. Irreversible."""
    ...
