"""
agent-recall: Persistent, semantically searchable memory for LLM agents.

Gives agents a zero-infrastructure memory layer backed by SQLite and
cosine-similarity search. Memories survive process restarts and are
queryable in milliseconds at the start of every new session.
"""
from agent_recall.core import forget, search, store, wipe

__version__ = "0.1.0"
__all__ = ["store", "search", "forget", "wipe", "__version__"]
