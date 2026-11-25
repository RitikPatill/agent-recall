# agent-recall

> Persistent, semantically searchable memory for LLM agents — zero infrastructure, one file on disk.

---

## Problem

Most LLM agent demos lose all context when the process exits. Full vector databases (Pinecone, Weaviate) solve this but require external infrastructure, accounts, and ongoing cost. Agent Recall fills the gap: it is the **SQLite of agent memory** — a local file, one `pip install`, no servers.

```
 Agent Loop
 ──────────
 User message
      │
      ▼
 recall.search(msg)        ← retrieve top-k relevant memories
      │
      ▼
 Inject into system prompt
      │
      ▼
 LLM call (Claude / GPT)
      │
      ▼
 recall.store(response)    ← persist assistant reply
      │
      ▼
 Return to user
```

Memories survive process restarts and are queryable in milliseconds using cosine similarity over locally stored embeddings.

---

## What works (M1 + M2)

The package is installable via `pip install -e ".[dev]"` and the core memory operations are fully functional.

**M1 — Scaffold**

- **Package layout** — `src/agent_recall/` with modules for every planned component: `core`, `db`, `embeddings`, `cli`, `server`.
- **`pyproject.toml`** — `hatchling` build, Python >= 3.10, all runtime and dev dependencies pinned (`fastapi`, `uvicorn`, `click`, `openai`, `scikit-learn`, `numpy`, `pydantic`; dev extras: `pytest`, `ruff`, `mypy`, `httpx`).
- **`recall` entry point** — registered as `recall = agent_recall.cli:cli`; the Click app is stubbed pending M3.
- **Tooling** — `ruff` (line-length 88, py310) and `mypy` (strict) configured in `pyproject.toml`.

**M2 — Core storage layer**

- **SQLite schema** — `memories` table with columns `id` (UUID), `text`, `embedding` (BLOB), `metadata` (JSON), and `created_at`. Database file is created automatically on first use.
- **`embed()`** — dual-backend embedding helper: `text-embedding-3-small` via the OpenAI API when `OPENAI_API_KEY` is set; offline scikit-learn TF-IDF fallback when it is not.
- **`store(text, metadata)`** — embeds and persists a memory; returns the assigned UUID.
- **`search(query, top_k)`** — embeds the query, loads all stored vectors into NumPy, ranks by cosine similarity, and returns the top-k results with scores.
- **`forget(memory_id)`** — hard-deletes a single memory by UUID.
- **`wipe()`** — clears all memories from the database.
- **Unit tests** — `tests/test_core.py` covers all four public functions using the offline TF-IDF backend (no API key required).

---

## API

```python
import agent_recall as recall

# Store a memory
memory_id = recall.store("User prefers dark mode", metadata={"session": "abc"})

# Retrieve relevant memories
results = recall.search("What are the user's UI preferences?", top_k=3)
# [{"id": "...", "text": "User prefers dark mode", "metadata": {...}, "score": 0.92}]

# Delete a specific memory
recall.forget(memory_id)

# Wipe everything
recall.wipe()
```

---

## CLI

> Stubbed; wired to the `recall` entry point. Full implementation in M3.

```bash
# Store a snippet
recall store "User is based in Berlin"

# Semantic search
recall search "where does the user live?" -k 5

# List all memories
recall list

# Delete by ID
recall forget <memory-id>
```

---

## REST API

> Stubbed in `server.py`. Full implementation in M4.

| Method   | Endpoint             | Description                    |
|----------|----------------------|--------------------------------|
| `POST`   | `/store`             | Embed and store a memory       |
| `GET`    | `/search?q=...&k=5`  | Semantic search over memories  |
| `DELETE` | `/forget/{id}`       | Hard-delete a memory by ID     |
| `DELETE` | `/wipe`              | Clear all memories             |

Start the server:

```bash
uvicorn agent_recall.server:app --reload
```

---

## Architecture

```
src/agent_recall/
├── __init__.py      # package version + public re-exports
├── core.py          # public API: store, search, forget, wipe
├── db.py            # SQLite persistence layer
├── embeddings.py    # OpenAI + offline TF-IDF backends
├── cli.py           # Click CLI                         (M3)
└── server.py        # FastAPI REST server               (M4)

tests/
├── test_placeholder.py   # version smoke test
└── test_core.py          # unit tests for M2 (offline TF-IDF backend)

examples/                 # agent integration demos       (M5)
```

Storage: a single `memories.db` SQLite file in the working directory. Each row holds the raw text, a JSON blob for optional metadata, and the embedding vector serialised as a binary blob. At query time, all vectors are loaded into NumPy and ranked by cosine similarity; no approximate-nearest-neighbour index is required at the scale this library targets.

**Embedding backends:**
- `openai` — `text-embedding-3-small` via the OpenAI API; requires `OPENAI_API_KEY`.
- `tfidf` — offline scikit-learn TF-IDF; no API key required; suitable for CI and local demos.

---

## Installation

Once published:

```bash
pip install agent-recall
```

For development (current):

```bash
pip install -e ".[dev]"
```

**Embedding backends:**
- **OpenAI** (`text-embedding-3-small`): set `OPENAI_API_KEY` in your environment.
- **Offline TF-IDF**: works with no API key — useful for CI and local demos.

---

## Status

| Milestone | Description                              | Status  |
|-----------|------------------------------------------|---------|
| M1        | Scaffold + README                        | Done    |
| M2        | SQLite storage + embedding backends      | Done    |
| M3        | Click CLI                                | Planned |
| M4        | FastAPI REST server                      | Planned |
| M5        | Claude agent example                     | Planned |
| M6        | Benchmarks + packaging                   | Planned |

---

## License

MIT © 2024 Ritik
