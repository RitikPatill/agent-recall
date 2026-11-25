"""FastAPI REST server for agent-recall (implemented in M4)."""

from fastapi import FastAPI

app = FastAPI(title="agent-recall", version="0.1.0")


@app.post("/store")
async def store() -> dict:
    """Store a new memory."""
    return {"status": "not implemented"}


@app.get("/search")
async def search() -> dict:
    """Search memories by semantic similarity."""
    return {"status": "not implemented"}


@app.delete("/forget/{memory_id}")
async def forget(memory_id: str) -> dict:
    """Delete a memory by ID."""
    return {"status": "not implemented"}


@app.delete("/wipe")
async def wipe() -> dict:
    """Delete all memories."""
    return {"status": "not implemented"}
