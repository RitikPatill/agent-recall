"""CLI entry-point for agent-recall (implemented in M3)."""

import click


@click.group()
def cli() -> None:
    """agent-recall: persistent semantic memory for LLM agents."""


@cli.command()
@click.argument("text")
@click.option("--meta", default=None, help="JSON metadata string")
def store(text: str, meta: str | None) -> None:
    """Store a new memory snippet."""
    click.echo("not implemented")


@cli.command()
@click.argument("query")
@click.option("-k", default=5, help="Number of results to return")
def search(query: str, k: int) -> None:
    """Search memories semantically."""
    click.echo("not implemented")


@cli.command("list")
def list_memories() -> None:
    """List all stored memories."""
    click.echo("not implemented")


@cli.command()
@click.argument("memory_id")
def forget(memory_id: str) -> None:
    """Delete a memory by ID."""
    click.echo("not implemented")
