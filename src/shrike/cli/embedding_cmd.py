from __future__ import annotations

from typing import Any

import click
import httpx

from shrike.cli import output
from shrike.cli.output import output_options


@click.group("embedding", short_help="Manage the embedding service")
def embedding() -> None:
    """Inspect the embedding service used for semantic search."""


@embedding.command("status", short_help="Show embedding service status")
@output_options
@click.pass_context
def embedding_status(ctx: click.Context) -> None:
    """Show the current state of the embedding service.

    \b
    Examples:
      shrike embedding status
      shrike --json embedding status
    """
    base_url = ctx.obj["url"].rsplit("/", 1)[0]
    json_out: bool = ctx.obj["json"]

    with output.spinner("Checking embedding service…"):
        emb_status = _fetch_embedding_status(base_url)

    if emb_status is None:
        raise click.ClickException("Cannot connect to server. Is it running?")

    if json_out:
        output.emit_json(emb_status)
        return

    if emb_status.get("available"):
        output.kv("Embedding", "[green]available[/green]")
        if emb_status.get("url"):
            output.kv("URL", f"[cyan]{emb_status['url']}[/cyan]", indent=2)
        if emb_status.get("pid"):
            output.kv("PID", f"[cyan]{emb_status['pid']}[/cyan]", indent=2)
        if emb_status.get("model"):
            output.kv("Model", f"[cyan]{emb_status['model']}[/cyan]", indent=2)
    else:
        output.kv("Embedding", "[dim]unavailable[/dim]")


def _fetch_embedding_status(base_url: str) -> dict[str, Any] | None:
    """Fetch just the embedding portion of /status."""
    try:
        resp = httpx.get(f"{base_url}/status", timeout=5.0)
        if resp.status_code == 200:
            data: dict[str, Any] = resp.json()
            return data.get("embedding")
    except (httpx.ConnectError, httpx.TimeoutException):
        pass
    return None
