from __future__ import annotations

from typing import Any

import click

from shrike.cli import output
from shrike.cli.config import resolve_embedding
from shrike.cli.index_cmd import _poll_progress
from shrike.cli.output import output_options
from shrike.client import ShrikeClient


@click.group("embedding", short_help="Manage the embedding service")
def embedding() -> None:
    """Start, stop, and inspect the embedding service used for semantic search."""


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
    client: ShrikeClient = ctx.obj["client"]
    json_out: bool = ctx.obj["json"]

    with output.spinner("Checking embedding service…"):
        emb_status = client.embedding_status()

    if json_out:
        output.emit_json(emb_status)
        return

    _render_embedding(emb_status)


@embedding.command("start", short_help="Start the embedding service")
@output_options
@click.option("--embedding-model", "model", type=click.Path(), help="Path to GGUF embedding model.")
@click.option(
    "--llama-server", "llama_server", type=click.Path(), help="Path to llama-server binary."
)
@click.option(
    "--embedding-port", "port", type=int, help="Port for the embedding server (default: 8373)."
)
@click.option(
    "--embedding-context-size", "context_size", type=int, help="Context size for embedding model."
)
@click.option(
    "--embedding-threads", "threads", type=int, help="CPU threads for embedding inference."
)
@click.option("--embedding-gpu-layers", "gpu_layers", type=int, help="Layers to offload to GPU.")
@click.option(
    "--background", is_flag=True, help="Return immediately without waiting for any index rebuild."
)
@click.pass_context
def embedding_start(
    ctx: click.Context,
    model: str | None,
    llama_server: str | None,
    port: int | None,
    context_size: int | None,
    threads: int | None,
    gpu_layers: int | None,
    background: bool,
) -> None:
    """Start the embedding service on a running server.

    Resolves the model and llama-server settings via the same config → env →
    flag cascade as `shrike server start`. If the embedding model changed (or
    the index is stale), an index rebuild is triggered automatically.

    \b
    Examples:
      shrike embedding start
      shrike embedding start --embedding-model ~/models/embed.gguf
      shrike embedding start --background
    """
    config = ctx.obj["config"]
    client: ShrikeClient = ctx.obj["client"]
    json_out: bool = ctx.obj["json"]

    resolved = resolve_embedding(
        config,
        model=model,
        port=port,
        context_size=context_size,
        threads=threads,
        gpu_layers=gpu_layers,
        llama_server=llama_server,
    )

    with output.spinner("Starting embedding service…"):
        data = client.embedding_start(**resolved)

    if data.get("status") == "already_running":
        if json_out:
            output.emit_json(data)
        else:
            output.console.print("[dim]Embedding service is already running.[/dim]")
        return

    idx = data.get("index", {})
    building = idx.get("state") == "building"

    if building and not background:
        total = idx.get("progress", {}).get("total", 0)
        _poll_progress(client, total, json_out=json_out)
        if not json_out:
            _render_embedding(client.embedding_status())
        return

    if json_out:
        output.emit_json(data)
    else:
        output.success("Embedding service started.")
        _render_embedding(data.get("embedding", {}))
        if building:
            output.console.print("[dim]Index rebuild started in the background.[/dim]")


@embedding.command("stop", short_help="Stop the embedding service")
@output_options
@click.pass_context
def embedding_stop(ctx: click.Context) -> None:
    """Stop the embedding service on a running server.

    The Shrike server and the Anki collection stay up; semantic search becomes
    unavailable until the service is started again. Useful for llama-server
    upgrades, model swaps, or freeing GPU/RAM during maintenance.

    \b
    Examples:
      shrike embedding stop
      shrike --json embedding stop
    """
    client: ShrikeClient = ctx.obj["client"]
    json_out: bool = ctx.obj["json"]

    with output.spinner("Stopping embedding service…"):
        data = client.embedding_stop()

    if json_out:
        output.emit_json(data)
        return

    if data.get("status") == "stopped":
        output.success("Embedding service stopped.")
    else:
        output.console.print("[dim]Embedding service is not running.[/dim]")


def _render_embedding(emb: dict[str, Any]) -> None:
    """Render the embedding status block (shared by status and start)."""
    if emb.get("available"):
        output.kv("Embedding", "[green]available[/green]")
        if emb.get("url"):
            output.kv("URL", f"[cyan]{emb['url']}[/cyan]", indent=2)
        if emb.get("pid"):
            output.kv("PID", f"[cyan]{emb['pid']}[/cyan]", indent=2)
        if emb.get("model"):
            output.kv("Model", f"[cyan]{emb['model']}[/cyan]", indent=2)
    else:
        output.kv("Embedding", "[dim]unavailable[/dim]")
