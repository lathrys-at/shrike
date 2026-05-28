from __future__ import annotations

from typing import Any

import click
import httpx

from shrike.cli import output
from shrike.cli.config import resolve_embedding
from shrike.cli.index_cmd import _poll_progress
from shrike.cli.output import output_options


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
    base_url = ctx.obj["url"].rsplit("/", 1)[0]
    json_out: bool = ctx.obj["json"]

    with output.spinner("Checking embedding service…"):
        emb_status = _fetch_embedding_status(base_url)

    if emb_status is None:
        raise click.ClickException("Cannot connect to server. Is it running?")

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
    base_url = ctx.obj["url"].rsplit("/", 1)[0]
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
    if not resolved.get("model"):
        raise click.UsageError(
            "No embedding model configured. Provide --embedding-model, set "
            "SHRIKE_EMBEDDING_MODEL, or add embedding.model to your config."
        )

    body = {k: v for k, v in resolved.items() if v is not None}

    with output.spinner("Starting embedding service…"):
        try:
            resp = httpx.post(f"{base_url}/embedding/start", json=body, timeout=120.0)
        except httpx.ConnectError as err:
            raise click.ClickException("Cannot connect to server. Is it running?") from err
        except httpx.TimeoutException as err:
            raise click.ClickException("Timed out starting the embedding service.") from err

    if resp.status_code >= 400:
        raise click.ClickException(resp.json().get("error", "Failed to start embedding service"))

    data = resp.json()

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
        _poll_progress(base_url, total, json_out=json_out)
        if not json_out:
            emb = _fetch_embedding_status(base_url)
            if emb:
                _render_embedding(emb)
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
    base_url = ctx.obj["url"].rsplit("/", 1)[0]
    json_out: bool = ctx.obj["json"]

    with output.spinner("Stopping embedding service…"):
        try:
            resp = httpx.post(f"{base_url}/embedding/stop", timeout=30.0)
        except httpx.ConnectError as err:
            raise click.ClickException("Cannot connect to server. Is it running?") from err

    data = resp.json()
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
