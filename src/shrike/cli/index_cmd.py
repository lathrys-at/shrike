from __future__ import annotations

import time

import click

from shrike.cli import output
from shrike.cli.output import output_options
from shrike.client import ShrikeClient


@click.group("index", short_help="Manage the vector index")
def index() -> None:
    """Build and inspect the semantic search index."""


@index.command("rebuild", short_help="Rebuild the vector index from scratch")
@output_options
@click.option(
    "--background",
    is_flag=True,
    help="Start the rebuild and return immediately without waiting.",
)
@click.pass_context
def index_rebuild(ctx: click.Context, background: bool) -> None:
    """Drop and rebuild the vector index by re-embedding every note.

    The server continues accepting requests during the rebuild.
    Search results may be incomplete until the rebuild finishes.

    \b
    Examples:
      shrike index rebuild
      shrike index rebuild --background
      shrike --json index rebuild
    """
    client: ShrikeClient = ctx.obj["client"]
    json_out: bool = ctx.obj["json"]

    body = client.index_rebuild()

    if body.get("status") == "complete":
        if json_out:
            output.emit_json(body)
        else:
            output.console.print("[dim]Collection is empty, nothing to index.[/dim]")
        return

    if body.get("status") == "already_building" and not json_out:
        output.console.print("[dim]Index rebuild already in progress.[/dim]")

    total = body.get("total") or body.get("progress", {}).get("total", 0)

    if background:
        if json_out:
            output.emit_json(body)
        else:
            output.console.print(f"Index rebuild started ({total} notes)")
        return

    _poll_progress(client, total, json_out=json_out)


@index.command("status", short_help="Show index status")
@output_options
@click.pass_context
def index_status(ctx: click.Context) -> None:
    """Show the current state of the vector index.

    \b
    Examples:
      shrike index status
      shrike --json index status
    """
    client: ShrikeClient = ctx.obj["client"]
    json_out: bool = ctx.obj["json"]

    with output.spinner("Checking index…"):
        idx_status = client.index_status()

    if json_out:
        output.emit_json(idx_status)
        return

    state = idx_status.get("state", "unknown")

    if state == "ready":
        output.kv("Index", "[green]ready[/green]")
        output.kv("Vectors", f"[green]{idx_status.get('size', 0)}[/green]", indent=2)
        output.kv("Dimensions", str(idx_status.get("ndim", "?")), indent=2)
    elif state == "building":
        progress = idx_status.get("progress", {})
        indexed = progress.get("indexed", 0)
        total = progress.get("total", 0)
        output.kv("Index", "[yellow]building[/yellow]")
        output.kv("Progress", f"{indexed} / {total} notes", indent=2)
    elif state == "error":
        output.kv("Index", "[red]error[/red]")
        output.kv("Error", idx_status.get("error", "unknown"), indent=2)
    elif state == "unavailable":
        output.kv("Index", "[dim]unavailable (no embedding service configured)[/dim]")
    else:
        output.kv("Index", f"[dim]{state}[/dim]")

    if idx_status.get("col_mod") is not None:
        output.kv("Collection mod", str(idx_status["col_mod"]), indent=2)
    if idx_status.get("path"):
        output.kv("Path", f"[cyan]{idx_status['path']}[/cyan]", indent=2)


def _poll_progress(client: ShrikeClient, total: int, *, json_out: bool) -> None:
    """Poll /status until the index build completes or errors."""
    if json_out:
        _poll_progress_json(client)
        return

    with output.console.status("", spinner="dots") as status:
        while True:
            full = client.server_status()
            idx_status = full.get("index") if full else None
            if idx_status is None:
                status.update("Indexing…")
                time.sleep(0.5)
                continue

            state = idx_status.get("state", "")
            if state == "ready":
                break

            if state == "error":
                raise click.ClickException(
                    f"Index rebuild failed: {idx_status.get('error', 'unknown error')}"
                )

            progress = idx_status.get("progress", {})
            indexed = progress.get("indexed", 0)
            total = progress.get("total", total)
            status.update(f"Indexing… {indexed} / {total} notes")
            time.sleep(0.5)

    full = client.server_status() or {}
    idx_status = full.get("index", {})
    size = idx_status.get("size", 0)
    ndim = idx_status.get("ndim", "?")
    output.console.print(f"Index ready: [green]{size}[/green] notes, {ndim} dims")


def _poll_progress_json(client: ShrikeClient) -> None:
    """Poll and emit final JSON when done."""
    while True:
        full = client.server_status()
        idx_status = full.get("index") if full else None
        if idx_status is None:
            time.sleep(0.5)
            continue

        state = idx_status.get("state", "")
        if state == "ready":
            output.emit_json({"status": "complete", **idx_status})
            return
        if state == "error":
            output.emit_json({"status": "error", **idx_status})
            raise SystemExit(1)

        time.sleep(0.5)
