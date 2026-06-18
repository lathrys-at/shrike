from __future__ import annotations

import time

import click

from shrike.cli import output, status_render
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
      shrike server index rebuild
      shrike server index rebuild --background
      shrike --json server index rebuild
    """
    client: ShrikeClient = ctx.obj["client"]
    json_out: bool = ctx.obj["json"]

    body = client.index_rebuild()

    if body.status == "complete":
        if json_out:
            output.emit_json(body)
        else:
            output.console.print("[dim]Collection is empty, nothing to index.[/dim]")
        return

    if body.status == "already_building":
        if not json_out:
            output.console.print("[dim]Index rebuild already in progress.[/dim]")
        total = body.progress.total
    else:  # started
        total = body.total

    if background:
        if json_out:
            output.emit_json(body)
        else:
            output.console.print(f"Index rebuild started ({total} notes)")
        return

    _poll_progress(client, total, json_out=json_out)


@index.command("save", short_help="Persist the vector index to disk now")
@output_options
@click.pass_context
def index_save(ctx: click.Context) -> None:
    """Flush the in-memory vector index to disk immediately.

    The index is saved automatically — after a quiet period following edits, on
    a large batch of changes, and on graceful shutdown — so this is rarely
    needed. Use it to force a checkpoint: before a risky operation, or to
    capture recent edits without stopping the server.

    \b
    Examples:
      shrike server index save
      shrike --json server index save
    """
    client: ShrikeClient = ctx.obj["client"]
    json_out: bool = ctx.obj["json"]

    with output.spinner("Saving index…"):
        body = client.index_save()

    if json_out:
        output.emit_json(body)
        return

    if body.status == "saved":
        if body.size == 0 and body.pending == 0:
            output.console.print("[dim]Nothing to save — the index is empty.[/dim]")
        else:
            pending = f" ([yellow]{body.pending}[/yellow] pending)" if body.pending else ""
            output.console.print(f"Index saved: [green]{body.size}[/green] vectors{pending}")
    elif body.status == "empty":
        output.console.print("[dim]No index to save — none has been built yet.[/dim]")
    else:  # building
        p = body.progress
        output.console.print(
            f"[yellow]Index is building[/yellow] ({p.indexed} / {p.total} notes); not saved."
        )


@index.command("status", short_help="Show index status")
@output_options
@click.pass_context
def index_status(ctx: click.Context) -> None:
    """Show the current state of the vector index.

    \b
    Examples:
      shrike server index status
      shrike --json server index status
    """
    client: ShrikeClient = ctx.obj["client"]
    json_out: bool = ctx.obj["json"]

    with output.spinner("Checking index…"):
        idx_status = client.index_status()

    if json_out:
        output.emit_json(idx_status)
        return

    # Shared renderer: identical to the `server status` Index block,
    # including the per-modality sub-index breakdown.
    status_render.render_index(idx_status)


def _poll_progress(client: ShrikeClient, total: int, *, json_out: bool) -> None:
    """Poll /status until the index build completes or errors."""
    if json_out:
        _poll_progress_json(client)
        return

    with output.console.status("", spinner="dots") as status:
        while True:
            full = client.server_status()
            idx_status = full.index if full else None
            if idx_status is None:
                status.update("Indexing…")
                time.sleep(0.5)
                continue

            if idx_status.state == "ready":
                break
            if idx_status.state == "error":
                raise click.ClickException(f"Index rebuild failed: {idx_status.error}")
            if idx_status.state == "building":
                total = idx_status.progress.total
                status.update(f"Indexing… {idx_status.progress.indexed} / {total} notes")
            else:
                status.update("Indexing…")
            time.sleep(0.5)

    full = client.server_status()
    idx_status = full.index if full else None
    size = idx_status.size if idx_status else 0
    ndim = idx_status.ndim if idx_status and idx_status.ndim is not None else "?"
    output.console.print(f"Index ready: [green]{size}[/green] notes, {ndim} dims")


def _poll_progress_json(client: ShrikeClient) -> None:
    """Poll and emit final JSON when done."""
    while True:
        full = client.server_status()
        idx_status = full.index if full else None
        if idx_status is None:
            time.sleep(0.5)
            continue

        if idx_status.state == "ready":
            output.emit_json({"status": "complete", **idx_status.model_dump(exclude_none=True)})
            return
        if idx_status.state == "error":
            output.emit_json({"status": "error", **idx_status.model_dump(exclude_none=True)})
            raise SystemExit(1)

        time.sleep(0.5)
