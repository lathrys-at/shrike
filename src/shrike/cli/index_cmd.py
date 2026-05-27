from __future__ import annotations

import time
from typing import Any

import click
import httpx

from shrike.cli import output
from shrike.cli.output import output_options


@click.group("index", short_help="Manage the vector index")
def index() -> None:
    """Build and inspect the semantic search index."""


@index.command("rebuild", short_help="Rebuild the vector index from scratch")
@output_options
@click.pass_context
def index_rebuild(ctx: click.Context) -> None:
    """Drop and rebuild the vector index by re-embedding every note.

    The server continues accepting requests during the rebuild.
    Search results may be incomplete until the rebuild finishes.

    \b
    Examples:
      shrike index rebuild
      shrike --json index rebuild
    """
    base_url = ctx.obj["url"].rsplit("/", 1)[0]
    json_out: bool = ctx.obj["json"]

    try:
        resp = httpx.post(f"{base_url}/index/rebuild", timeout=30.0)
    except httpx.ConnectError as err:
        raise click.ClickException("Cannot connect to server. Is it running?") from err

    if resp.status_code == 400:
        body = resp.json()
        raise click.ClickException(body.get("error", "Index rebuild failed"))

    body = resp.json()

    if body.get("status") == "complete":
        if json_out:
            output.emit_json(body)
        else:
            output.console.print("[dim]Collection is empty, nothing to index.[/dim]")
        return

    if body.get("status") == "already_building" and not json_out:
        output.console.print("[dim]Index rebuild already in progress.[/dim]")

    total = body.get("total") or body.get("progress", {}).get("total", 0)

    _poll_progress(base_url, total, json_out=json_out)


def _poll_progress(base_url: str, total: int, *, json_out: bool) -> None:
    """Poll /status until the index build completes or errors."""
    if json_out:
        _poll_progress_json(base_url)
        return

    with output.console.status("", spinner="dots") as status:
        while True:
            idx_status = _fetch_index_status(base_url)
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

    idx_status = _fetch_index_status(base_url) or {}
    size = idx_status.get("size", 0)
    ndim = idx_status.get("ndim", "?")
    output.console.print(f"Index ready: [green]{size}[/green] notes, {ndim} dims")


def _poll_progress_json(base_url: str) -> None:
    """Poll and emit final JSON when done."""
    while True:
        idx_status = _fetch_index_status(base_url)
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


def _fetch_index_status(base_url: str) -> dict[str, Any] | None:
    """Fetch just the index portion of /status."""
    try:
        resp = httpx.get(f"{base_url}/status", timeout=5.0)
        if resp.status_code == 200:
            data: dict[str, Any] = resp.json()
            return data.get("index")
    except (httpx.ConnectError, httpx.TimeoutException):
        pass
    return None
