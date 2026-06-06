from __future__ import annotations

import base64
import mimetypes
import os
import shutil
from urllib.parse import urlparse

import click

from shrike.cli import output
from shrike.cli.output import output_options


@click.group("media", short_help="Manage media files")
def media() -> None:
    """Store, fetch, list, and delete files in the collection's media folder.

    Use this to get images/audio into the collection so cards can reference them
    (`<img src="NAME">`, `[sound:NAME]`). Anki resolves name collisions, so the
    stored name may differ from what you asked for — always use the returned name.
    """


def _fmt_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def _client_fetch(url: str) -> tuple[bytes, str | None]:
    """Download a URL on the client side (the user's machine is trusted).

    Honors httpx's proxy env vars (SOCKS needs the optional `httpx[socks]` extra);
    no SSRF guard here — that's the server-side concern for the `url` tool input.
    """
    import httpx

    resp = httpx.get(url, follow_redirects=True, timeout=30.0, trust_env=True)
    resp.raise_for_status()
    content_type = (resp.headers.get("content-type") or "").split(";")[0].strip() or None
    return resp.content, content_type


def _download(url: str) -> bytes:
    """GET a URL and return its bytes (used to pull a media file's server URL)."""
    import httpx

    resp = httpx.get(url, follow_redirects=True, timeout=30.0, trust_env=True)
    resp.raise_for_status()
    return resp.content


def _name_from_url(url: str, content_type: str | None) -> str:
    name = os.path.basename(urlparse(url).path)
    if name and "." in name:
        return name
    ext = mimetypes.guess_extension(content_type) if content_type else None
    return f"{name or 'media'}{ext or ''}"


@media.command("store", short_help="Store local files or URLs into the media folder")
@output_options
@click.argument("paths", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option("--name", help="Override the stored filename (single item only).")
@click.option("--url", "urls", multiple=True, help="URL to fetch and store (repeatable).")
@click.option(
    "--client-fetch",
    is_flag=True,
    help="Download --url files locally and upload the bytes (use when this machine "
    "has the network path/proxy, not the server). Default: the server fetches.",
)
@click.pass_context
def media_store(
    ctx: click.Context,
    paths: tuple[str, ...],
    name: str | None,
    urls: tuple[str, ...],
    client_fetch: bool,
) -> None:
    """Store one or more local PATHs and/or --url files into the media folder.

    Local files are read here and sent as bytes, so this works against a remote
    daemon too. URLs are fetched by the server by default (http/https, private
    addresses refused); pass --client-fetch to download them locally and upload
    the bytes instead.

    \b
    Examples:
      shrike media store diagram.png
      shrike media store a.png b.jpg c.ogg
      shrike media store --url https://example.com/cell.png
      shrike media store --url https://intranet/x.png --client-fetch
    """
    if name and len(paths) + len(urls) != 1:
        raise click.UsageError("--name can only be used with a single file.")
    if not paths and not urls:
        raise click.UsageError("Provide at least one PATH or --url.")

    items: list[dict[str, str]] = []
    for p in paths:
        with open(p, "rb") as fh:
            data = base64.b64encode(fh.read()).decode("ascii")
        items.append({"data": data, "filename": name or os.path.basename(p)})
    for u in urls:
        if client_fetch:
            raw, content_type = _client_fetch(u)
            items.append(
                {
                    "data": base64.b64encode(raw).decode("ascii"),
                    "filename": name or _name_from_url(u, content_type),
                }
            )
        else:
            item: dict[str, str] = {"url": u}
            if name:
                item["filename"] = name
            items.append(item)

    client = ctx.obj["client"]
    with output.spinner("Storing…"):
        result = client.store_media(items)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    errors = False
    for r in result.results:
        if r.status == "stored":
            note = " [dim](already present)[/dim]" if r.deduped else ""
            output.console.print(
                f"[green]+[/green] Stored [cyan]{r.filename}[/cyan] "
                f"[dim]({_fmt_size(r.size_bytes)})[/dim]{note}"
            )
        else:
            errors = True
            label = f" [cyan]{r.filename}[/cyan]" if r.filename else ""
            output.console.print(f"[bold red]![/bold red]{label} [red]{r.error}[/red]")
    if errors:
        ctx.exit(1)


@media.command("fetch", short_help="Read media files back out of the collection")
@output_options
@click.argument("names", nargs=-1, required=True)
@click.option("-o", "--output", "out", type=click.Path(), help="Write a single file to this path.")
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False),
    help="Directory to write files into (default: current directory).",
)
@click.pass_context
def media_fetch(
    ctx: click.Context,
    names: tuple[str, ...],
    out: str | None,
    out_dir: str | None,
) -> None:
    """Write media files NAMES out to local disk.

    With one NAME, -o sets the output path; with several, files are written into
    --out-dir (or the current directory) under their own names.

    \b
    Examples:
      shrike media fetch cell.png -o /tmp/cell.png
      shrike media fetch a.png b.png --out-dir ./exported
    """
    if out and len(names) > 1:
        raise click.UsageError("-o/--output works with a single NAME; use --out-dir for several.")

    client = ctx.obj["client"]
    with output.spinner("Fetching…"):
        result = client.fetch_media(names)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    failed = False
    for r in result.results:
        if r.status == "missing":
            failed = True
            output.console.print(f"[bold red]![/bold red] Not found: [cyan]{r.filename}[/cyan]")
            continue
        # `found`: read the server-side path if we share its disk, else download
        # the file's url over HTTP. The response never carries bytes.
        dest = out or os.path.join(out_dir or ".", r.filename)
        if os.path.isfile(r.path):
            shutil.copyfile(r.path, dest)
        elif r.url:
            with open(dest, "wb") as fh:
                fh.write(_download(r.url))
        else:
            failed = True
            output.console.print(
                f"[bold red]![/bold red] [cyan]{r.filename}[/cyan]: no local path and no URL "
                "to fetch from"
            )
            continue
        output.console.print(
            f"[green]+[/green] Wrote [cyan]{dest}[/cyan] [dim]({_fmt_size(r.size_bytes)})[/dim]"
        )
    if failed:
        ctx.exit(1)


@media.command("list", short_help="List media files in the collection")
@output_options
@click.argument("pattern", required=False)
@click.option("--limit", type=int, default=None, help="Maximum files to show.")
@click.pass_context
def media_list(ctx: click.Context, pattern: str | None, limit: int | None) -> None:
    """List media filenames, optionally filtered by a glob PATTERN.

    \b
    Examples:
      shrike media list
      shrike media list "*.png"
      shrike media list "cell-*" --limit 20
    """
    client = ctx.obj["client"]
    with output.spinner("Listing…"):
        result = client.list_media(pattern=pattern, limit=limit)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    if not result.files:
        output.console.print("[dim]No media files found.[/dim]")
        return

    shown = len(result.files)
    count = f"{shown} of {result.count}" if result.count > shown else str(result.count)
    filt = f" matching [cyan]{pattern}[/cyan]" if pattern else ""
    output.console.print(
        f"[dim]Showing {count} media file(s){filt} in [cyan]{result.media_dir}[/cyan][/dim]"
    )
    output.console.print()
    rows = [[f.filename, f.mime or "", _fmt_size(f.size_bytes)] for f in result.files]
    output.table(["Name", "Type", "Size"], rows)
    output.console.print()


@media.command("delete", short_help="Delete media files (to Anki's trash)")
@output_options
@click.argument("names", nargs=-1, required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
@click.pass_context
def media_delete(ctx: click.Context, names: tuple[str, ...], yes: bool) -> None:
    """Delete media files by NAME, moving them to Anki's recoverable trash.

    Does not check whether a note still references the file — run
    'shrike collection check' first to find unused media.

    \b
    Examples:
      shrike media delete old.png
      shrike media delete a.png b.png --yes
    """
    if not yes and not click.confirm(f"Delete {len(names)} media file(s)?"):
        output.console.print("Cancelled.")
        return

    client = ctx.obj["client"]
    with output.spinner("Deleting…"):
        result = client.delete_media(list(names))

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    for n in result.deleted:
        output.console.print(f"[red]-[/red] Deleted [cyan]{n}[/cyan]")
    for n in result.not_found:
        output.console.print(f"[bold red]![/bold red] Not found: [cyan]{n}[/cyan]")
    if result.not_found:
        ctx.exit(1)
