"""``shrike export`` — export the collection (or a deck/selection) to a package (#71)."""

from __future__ import annotations

import click

from shrike.cli import output
from shrike.cli.output import output_options


@click.command("export", short_help="Export the collection or a deck to an Anki package")
@output_options
@click.argument("dest", type=click.Path(dir_okay=False), required=False)
@click.option("--deck", default=None, help="Export only this deck (name, id, or #id).")
@click.option(
    "--note-id",
    "note_ids",
    multiple=True,
    type=int,
    help="Export only these notes (repeatable). Mutually exclusive with --deck.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["apkg", "colpkg"]),
    default=None,
    help="Package format. Default inferred from DEST's extension, else 'apkg'. "
    "A .colpkg is a whole-collection backup (no --deck/--note-id).",
)
@click.option(
    "--scheduling/--no-scheduling",
    "include_scheduling",
    default=False,
    help="Include review/scheduling data and deck options (default: off).",
)
@click.option(
    "--media/--no-media",
    "include_media",
    default=True,
    help="Bundle referenced media into the package (default: on).",
)
@click.option(
    "--server-path",
    default=None,
    help="Write the package to this path on the SERVER's disk (zero-copy; requires a "
    "purely-local daemon with a matching --export-path-root). Without it, the server "
    "writes a temp package and the CLI downloads it to DEST.",
)
@click.pass_context
def export(
    ctx: click.Context,
    dest: str | None,
    deck: str | None,
    note_ids: tuple[int, ...],
    fmt: str | None,
    include_scheduling: bool,
    include_media: bool,
    server_path: str | None,
) -> None:
    """Export the collection (or a deck/selection) to an Anki package at DEST.

    Writes a `.apkg` (shareable, scopable) or `.colpkg` (whole-collection backup).
    By default the server writes a temporary package and the CLI downloads it to
    DEST. With --server-path the server writes directly to a path on its own disk
    (no download), for a co-located operator.

    \b
    Examples:
      shrike collection export backup.colpkg
      shrike collection export spanish.apkg --deck Spanish
      shrike collection export deck.apkg --deck Spanish --scheduling
      shrike collection export --server-path /srv/exports/backup.colpkg
    """
    import os

    if deck and note_ids:
        raise click.UsageError("Use at most one of --deck or --note-id, not both.")
    if not dest and not server_path:
        raise click.UsageError("Provide a DEST to download to, or --server-path.")

    # Format: explicit flag, else inferred from the relevant path's extension,
    # else apkg. (A .colpkg DEST/server-path implies a whole-collection backup.)
    ref_path = server_path or dest or ""
    resolved_fmt = fmt or ("colpkg" if ref_path.endswith(".colpkg") else "apkg")
    if resolved_fmt == "colpkg" and (deck or note_ids):
        raise click.UsageError(
            "A .colpkg is a whole-collection backup — drop --deck/--note-id, or use --format apkg."
        )

    client = ctx.obj["client"]

    with output.spinner("Exporting"):
        result = client.export_package(
            deck=deck,
            note_ids=list(note_ids) or None,
            format=resolved_fmt,
            include_scheduling=include_scheduling,
            include_media=include_media,
            output_path=server_path,
        )

    if result.delivery == "path":
        # Server-local write — nothing to download; report where it landed.
        if ctx.obj["json"]:
            output.emit_json(
                {"note_count": result.note_count, "bytes": result.bytes, "path": result.path}
            )
            return
        # The destination path can contain brackets → escaped so it renders
        # literally rather than crashing on stray markup.
        output.console.print(
            f"[green]+[/green] Exported [green]{result.note_count}[/green] note(s) "
            f"-> [cyan]{output.esc(result.path)}[/cyan] ({result.bytes} bytes, on the server)"
        )
        return

    # Download delivery: stream the server's url to DEST.
    data = client.download_export(result.url)
    assert dest is not None  # guarded above (no dest only with --server-path → path delivery)
    with open(dest, "wb") as f:
        f.write(data)
    if ctx.obj["json"]:
        output.emit_json(
            {"note_count": result.note_count, "bytes": len(data), "path": os.path.abspath(dest)}
        )
        return
    output.console.print(
        f"[green]+[/green] Exported [green]{result.note_count}[/green] note(s) "
        f"-> [cyan]{output.esc(dest)}[/cyan] ({len(data)} bytes)"
    )
