"""``shrike import`` — import an Anki package (.apkg/.colpkg) into the collection (#72)."""

from __future__ import annotations

import os

import click

from shrike.cli import output
from shrike.cli.output import output_options
from shrike.schemas import ImportPackageResponse


@click.command("import", short_help="Import an Anki package (.apkg/.colpkg)")
@output_options
@click.argument("path")
@click.option(
    "--update-notes",
    type=click.Choice(["if_newer", "always", "never"]),
    default="if_newer",
    show_default=True,
    help="How to handle an imported note whose GUID matches an existing one: update "
    "only if newer (default), always, or never. New notes always add.",
)
@click.option(
    "--update-notetypes",
    type=click.Choice(["if_newer", "always", "never"]),
    default="if_newer",
    show_default=True,
    help="Same condition, applied to note types.",
)
@click.option(
    "--with-scheduling",
    is_flag=True,
    help="Import the package's review scheduling (due dates, intervals). Off by default.",
)
@click.option(
    "--merge-notetypes",
    is_flag=True,
    help="Merge imported note types into existing ones by name rather than adding new ones.",
)
@click.pass_context
def import_cmd(
    ctx: click.Context,
    path: str,
    update_notes: str,
    update_notetypes: str,
    with_scheduling: bool,
    merge_notetypes: bool,
) -> None:
    """Import an Anki package into the collection.

    PATH is read by the **server** from its own filesystem, so the operator must
    have enabled it with --import-path-root (on a purely-local daemon) containing
    the file — import overwrites collection data, so it uses its own root,
    separate from media's. A relative PATH is resolved against your current
    directory before being sent.

    By default a same-GUID note is updated only when the imported one is newer,
    and scheduling is not imported (Shrike manages cards, it does not review).

    \b
    Examples:
      shrike import ~/Downloads/shared-deck.apkg
      shrike import /srv/anki/backup.colpkg --with-scheduling
      shrike import deck.apkg --update-notes always
    """
    # Absolutize so a relative path isn't resolved against the (possibly
    # different) server cwd — the server still gates it against --import-path-root.
    abs_path = os.path.abspath(os.path.expanduser(path))
    client = ctx.obj["client"]
    with output.spinner("Importing package…"):
        result = client.import_package(
            abs_path,
            update_notes=update_notes,
            update_notetypes=update_notetypes,
            with_scheduling=with_scheduling,
            merge_notetypes=merge_notetypes,
        )

    if ctx.obj["json"]:
        output.emit_json(result)
        return
    _render(result)


def _render(result: ImportPackageResponse) -> None:
    output.console.print(
        f"[green]+[/green] Imported [green]{result.new}[/green] new, "
        f"[yellow]{result.updated}[/yellow] updated "
        f"([dim]{result.found_notes} notes in package[/dim])"
    )
    # Only surface the non-zero "needs attention" buckets — a clean import stays quiet.
    skipped = [
        ("duplicate", result.duplicate),
        ("conflicting", result.conflicting),
        ("first-field match", result.first_field_match),
        ("missing note type", result.missing_notetype),
        ("missing deck", result.missing_deck),
        ("empty first field", result.empty_first_field),
    ]
    for label, count in skipped:
        if count:
            output.console.print(f"  [dim]{label}:[/dim] {count}")
    if result.reindexed:
        output.console.print("  [dim]search index reconciled[/dim]")
