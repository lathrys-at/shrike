from __future__ import annotations

import click

from shrike.cli import output
from shrike.cli.groups import OrderedGroup
from shrike.cli.output import NOTE_ID, output_options


@click.group("tag", cls=OrderedGroup, short_help="Manage tags across the collection")
def tag() -> None:
    """Rename tags across the collection.

    Note-level tag editing (set/add/remove on specific notes) lives under
    'shrike note tag'. These commands act on the collection's tag taxonomy.
    """


@tag.command("rename", short_help="Rename a tag")
@output_options
@click.argument("old")
@click.argument("new")
@click.option(
    "--note",
    "note_ids",
    type=NOTE_ID,
    multiple=True,
    help="Restrict the rename to these note IDs (repeatable). "
    "Omit to rename the tag across the whole collection.",
)
@click.pass_context
def tag_rename(ctx: click.Context, old: str, new: str, note_ids: tuple[int, ...]) -> None:
    """Rename a tag, collection-wide or on specific notes.

    With no --note, the tag is renamed everywhere it appears, children included
    (renaming "history" also moves "history::ww2"). With --note, only those
    notes are affected and the tag is matched exactly — renaming "jp" never
    touches "jp-verbs".

    \b
    Examples:
      shrike collection tag rename history::ww2 history::wwii
      shrike collection tag rename ww2 wwii --note 170000123 --note 170000456
    """
    client = ctx.obj["client"]
    with output.spinner("Renaming tag…"):
        result = client.rename_tag(old, new, list(note_ids) or None)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    # Tag names can contain brackets → escaped so they render literally.
    output.console.print(
        f"Renamed [yellow]{output.esc(old)}[/yellow] → [yellow]{output.esc(new)}[/yellow] "
        f"on {result.notes_modified} note(s)."
    )
