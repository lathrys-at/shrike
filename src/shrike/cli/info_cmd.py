from __future__ import annotations

import click

from shrike.cli import output
from shrike.cli.output import output_options
from shrike.schemas import DeckInfo, NoteTypeInfo, Stats, Summary


@click.command("info", short_help="Show collection summary")
@output_options
@click.option("--types", "show_types", is_flag=True, help="List note types with fields.")
@click.option("--decks", "show_decks", is_flag=True, help="List decks with note counts.")
@click.option("--tags", "show_tags", is_flag=True, help="List all tags.")
@click.option("--stats", "show_stats", is_flag=True, help="Show scheduling statistics.")
@click.option(
    "--type-details",
    multiple=True,
    metavar="NAME",
    help="Show full templates and CSS for a note type.",
)
@click.pass_context
def info(
    ctx: click.Context,
    show_types: bool,
    show_decks: bool,
    show_tags: bool,
    show_stats: bool,
    type_details: tuple[str, ...],
) -> None:
    """Show collection information.

    Without flags, prints a compact summary. Use flags to see details.

    \b
    Examples:
      shrike info
      shrike info --types
      shrike info --decks --stats
      shrike info --type-details Basic
    """
    client = ctx.obj["client"]
    has_detail_flags = show_types or show_decks or show_tags or show_stats or type_details

    include: list[str] = ["summary"]
    if has_detail_flags:
        if show_types or type_details:
            include.append("note_types")
        if show_decks:
            include.append("decks")
        if show_tags:
            include.append("tags")
        if show_stats:
            include.append("stats")

    with output.spinner("Fetching collection info…"):
        result = client.collection_info(
            include=include,
            note_type_details=list(type_details) or None,
        )

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    col_path = result.summary.path if result.summary else "collection"

    if not has_detail_flags:
        if result.summary:
            _render_summary(result.summary)
        return

    printed = False
    if result.note_types is not None:
        _render_note_types(result.note_types, col_path)
        printed = True
    if result.decks is not None:
        if printed:
            output.console.print()
        _render_decks(result.decks, col_path)
        printed = True
    if result.tags is not None:
        if printed:
            output.console.print()
        _render_tags(result.tags, col_path)
        printed = True
    if result.stats is not None:
        if printed:
            output.console.print()
        _render_stats(result.stats, col_path)


def _render_summary(summary: Summary) -> None:
    # The collection path is escaped (kv() does not escape its value).
    output.kv("Collection", f"[cyan]{output.esc(summary.path)}[/cyan]")
    output.kv("Created", summary.created)
    output.kv("Modified", summary.modified)
    output.kv("Notes", summary.notes)
    output.kv("Cards", summary.cards)
    output.kv("Decks", summary.decks)
    output.kv("Note types", summary.note_types)
    output.kv("Tags", summary.tags)
    output.kv("Due today", summary.due_today)


def _render_note_types(note_types: list[NoteTypeInfo], col_path: str) -> None:
    output.note_type_table(note_types, col_path)

    for nt in note_types:
        if nt.detail is not None:
            output.console.print()
            output.note_type_detail(nt)


def _render_decks(decks: list[DeckInfo], col_path: str) -> None:
    # Deck names + collection path are collection-authored → escaped.
    count = len(decks)
    output.console.print(f"Showing {count} decks in [cyan]{output.esc(col_path)}[/cyan]")
    output.console.print()
    rows = [[f"[cyan]{output.esc(d.name)}[/cyan]", str(d.note_count)] for d in decks]
    output.table(["Name", "Notes"], rows)


def _render_tags(tags: list[str], col_path: str) -> None:
    # Tag names + collection path are collection-authored → escaped.
    count = len(tags)
    output.console.print(f"Showing {count} tags in [cyan]{output.esc(col_path)}[/cyan]")
    output.console.print()
    rows = [[f"[yellow]{output.esc(t)}[/yellow]"] for t in sorted(tags)]
    output.table(["Name"], rows)


def _render_stats(stats: Stats, col_path: str) -> None:
    output.console.print(f"Showing statistics for [cyan]{output.esc(col_path)}[/cyan]")
    output.console.print()
    output.kv("Notes", stats.total_notes)
    output.kv("Cards", stats.total_cards)
    output.kv("Due today", stats.cards_due_today)
    output.kv("New cards", stats.new_cards)

    if stats.decks_summary:
        output.console.print()
        # Deck names are collection-authored → escaped.
        rows = [
            [f"[cyan]{output.esc(name)}[/cyan]", str(d.notes), str(d.due)]
            for name, d in stats.decks_summary.items()
        ]
        output.table(["Deck", "Notes", "Due"], rows)
