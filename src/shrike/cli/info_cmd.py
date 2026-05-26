from __future__ import annotations

from typing import Any

import click

from shrike.cli import output
from shrike.cli.output import output_options


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

    summary = result.get("summary", {})
    col_path = summary.get("path", "collection")

    if not has_detail_flags:
        _render_summary(summary)
        return

    printed = False
    if "note_types" in result:
        _render_note_types(result["note_types"], col_path)
        printed = True
    if "decks" in result:
        if printed:
            output.console.print()
        _render_decks(result["decks"], col_path)
        printed = True
    if "tags" in result:
        if printed:
            output.console.print()
        _render_tags(result["tags"], col_path)
        printed = True
    if "stats" in result:
        if printed:
            output.console.print()
        _render_stats(result["stats"], col_path)


def _render_summary(summary: dict[str, Any]) -> None:
    output.kv("Collection", f"[cyan]{summary.get('path', '')}[/cyan]")
    output.kv("Created", summary.get("created", ""))
    output.kv("Modified", summary.get("modified", ""))
    output.kv("Notes", summary.get("notes", 0))
    output.kv("Cards", summary.get("cards", 0))
    output.kv("Decks", summary.get("decks", 0))
    output.kv("Note types", summary.get("note_types", 0))
    output.kv("Tags", summary.get("tags", 0))
    output.kv("Due today", summary.get("due_today", 0))


def _render_note_types(note_types: list[dict[str, Any]], col_path: str) -> None:
    output.note_type_table(note_types, col_path)

    for nt in note_types:
        if nt.get("templates") is not None or nt.get("css") is not None:
            output.console.print()
            output.note_type_detail(nt)


def _render_decks(decks: list[dict[str, Any]], col_path: str) -> None:
    count = len(decks)
    output.console.print(f"Showing {count} decks in [cyan]{col_path}[/cyan]")
    output.console.print()
    rows = [[f"[cyan]{d['name']}[/cyan]", str(d.get("note_count", 0))] for d in decks]
    output.table(["Name", "Notes"], rows)


def _render_tags(tags: list[str], col_path: str) -> None:
    count = len(tags)
    output.console.print(f"Showing {count} tags in [cyan]{col_path}[/cyan]")
    output.console.print()
    rows = [[f"[yellow]{t}[/yellow]"] for t in sorted(tags)]
    output.table(["Name"], rows)


def _render_stats(stats: dict[str, Any], col_path: str) -> None:
    output.console.print(f"Showing statistics for [cyan]{col_path}[/cyan]")
    output.console.print()
    output.kv("Notes", stats.get("total_notes", 0))
    output.kv("Cards", stats.get("total_cards", 0))
    output.kv("Due today", stats.get("cards_due_today", 0))
    output.kv("New cards", stats.get("new_cards", 0))

    decks_summary = stats.get("decks_summary", {})
    if decks_summary:
        output.console.print()
        rows = [
            [f"[cyan]{name}[/cyan]", str(d.get("notes", 0)), str(d.get("due", 0))]
            for name, d in decks_summary.items()
        ]
        output.table(["Deck", "Notes", "Due"], rows)
