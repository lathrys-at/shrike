from __future__ import annotations

import click

from shrike.cli import output
from shrike.cli.output import output_options


@click.command("info", short_help="Show collection structure and stats")
@output_options
@click.option("--types", "show_types", is_flag=True, help="Show only note types.")
@click.option("--decks", "show_decks", is_flag=True, help="Show only decks.")
@click.option("--tags", "show_tags", is_flag=True, help="Show only tags.")
@click.option("--stats", "show_stats", is_flag=True, help="Show only stats.")
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
    """Show the structure of the Anki collection.

    Displays note types, decks, tags, and scheduling statistics.
    Use flags to show only specific sections.
    """
    client = ctx.obj["client"]

    # Build include list
    include = []
    if show_types:
        include.append("note_types")
    if show_decks:
        include.append("decks")
    if show_tags:
        include.append("tags")
    if show_stats:
        include.append("stats")
    if type_details and not show_types:
        include.append("note_types")

    result = client.collection_info(
        include=include or None,
        note_type_details=list(type_details) or None,
    )

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    sections: list[bool] = []

    # Note types
    note_types = result.get("note_types")
    if note_types is not None:
        rows = [
            [nt["name"], nt.get("type", "standard"), ", ".join(nt.get("fields", []))]
            for nt in note_types
        ]
        output.table(["Note Type", "Kind", "Fields"], rows)
        sections.append(True)

        for nt in note_types:
            if nt.get("templates") is not None or nt.get("css") is not None:
                output.note_type_detail(nt)

    # Decks
    decks = result.get("decks")
    if decks is not None:
        if sections:
            output.console.print()
        rows = [[d["name"], str(d.get("note_count", 0))] for d in decks]
        output.table(["Deck", "Notes"], rows)
        sections.append(True)

    # Tags
    tags = result.get("tags")
    if tags is not None:
        if sections:
            output.console.print()
        if tags:
            styled = " ".join(f"[yellow]{t}[/yellow]" for t in tags)
            output.console.print(f"  [bold]Tags[/bold]  {styled}")
        else:
            output.console.print("  [bold]Tags[/bold]  [dim](none)[/dim]")
        sections.append(True)

    # Stats
    stats = result.get("stats")
    if stats is not None:
        if sections:
            output.console.print()
        rows = [
            ["Notes", str(stats.get("total_notes", 0))],
            ["Cards", str(stats.get("total_cards", 0))],
            ["Due today", str(stats.get("cards_due_today", 0))],
            ["New", str(stats.get("new_cards", 0))],
        ]
        output.table(["Stat", "Count"], rows)
