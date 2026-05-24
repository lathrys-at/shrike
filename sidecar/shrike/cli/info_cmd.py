from __future__ import annotations

import click

from shrike.cli import output


@click.command("info", short_help="Show collection structure and stats")
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
def info(ctx, show_types, show_decks, show_tags, show_stats, type_details):
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

    # Note types
    note_types = result.get("note_types")
    if note_types is not None:
        output.section("Note Types")
        rows = []
        for nt in note_types:
            rows.append([
                nt["name"],
                nt.get("type", "standard"),
                ", ".join(nt.get("fields", [])),
            ])
        output.table(["Name", "Type", "Fields"], rows)

        # Show details for requested types
        for nt in note_types:
            if nt.get("templates") is not None or nt.get("css") is not None:
                output.note_type_detail(nt)

    # Decks
    decks = result.get("decks")
    if decks is not None:
        output.section("Decks")
        rows = [
            [d["name"], str(d.get("note_count", 0))]
            for d in decks
        ]
        output.table(["Name", "Notes"], rows)

    # Tags
    tags = result.get("tags")
    if tags is not None:
        output.section("Tags")
        if tags:
            styled = " ".join(click.style(t, **output.TAG_STYLE) for t in tags)
            click.echo(f"  {styled}")
        else:
            click.echo(click.style("  (none)", dim=True))

    # Stats
    stats = result.get("stats")
    if stats is not None:
        output.section("Stats")
        output.kv("Total notes", stats.get("total_notes", 0))
        output.kv("Total cards", stats.get("total_cards", 0))
        output.kv("Cards due today", stats.get("cards_due_today", 0))
        output.kv("New cards", stats.get("new_cards", 0))

    click.echo()
