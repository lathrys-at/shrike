from __future__ import annotations

import json
import sys
from typing import Any

import click

from shrike.cli import output
from shrike.cli.output import output_options


def _parse_field(value: str) -> tuple[str, str]:
    """Parse a KEY=VALUE field argument."""
    if "=" not in value:
        raise click.BadParameter(
            f"Invalid field format: {value!r}. Use KEY=VALUE (e.g., -f Front='What is X?')"
        )
    key, _, val = value.partition("=")
    return key.strip(), val.strip()


def _parse_comma_separated(
    ctx: click.Context,
    param: click.Parameter,
    value: tuple[str, ...],
) -> tuple[str, ...]:
    """Split comma-separated values so --tags a,b and --tags a --tags b both work."""
    result: list[str] = []
    for v in value:
        result.extend(part.strip() for part in v.split(",") if part.strip())
    return tuple(result)


@click.group("note", short_help="Manage notes")
def note() -> None:
    """Create, list, update, search, and delete notes."""


@note.command("list", short_help="List notes by filters")
@output_options
@click.option("--deck", help="Filter by deck name.")
@click.option(
    "--tags",
    multiple=True,
    callback=_parse_comma_separated,
    expose_value=True,
    help="Filter by tag (repeatable, comma-separated, ANDed).",
)
@click.option("--type", "note_type", help="Filter by note type.")
@click.option("--ids", multiple=True, type=int, help="Fetch specific note IDs.")
@click.option("--since", "modified_since", help="Notes modified after this date (ISO 8601).")
@click.option("--query", help="Raw Anki search query.")
@click.option("--meta", is_flag=True, help="Show only metadata, not field content.")
@click.option("--limit", type=int, default=50, help="Max notes to return (default: 50).")
@click.pass_context
def note_list(
    ctx: click.Context,
    deck: str | None,
    tags: tuple[str, ...],
    note_type: str | None,
    ids: tuple[int, ...],
    modified_since: str | None,
    query: str | None,
    meta: bool,
    limit: int,
) -> None:
    """List notes matching structured filters.

    At least one filter is required. Use --meta for compact output.

    \b
    Examples:
      shrike note list --deck "Japanese::Vocabulary"
      shrike note list --tags verb,chapter-3
      shrike note list --type Cloze --meta --limit 20
    """
    client = ctx.obj["client"]

    kwargs: dict[str, Any] = {
        "deck": deck,
        "tags": list(tags) or None,
        "note_type": note_type,
        "ids": list(ids) or None,
        "modified_since": modified_since,
        "query": query,
        "fields": "meta" if meta else "full",
        "limit": limit,
    }

    result = client.list_notes(**kwargs)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    notes = result.get("notes", [])
    total = result.get("total", len(notes))

    if not notes:
        output.console.print("[dim]No notes found.[/dim]")
        return

    if total > len(notes):
        output.console.print(f"  [dim]Showing {len(notes)} of {total} matching notes[/dim]")
    else:
        output.console.print(f"  [dim]{total} note(s)[/dim]")

    output.console.print()

    if meta or not any(n.get("content") for n in notes):
        rows = [output.note_summary_row(n) for n in notes]
        output.table(["ID", "Type", "Deck", "Tags", "Modified"], rows)
    else:
        for n in notes:
            output.note_detail(n)

    output.console.print()


@note.command("show", short_help="Show a note by ID")
@output_options
@click.argument("note_id", type=int)
@click.pass_context
def note_show(ctx: click.Context, note_id: int) -> None:
    """Show the full content of a specific note."""
    client = ctx.obj["client"]
    result = client.list_notes(ids=[note_id], fields="full")

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    notes = result.get("notes", [])
    if not notes:
        raise click.ClickException(f"Note {note_id} not found.")

    output.note_detail(notes[0])


@note.command("create", short_help="Create a new note")
@output_options
@click.option("--deck", help="Target deck.")
@click.option("--type", "note_type", help="Note type (e.g., Basic, Cloze).")
@click.option(
    "-f",
    "--field",
    multiple=True,
    metavar="KEY=VALUE",
    help="Field value (repeatable). E.g., -f Front='Question' -f Back='Answer'",
)
@click.option(
    "--tags",
    multiple=True,
    callback=_parse_comma_separated,
    expose_value=True,
    help="Tags for the note (repeatable, comma-separated).",
)
@click.option(
    "--json-input",
    is_flag=True,
    help="Read a JSON array of note objects from stdin.",
)
@click.pass_context
def note_create(
    ctx: click.Context,
    deck: str | None,
    note_type: str | None,
    field: tuple[str, ...],
    tags: tuple[str, ...],
    json_input: bool,
) -> None:
    """Create one or more new notes.

    \b
    Inline:
      shrike note create --deck Test --type Basic -f Front="Q" -f Back="A"

    \b
    Bulk (from JSON on stdin):
      echo '[{"deck":"Test","note_type":"Basic","fields":{"Front":"Q","Back":"A"}}]' | \\
        shrike note create --json-input
    """
    client = ctx.obj["client"]

    if json_input:
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON input: {e}") from e
        if not isinstance(data, list):
            data = [data]
        notes = data
    else:
        if not deck:
            raise click.ClickException("--deck is required for inline creation.")
        if not note_type:
            raise click.ClickException("--type is required for inline creation.")
        if not field:
            raise click.ClickException("At least one field is required. Use -f KEY=VALUE.")

        fields = dict(_parse_field(f) for f in field)
        note_obj: dict[str, Any] = {
            "deck": deck,
            "note_type": note_type,
            "fields": fields,
        }
        if tags:
            note_obj["tags"] = list(tags)
        notes = [note_obj]

    result = client.upsert_notes(notes)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    output.result_status(result.get("results", []))


@note.command("update", short_help="Update an existing note")
@output_options
@click.argument("note_id", type=int)
@click.option(
    "-f",
    "--field",
    multiple=True,
    metavar="KEY=VALUE",
    help="Field to update (repeatable).",
)
@click.option(
    "--tags",
    multiple=True,
    callback=_parse_comma_separated,
    expose_value=True,
    help="Replace all tags (repeatable, comma-separated).",
)
@click.option("--deck", help="Move note to this deck.")
@click.pass_context
def note_update(
    ctx: click.Context,
    note_id: int,
    field: tuple[str, ...],
    tags: tuple[str, ...],
    deck: str | None,
) -> None:
    """Update an existing note by ID.

    Only specified fields are changed; unspecified fields are left as-is.
    Tags are fully replaced if --tags is provided.

    \b
    Examples:
      shrike note update 170000123 -f Back="New answer"
      shrike note update 170000123 --tags newtag,kept-tag
      shrike note update 170000123 --deck "Other::Deck"
    """
    client = ctx.obj["client"]

    note_obj: dict[str, Any] = {"id": note_id}
    if field:
        note_obj["fields"] = dict(_parse_field(f) for f in field)
    if tags:
        note_obj["tags"] = list(tags)
    if deck:
        note_obj["deck"] = deck

    if len(note_obj) == 1:
        raise click.ClickException("Nothing to update. Use -f, --tags, or --deck.")

    result = client.upsert_notes([note_obj])

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    output.result_status(result.get("results", []))


@note.command("delete", short_help="Delete notes by ID")
@output_options
@click.argument("note_ids", type=int, nargs=-1, required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
@click.pass_context
def note_delete(ctx: click.Context, note_ids: tuple[int, ...], yes: bool) -> None:
    """Permanently delete notes and their cards.

    \b
    Example:
      shrike note delete 170000123 170000456
      shrike note delete 170000123 --yes
    """
    if not yes:
        ids_str = ", ".join(str(i) for i in note_ids)
        if not click.confirm(
            f"Delete {len(note_ids)} note(s)? IDs: {ids_str}\nThis cannot be undone"
        ):
            output.console.print("Cancelled.")
            return

    client = ctx.obj["client"]
    result = client.delete_notes(list(note_ids))

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    deleted = result.get("deleted", [])
    not_found = result.get("not_found", [])

    if deleted:
        output.success(f"Deleted {len(deleted)} note(s).")
    if not_found:
        output.console.print(f"[dim]Not found: {', '.join(str(i) for i in not_found)}[/dim]")


@note.command("search", short_help="Semantic search over notes")
@output_options
@click.argument("queries", nargs=-1)
@click.option(
    "--similar-to",
    multiple=True,
    type=int,
    metavar="ID",
    help="Find notes similar to this note ID.",
)
@click.option("--top-k", type=int, default=10, help="Results per query (default: 10).")
@click.option("--deck", help="Restrict search to this deck.")
@click.option(
    "--tags",
    multiple=True,
    callback=_parse_comma_separated,
    expose_value=True,
    help="Restrict search to notes with these tags.",
)
@click.pass_context
def note_search(
    ctx: click.Context,
    queries: tuple[str, ...],
    similar_to: tuple[int, ...],
    top_k: int,
    deck: str | None,
    tags: tuple[str, ...],
) -> None:
    """Semantic similarity search over the collection.

    \b
    Examples:
      shrike note search "electron transport chain"
      shrike note search --similar-to 170000123
      shrike note search "mitochondria" --deck Biochemistry
    """
    if not queries and not similar_to:
        raise click.ClickException("Provide query strings and/or --similar-to note IDs.")

    client = ctx.obj["client"]

    kwargs: dict[str, Any] = {"top_k": top_k}
    if queries:
        kwargs["queries"] = list(queries)
    if similar_to:
        kwargs["ids"] = list(similar_to)
    if deck:
        kwargs["deck"] = deck
    if tags:
        kwargs["tags"] = list(tags)

    result = client.search_notes(**kwargs)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    message = result.get("_message")
    if message:
        output.console.print(f"  [dim]{message}[/dim]")
        return

    results = result.get("results", [])
    if not results:
        output.console.print("[dim]No results.[/dim]")
        return

    for group in results:
        source = group.get("source", "")
        output.console.print(f"\n  [bold]Results for:[/bold] {source}")
        matches = group.get("matches", [])
        for m in matches:
            score = m.get("score", 0)
            output.console.print(
                f"    \\[[cyan]{score:.2f}[/cyan]] [cyan]{m['id']}[/cyan] ({m.get('deck', '')})"
            )
            content = m.get("content", {})
            if content:
                first_field = next(iter(content.values()), "")
                if len(first_field) > 80:
                    first_field = first_field[:77] + "..."
                output.console.print(f"      {first_field}")

    output.console.print()
