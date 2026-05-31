from __future__ import annotations

import json
import sys
from typing import Any

import click

from shrike.cli import output
from shrike.cli.config import resolve_collection
from shrike.cli.output import NOTE_ID, output_options


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
@click.option("--ids", multiple=True, type=NOTE_ID, help="Fetch specific note IDs.")
@click.option("--since", "modified_since", help="Notes modified after this date (ISO 8601).")
@click.option("--query", help="Raw Anki search query.")
@click.option("--brief", is_flag=True, help="Show only IDs and metadata, not field content.")
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
    brief: bool,
    limit: int,
) -> None:
    """List notes matching structured filters.

    At least one filter is required. Use --brief for compact output.

    \b
    Examples:
      shrike note list --deck "Japanese::Vocabulary"
      shrike note list --tags verb,chapter-3
      shrike note list --type Cloze --meta --limit 20
    """
    client = ctx.obj["client"]

    if not any([deck, tags, note_type, ids, modified_since, query]):
        raise click.UsageError(
            "At least one filter is required: --deck, --tags, --type, --ids, --since, or --query"
        )

    kwargs: dict[str, Any] = {
        "deck": deck,
        "tags": list(tags) or None,
        "note_type": note_type,
        "ids": list(ids) or None,
        "modified_since": modified_since,
        "query": query,
        "fields": "meta" if brief else "full",
        "limit": limit,
    }

    with output.spinner("Fetching notes…"):
        result = client.list_notes(**kwargs)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    notes = result.notes
    total = result.total

    if not notes:
        output.console.print("[dim]No notes found.[/dim]")
        return

    col_path = resolve_collection(ctx.obj["config"]) or "collection"

    filter_parts: list[str] = []
    if deck:
        filter_parts.append(f"in [cyan]{deck}[/cyan]")
    if note_type:
        filter_parts.append(f"of type [cyan]{note_type}[/cyan]")
    if tags:
        filter_parts.append(f"tagged {', '.join(f'[yellow]{t}[/yellow]' for t in tags)}")
    filter_desc = " ".join(filter_parts)
    if filter_desc:
        filter_desc = f" {filter_desc}"

    count = f"{len(notes)} of {total}" if total > len(notes) else str(total)
    output.console.print(
        f"[dim]Showing {count} note(s){filter_desc} from [cyan]{col_path}[/cyan][/dim]"
    )

    output.console.print()

    if brief or not any(n.content for n in notes):
        rows = [output.note_summary_row(n) for n in notes]
        output.table(["ID", "Type", "Deck", "Tags", "Modified"], rows)
    else:
        for n in notes:
            output.note_detail(n)

    output.console.print()


@note.command("show", short_help="Show a note by ID")
@output_options
@click.argument("note_id", type=NOTE_ID)
@click.pass_context
def note_show(ctx: click.Context, note_id: int) -> None:
    """Shorthand for ``note list --ids ID``.

    Errors if the note does not exist.
    """
    client = ctx.obj["client"]
    with output.spinner("Fetching note…"):
        result = client.list_notes(ids=[note_id], fields="full")

    notes = result.notes
    if not notes:
        raise click.ClickException(f"Note #{note_id} not found.")

    if ctx.obj["json"]:
        output.emit_json(result)
        return

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
            raise click.UsageError("--deck is required for inline creation.")
        if not note_type:
            raise click.UsageError("--type is required for inline creation.")
        if not field:
            raise click.UsageError("At least one field is required. Use -f KEY=VALUE.")

        fields = dict(_parse_field(f) for f in field)
        note_obj: dict[str, Any] = {
            "deck": deck,
            "note_type": note_type,
            "fields": fields,
        }
        if tags:
            note_obj["tags"] = list(tags)
        notes = [note_obj]

    with output.spinner("Creating notes…"):
        result = client.upsert_notes(notes)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    output.result_status(result.results)


@note.command("update", short_help="Update an existing note")
@output_options
@click.argument("note_id", type=NOTE_ID)
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
        raise click.UsageError("Nothing to update. Use -f, --tags, or --deck.")

    with output.spinner("Updating note…"):
        result = client.upsert_notes([note_obj])

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    output.result_status(result.results)


@note.command("tag", short_help="Replace tags on one or more notes")
@output_options
@click.argument("note_ids", type=NOTE_ID, nargs=-1, required=True)
@click.option(
    "--set",
    "set_tags",
    multiple=True,
    required=True,
    callback=_parse_comma_separated,
    expose_value=True,
    help="New tag set, replacing existing tags (repeatable, comma-separated). "
    'Pass --set "" to clear all tags.',
)
@click.pass_context
def note_tag(ctx: click.Context, note_ids: tuple[int, ...], set_tags: tuple[str, ...]) -> None:
    """Replace the tags on each given note with the same new set.

    Tags are fully replaced, not merged — the notes end up with exactly the tags
    you pass (and nothing else). Fields and decks are untouched.

    \b
    Examples:
      shrike note tag 170000123 --set world-war-2,history
      shrike note tag 170000123 170000456 --set needs-review
      shrike note tag 170000123 --set ""        # clear all tags
    """
    client = ctx.obj["client"]
    tags = list(set_tags)
    notes = [{"id": nid, "tags": tags} for nid in note_ids]

    with output.spinner("Tagging notes…"):
        result = client.upsert_notes(notes)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    output.result_status(result.results)


@note.command("delete", short_help="Delete notes by ID")
@output_options
@click.argument("note_ids", type=NOTE_ID, nargs=-1, required=True)
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
    with output.spinner("Deleting notes…"):
        result = client.delete_notes(list(note_ids))

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    deleted = result.deleted
    not_found = result.not_found

    if deleted:
        ids_str = ", ".join(f"[green]#{i}[/green]" for i in deleted)
        output.console.print(f"Deleted {len(deleted)} note(s): {ids_str}")
    if not_found:
        ids_str = ", ".join(str(i) for i in not_found)
        output.console.print(f"[dim]Not found: {ids_str}[/dim]")


@note.command("search", short_help="Semantic search over notes")
@output_options
@click.argument("queries", nargs=-1)
@click.option(
    "--similar-to",
    multiple=True,
    type=NOTE_ID,
    metavar="ID",
    help="Find notes similar to this note ID.",
)
@click.option("--top-k", type=int, default=10, help="Results per query (default: 10).")
@click.option(
    "--threshold", type=float, default=0.5, help="Minimum similarity score (default: 0.5)."
)
@click.option("--deck", help="Restrict search to this deck.")
@click.option(
    "--tags",
    multiple=True,
    callback=_parse_comma_separated,
    expose_value=True,
    help="Restrict search to notes with these tags.",
)
@click.option("--brief", is_flag=True, help="Show only IDs and scores, not full note content.")
@click.pass_context
def note_search(
    ctx: click.Context,
    queries: tuple[str, ...],
    similar_to: tuple[int, ...],
    top_k: int,
    threshold: float,
    deck: str | None,
    tags: tuple[str, ...],
    brief: bool,
) -> None:
    """Semantic similarity search over the collection.

    \b
    Examples:
      shrike note search "electron transport chain"
      shrike note search --similar-to 170000123
      shrike note search "mitochondria" --deck Biochemistry
    """
    if not queries and not similar_to:
        raise click.UsageError("Provide query strings and/or --similar-to note IDs.")

    client = ctx.obj["client"]

    kwargs: dict[str, Any] = {"top_k": top_k, "threshold": threshold}
    if queries:
        kwargs["queries"] = list(queries)
    if similar_to:
        kwargs["ids"] = list(similar_to)
    if deck:
        kwargs["deck"] = deck
    if tags:
        kwargs["tags"] = list(tags)

    with output.spinner("Searching notes…"):
        result = client.search_notes(**kwargs)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    if result.message:
        output.console.print(f"[dim]{result.message}[/dim]")
        return

    if not result.results:
        output.console.print("[dim]No results.[/dim]")
        return

    for group in result.results:
        output.console.print(f"\nResults for: [cyan]{group.source}[/cyan]")
        for m in group.matches:
            if brief:
                output.console.print(
                    f"  \\[{m.score:.2f}] [green]#{m.id}[/green] ([cyan]{m.deck}[/cyan])"
                )
            else:
                output.note_detail(m, subtitle=f"[{m.score:.2f}]")

    output.console.print()
