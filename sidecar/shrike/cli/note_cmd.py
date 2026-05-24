from __future__ import annotations

import json
import sys

import click

from shrike.cli import output
from shrike.cli.client import ServerError


def _parse_field(value: str) -> tuple[str, str]:
    """Parse a KEY=VALUE field argument."""
    if "=" not in value:
        raise click.BadParameter(
            f"Invalid field format: {value!r}. Use KEY=VALUE (e.g., -f Front='What is X?')"
        )
    key, _, val = value.partition("=")
    return key.strip(), val.strip()


@click.group("note", short_help="Manage notes")
def note():
    """Create, list, update, search, and delete notes."""
    pass


@note.command("list", short_help="List notes by filters")
@click.option("--deck", help="Filter by deck name.")
@click.option("--tags", multiple=True, help="Filter by tag (repeatable, ANDed).")
@click.option("--type", "note_type", help="Filter by note type.")
@click.option("--ids", multiple=True, type=int, help="Fetch specific note IDs.")
@click.option("--since", "modified_since", help="Notes modified after this date (ISO 8601).")
@click.option("--query", help="Raw Anki search query.")
@click.option("--meta", is_flag=True, help="Show only metadata, not field content.")
@click.option("--limit", type=int, default=50, help="Max notes to return (default: 50).")
@click.pass_context
def note_list(ctx, deck, tags, note_type, ids, modified_since, query, meta, limit):
    """List notes matching structured filters.

    At least one filter is required. Use --meta for compact output.

    \b
    Examples:
      shrike note list --deck "Japanese::Vocabulary"
      shrike note list --tags verb --tags chapter-3
      shrike note list --type Cloze --meta --limit 20
    """
    client = ctx.obj["client"]

    kwargs = {
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
        click.echo(click.style("No notes found.", dim=True))
        return

    # Summary header
    if total > len(notes):
        click.echo(click.style(
            f"  Showing {len(notes)} of {total} matching notes", dim=True
        ))
    else:
        click.echo(click.style(f"  {total} note(s)", dim=True))

    click.echo()

    if meta or not any(n.get("content") for n in notes):
        # Table view
        rows = [output.note_summary_row(n) for n in notes]
        output.table(["ID", "Type", "Deck", "Tags", "Modified"], rows)
    else:
        # Detail view for full results
        for n in notes:
            output.note_detail(n)

    click.echo()


@note.command("show", short_help="Show a note by ID")
@click.argument("note_id", type=int)
@click.pass_context
def note_show(ctx, note_id):
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
@click.option("--deck", help="Target deck.")
@click.option("--type", "note_type", help="Note type (e.g., Basic, Cloze).")
@click.option(
    "-f", "--field",
    multiple=True,
    metavar="KEY=VALUE",
    help="Field value (repeatable). E.g., -f Front='Question' -f Back='Answer'",
)
@click.option("--tags", multiple=True, help="Tags for the note (repeatable).")
@click.option(
    "--json-input", is_flag=True,
    help="Read a JSON array of note objects from stdin.",
)
@click.pass_context
def note_create(ctx, deck, note_type, field, tags, json_input):
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
            raise click.ClickException(f"Invalid JSON input: {e}")
        if not isinstance(data, list):
            data = [data]
        notes = data
    else:
        if not deck:
            raise click.ClickException("--deck is required for inline creation.")
        if not note_type:
            raise click.ClickException("--type is required for inline creation.")
        if not field:
            raise click.ClickException(
                "At least one field is required. Use -f KEY=VALUE."
            )

        fields = dict(_parse_field(f) for f in field)
        note_obj: dict = {
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
@click.argument("note_id", type=int)
@click.option(
    "-f", "--field",
    multiple=True,
    metavar="KEY=VALUE",
    help="Field to update (repeatable).",
)
@click.option("--tags", multiple=True, help="Replace all tags (repeatable).")
@click.option("--deck", help="Move note to this deck.")
@click.pass_context
def note_update(ctx, note_id, field, tags, deck):
    """Update an existing note by ID.

    Only specified fields are changed; unspecified fields are left as-is.
    Tags are fully replaced if --tags is provided.

    \b
    Examples:
      shrike note update 170000123 -f Back="New answer"
      shrike note update 170000123 --tags newtag --tags kept-tag
      shrike note update 170000123 --deck "Other::Deck"
    """
    client = ctx.obj["client"]

    note_obj: dict = {"id": note_id}
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
@click.argument("note_ids", type=int, nargs=-1, required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
@click.pass_context
def note_delete(ctx, note_ids, yes):
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
            click.echo("Cancelled.")
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
        click.echo(click.style(
            f"Not found: {', '.join(str(i) for i in not_found)}", dim=True
        ))


@note.command("search", short_help="Semantic search over notes")
@click.argument("queries", nargs=-1)
@click.option(
    "--similar-to", multiple=True, type=int, metavar="ID",
    help="Find notes similar to this note ID.",
)
@click.option("--top-k", type=int, default=10, help="Results per query (default: 10).")
@click.option("--deck", help="Restrict search to this deck.")
@click.option("--tags", multiple=True, help="Restrict search to notes with these tags.")
@click.pass_context
def note_search(ctx, queries, similar_to, top_k, deck, tags):
    """Semantic similarity search over the collection.

    \b
    Examples:
      shrike note search "electron transport chain"
      shrike note search --similar-to 170000123
      shrike note search "mitochondria" --deck Biochemistry
    """
    if not queries and not similar_to:
        raise click.ClickException(
            "Provide query strings and/or --similar-to note IDs."
        )

    client = ctx.obj["client"]

    kwargs: dict = {"top_k": top_k}
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
        click.echo(click.style(f"  {message}", dim=True))
        return

    results = result.get("results", [])
    if not results:
        click.echo(click.style("No results.", dim=True))
        return

    for group in results:
        source = group.get("source", "")
        click.echo(f"\n  {click.style('Results for:', bold=True)} {source}")
        matches = group.get("matches", [])
        for m in matches:
            score = m.get("score", 0)
            score_str = click.style(f"{score:.2f}", fg="cyan")
            click.echo(f"    [{score_str}] ", nl=False)
            click.echo(
                f"{click.style(str(m['id']), **output.ID_STYLE)} "
                f"({m.get('deck', '')}) "
            )
            content = m.get("content", {})
            if content:
                first_field = next(iter(content.values()), "")
                if len(first_field) > 80:
                    first_field = first_field[:77] + "..."
                click.echo(f"      {first_field}")

    click.echo()
