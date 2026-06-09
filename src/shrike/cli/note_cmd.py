from __future__ import annotations

import json
import sys
from typing import Any

import click
from click.core import ParameterSource

from shrike.cli import output
from shrike.cli.config import resolve_collection
from shrike.cli.output import NOTE_ID, output_options
from shrike.schemas import SearchMatch


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
    brief: bool,
    limit: int,
) -> None:
    """List notes matching structured filters.

    At least one filter is required. Use --brief for compact output. For text or
    semantic search, use 'shrike note search'.

    \b
    Examples:
      shrike note list --deck "Japanese::Vocabulary"
      shrike note list --tags verb,chapter-3
      shrike note list --type Cloze --brief --limit 20
    """
    client = ctx.obj["client"]

    if not any([deck, tags, note_type, ids, modified_since]):
        raise click.UsageError(
            "At least one filter is required: --deck, --tags, --type, --ids, or --since"
        )

    kwargs: dict[str, Any] = {
        "deck": deck,
        "tags": list(tags) or None,
        "note_type": note_type,
        "ids": list(ids) or None,
        "modified_since": modified_since,
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


@note.command("tag", short_help="Edit tags on one or more notes")
@output_options
@click.argument("note_ids", type=NOTE_ID, nargs=-1, required=True)
@click.option(
    "--set",
    "set_tags",
    multiple=True,
    callback=_parse_comma_separated,
    expose_value=True,
    help="Replace all tags with this set (repeatable, comma-separated). "
    'Pass --set "" to clear all tags. Mutually exclusive with --add/--remove.',
)
@click.option(
    "--add",
    "add_tags",
    multiple=True,
    callback=_parse_comma_separated,
    expose_value=True,
    help="Add these tags, leaving other tags intact (repeatable, comma-separated).",
)
@click.option(
    "--remove",
    "remove_tags",
    multiple=True,
    callback=_parse_comma_separated,
    expose_value=True,
    help="Remove these tags, leaving other tags intact (repeatable, comma-separated).",
)
@click.pass_context
def note_tag(
    ctx: click.Context,
    note_ids: tuple[int, ...],
    set_tags: tuple[str, ...],
    add_tags: tuple[str, ...],
    remove_tags: tuple[str, ...],
) -> None:
    """Edit the tags on one or more notes.

    Pick exactly one mode — there is no default:

    \b
      --set      replace all tags with the given set (--set "" clears)
      --add      add tags without disturbing the others
      --remove   remove specific tags without disturbing the others

    \b
    --add and --remove combine in one call; --set cannot mix with them.
    Fields and decks are untouched.

    \b
    Examples:
      shrike note tag 170000123 --set world-war-2,history
      shrike note tag 170000123 --add needs-review
      shrike note tag 170000123 --add jp --add verbs --remove jp-verbs
      shrike note tag 170000123 --set ""        # clear all tags
    """
    set_passed = ctx.get_parameter_source("set_tags") == ParameterSource.COMMANDLINE
    add = list(add_tags)
    remove = list(remove_tags)

    if set_passed and (add or remove):
        raise click.UsageError("--set cannot be combined with --add or --remove.")
    if not set_passed and not add and not remove:
        raise click.UsageError("Specify one of --set, --add, or --remove.")

    client = ctx.obj["client"]
    with output.spinner("Updating tags…"):
        if set_passed:
            result = client.update_note_tags(list(note_ids), set=list(set_tags))
        else:
            result = client.update_note_tags(list(note_ids), add=add or None, remove=remove or None)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    output.console.print(f"Updated tags on {result.notes_modified} note(s).")
    if result.not_found:
        ids = ", ".join(f"[green]#{i}[/green]" for i in result.not_found)
        output.console.print(f"[bold red]![/bold red] Not found: {ids}")


@note.command("replace", short_help="Find and replace text across notes")
@output_options
@click.argument("search")
@click.argument("replace")
@click.option("--deck", help="Scope to this deck (name, numeric id, or #id).")
@click.option(
    "--tags",
    multiple=True,
    callback=_parse_comma_separated,
    expose_value=True,
    help="Scope to notes with these tags (repeatable, comma-separated).",
)
@click.option("--type", "note_type", help="Scope to this note type.")
@click.option("--ids", multiple=True, type=NOTE_ID, help="Scope to these note IDs.")
@click.option("--field", help="Restrict to a single field (default: all fields).")
@click.option("--regex", is_flag=True, help="Treat SEARCH as a regular expression.")
@click.option("--match-case", is_flag=True, help="Case-sensitive match.")
@click.option("--dry-run", is_flag=True, help="Preview the changes without applying them.")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_context
def note_replace(
    ctx: click.Context,
    search: str,
    replace: str,
    deck: str | None,
    tags: tuple[str, ...],
    note_type: str | None,
    ids: tuple[int, ...],
    field: str | None,
    regex: bool,
    match_case: bool,
    dry_run: bool,
    yes: bool,
) -> None:
    """Find and replace text across the fields of a scoped set of notes.

    A scope is required (--deck, --tags, --type, or --ids). SEARCH is literal
    unless --regex. By default this previews the changes, asks for confirmation,
    then applies; --dry-run only previews, --yes skips the prompt.

    \b
    Examples:
      shrike note replace "teh" "the" --deck "Biology" --dry-run
      shrike note replace "colou?r" "color" --regex --tags spelling
    """
    if not any([deck, tags, note_type, ids]):
        raise click.UsageError("A scope is required: --deck, --tags, --type, or --ids.")

    client = ctx.obj["client"]
    common: dict[str, Any] = {
        "regex": regex,
        "match_case": match_case,
        "field": field,
        "deck": deck,
        "tags": list(tags) or None,
        "note_type": note_type,
        "ids": list(ids) or None,
    }

    # JSON mode is non-interactive: --dry-run previews, otherwise apply directly.
    if ctx.obj["json"]:
        result = client.find_replace_notes(search, replace, dry_run=dry_run, **common)
        output.emit_json(result)
        return

    with output.spinner("Scanning…"):
        preview = client.find_replace_notes(search, replace, dry_run=True, **common)

    if preview.notes_changed == 0:
        output.console.print("[dim]No matching notes.[/dim]")
        return

    output.console.print(f"[yellow]{preview.notes_changed}[/yellow] note(s) would change:")
    for s in preview.samples:
        output.console.print(f"  [green]#{s.id}[/green] [cyan]{s.field}[/cyan]")
        output.console.print(f"    [dim]- {s.before}[/dim]")
        output.console.print(f"    [dim]+ {s.after}[/dim]")
    extra = preview.notes_changed - len(preview.samples)
    if extra > 0:
        output.console.print(f"  [dim]… and {extra} more[/dim]")

    if dry_run:
        return
    if not yes and not click.confirm(f"Apply to {preview.notes_changed} note(s)?"):
        output.console.print("Cancelled.")
        return

    with output.spinner("Replacing…"):
        result = client.find_replace_notes(search, replace, dry_run=False, **common)
    output.console.print(f"Replaced in {result.notes_changed} note(s).")


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


def _search_match_badges(m: SearchMatch) -> str:
    """The ` · `-joined evidence badges for one search match (`note search` pretty output).

    Provenance (#182) surfaces only the signals the other badges don't already imply — `text` is
    covered by the score, `exact` by the `match:` field list — so the new, otherwise-invisible facet
    (a non-text modality like `image`, or a future lexical signal `fuzzy`/`tag`) shows on its own.
    """
    bits = []
    facet = [p.signal for p in m.provenance if p.signal not in ("text", "exact")]
    if facet:
        bits.append(", ".join(facet))
    if m.score is not None:
        bits.append(f"{m.score:.2f}")
    if m.substring is not None:
        bits.append("match: " + ", ".join(m.substring.matched_fields))
    return " · ".join(bits)


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

    # A message can accompany results (e.g. semantic ranking unavailable, exact
    # matches still shown), so print it but don't suppress the results below.
    if result.message:
        output.console.print(f"[dim]{result.message}[/dim]")

    if not result.results or not any(g.matches for g in result.results):
        if not result.message:
            output.console.print("[dim]No results.[/dim]")
        return

    for group in result.results:
        output.console.print(f"\nResults for: [cyan]{group.source}[/cyan]")
        for m in group.matches:
            badges = _search_match_badges(m)
            if brief:
                tag = f"\\[{badges}] " if badges else ""
                output.console.print(f"  {tag}[green]#{m.id}[/green] ([cyan]{m.deck}[/cyan])")
                if m.substring is not None and m.substring.snippet:
                    output.console.print(f"      [dim]{m.substring.snippet}[/dim]")
            else:
                output.note_detail(m, subtitle=f"[{badges}]" if badges else None)

    output.console.print()


@note.command("migrate-type", short_help="Change notes' note type with a field map")
@output_options
@click.argument("note_ids", type=NOTE_ID, nargs=-1, required=True)
@click.option("--to", "to_type", required=True, help="Target note type to migrate the notes to.")
@click.option(
    "--map",
    "field_maps",
    metavar="OLD=NEW",
    multiple=True,
    help="Field mapping, source=target (repeatable). Required.",
)
@click.option(
    "--template-map",
    "template_maps",
    metavar="OLD=NEW",
    multiple=True,
    help="Optional card-template mapping, source=target (repeatable).",
)
@click.option("--dry-run", is_flag=True, help="Preview the migration without applying it.")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_context
def note_migrate_type(
    ctx: click.Context,
    note_ids: tuple[int, ...],
    to_type: str,
    field_maps: tuple[str, ...],
    template_maps: tuple[str, ...],
    dry_run: bool,
    yes: bool,
) -> None:
    """Change one or more notes to a different note type, preserving history.

    The notes must all currently share one note type. --map moves field content
    by name (source=target); source fields you don't map are dropped and their
    content is lost. This is Anki's "Change Note Type": note IDs and scheduling
    for mapped templates are preserved. By default it previews, asks to confirm,
    then applies; --dry-run only previews, --yes skips the prompt.

    \b
    Examples:
      shrike note migrate-type 1700000000123 --to Cloze --map Front=Text --map "Back=Back Extra"
      shrike note migrate-type 170...1 170...2 --to Basic --map Text=Front --dry-run
    """
    if not field_maps:
        raise click.UsageError("At least one --map OLD=NEW is required.")
    field_map = dict(_parse_field(m) for m in field_maps)
    template_map = dict(_parse_field(m) for m in template_maps) or None

    client = ctx.obj["client"]
    common: dict[str, Any] = {"template_map": template_map}

    if ctx.obj["json"]:
        result = client.migrate_note_type(
            list(note_ids), to_type, field_map, dry_run=dry_run, **common
        )
        output.emit_json(result)
        return

    with output.spinner("Checking…"):
        preview = client.migrate_note_type(
            list(note_ids), to_type, field_map, dry_run=True, **common
        )

    output.console.print(
        f"[yellow]{len(preview.changed)}[/yellow] note(s): "
        f"[cyan]{preview.from_note_type}[/cyan] → [cyan]{preview.to_note_type}[/cyan]"
    )
    if preview.dropped_fields:
        output.console.print(
            "  [red]drops (content lost):[/red] " + ", ".join(preview.dropped_fields)
        )
    if preview.new_empty_fields:
        output.console.print("  [dim]empty in target:[/dim] " + ", ".join(preview.new_empty_fields))

    if dry_run:
        return
    if not yes and not click.confirm(f"Migrate {len(preview.changed)} note(s) to {to_type}?"):
        output.console.print("Cancelled.")
        return

    with output.spinner("Migrating…"):
        result = client.migrate_note_type(
            list(note_ids), to_type, field_map, dry_run=False, **common
        )
    output.console.print(f"Migrated {len(result.changed)} note(s) to {result.to_note_type}.")
