from __future__ import annotations

import json
import sys

import click

from shrike.cli import output
from shrike.cli.config import resolve_collection
from shrike.cli.output import output_options


def _resolve_note_type(client: object, identifier: str) -> tuple[int, str]:
    """Resolve a note type name or ID to (id, name).

    Accepts either a numeric ID or a name string. Raises ClickException
    if the note type is not found.
    """
    from shrike.cli.client import ShrikeClient

    assert isinstance(client, ShrikeClient)
    result = client.collection_info(include=["note_types"])
    note_types = result.note_types or []

    cleaned = identifier.lstrip("#")
    if cleaned.isdigit():
        nt_id = int(cleaned)
        match = next((nt for nt in note_types if nt.id == nt_id), None)
        if match is None:
            raise click.ClickException(f"Note type with ID {nt_id} not found.")
        return match.id, match.name

    match = next((nt for nt in note_types if nt.name == identifier), None)
    if match is None:
        available = ", ".join(nt.name for nt in note_types)
        raise click.ClickException(f"Note type '{identifier}' not found. Available: {available}")
    return match.id, match.name


def _parse_template(value: str) -> dict:
    """Parse a NAME:FRONT:BACK template argument.

    The delimiter is ':', but template HTML often contains ':' so we split
    on the first two occurrences only.
    """
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise click.BadParameter(
            f"Invalid template format: {value!r}. Use NAME:FRONT_HTML:BACK_HTML"
        )
    return {"name": parts[0], "front": parts[1], "back": parts[2]}


@click.group("type", short_help="Manage note types")
def type_group() -> None:
    """Create, list, show, and update note type definitions."""


@type_group.command("list", short_help="List note types")
@output_options
@click.argument("identifier", required=False, default=None)
@click.pass_context
def type_list(ctx: click.Context, identifier: str | None) -> None:
    """List all note types, or show detail for a specific one.

    \b
    Without IDENTIFIER, lists all note types in a table.
    With IDENTIFIER (name or numeric ID), shows full detail
    including templates and CSS.

    \b
    Examples:
      shrike type list
      shrike type list Basic
      shrike type list 1779649378945
    """
    client = ctx.obj["client"]

    if identifier is not None:
        with output.spinner("Fetching note type…"):
            nt_id, nt_name = _resolve_note_type(client, identifier)
            result = client.collection_info(
                include=["note_types"],
                note_type_details=[nt_name],
            )

        note_types = result.note_types or []
        match = next((nt for nt in note_types if nt.id == nt_id), None)

        if ctx.obj["json"]:
            output.emit_json(match)
            return

        if not match or match.detail is None:
            raise click.ClickException(f"Note type '{identifier}' not found.")

        output.note_type_detail(match)
        return

    with output.spinner("Fetching note types…"):
        result = client.collection_info(include=["note_types"])

    if ctx.obj["json"]:
        output.emit_json(result.note_types or [])
        return

    note_types = result.note_types or []
    if not note_types:
        output.console.print("[dim]No note types found.[/dim]")
        return

    col_path = resolve_collection(ctx.obj["config"]) or "collection"
    output.note_type_table(note_types, col_path)
    output.console.print()


@type_group.command("show", short_help="Show note type details")
@output_options
@click.argument("identifier")
@click.pass_context
def type_show(ctx: click.Context, identifier: str) -> None:
    """Shorthand for ``type list IDENTIFIER``."""
    ctx.invoke(type_list, identifier=identifier)


@type_group.command("create", short_help="Create a new note type")
@output_options
@click.option("--name", help="Name for the note type.")
@click.option("--field", multiple=True, help="Field name (repeatable, ordered).")
@click.option(
    "--template",
    multiple=True,
    metavar="NAME:FRONT:BACK",
    help="Card template (repeatable). E.g., 'Card 1:{{Front}}:{{FrontSide}}<hr>{{Back}}'",
)
@click.option("--css", "css_text", help="CSS styling for all cards.")
@click.option("--cloze", is_flag=True, help="Create a cloze deletion note type.")
@click.option(
    "--json-input",
    is_flag=True,
    help="Read a JSON note type definition from stdin.",
)
@click.pass_context
def type_create(
    ctx: click.Context,
    name: str | None,
    field: tuple[str, ...],
    template: tuple[str, ...],
    css_text: str | None,
    cloze: bool,
    json_input: bool,
) -> None:
    """Create a new note type definition.

    \b
    Inline:
      shrike type create --name "Vocab" --field Word --field Meaning \\
        --template 'Card 1:{{Word}}:{{FrontSide}}<hr>{{Meaning}}' \\
        --css ".card { font-size: 20px; }"

    \b
    From JSON on stdin:
      echo '{"name":"Vocab","fields":["Word","Meaning"],...}' | \\
        shrike type create --json-input
    """
    client = ctx.obj["client"]

    if json_input:
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON input: {e}") from e
        note_types = data if isinstance(data, list) else [data]
    else:
        if not name:
            raise click.UsageError("--name is required.")
        if not field:
            raise click.UsageError("At least one --field is required.")
        if not template:
            raise click.UsageError("At least one --template is required.")

        nt_obj: dict = {
            "name": name,
            "fields": list(field),
            "templates": [_parse_template(t) for t in template],
            "css": css_text or "",
        }
        if cloze:
            nt_obj["is_cloze"] = True
        note_types = [nt_obj]

    with output.spinner("Creating note type…"):
        result = client.upsert_note_types(note_types)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    for r in result.results:
        if r.status == "created":
            # Note-type names are collection-authored → escaped.
            output.console.print(
                f"[green]+[/green] Created note type [cyan]{output.esc(r.name or '')}[/cyan]"
                f" ([green]#{r.id}[/green])"
            )
        elif r.status == "error":
            output.error(r.error or "Unknown error")


@type_group.command("update", short_help="Update a note type")
@output_options
@click.argument("identifier")
@click.option("--name", help="New name for the note type.")
@click.option("--css", "css_text", help="New CSS styling.")
@click.option(
    "--json-input",
    is_flag=True,
    help="Read a full JSON note type definition from stdin (merged with ID).",
)
@click.pass_context
def type_update(
    ctx: click.Context,
    identifier: str,
    name: str | None,
    css_text: str | None,
    json_input: bool,
) -> None:
    """Update an existing note type.

    IDENTIFIER can be a note type name or numeric ID.

    \b
    Examples:
      shrike type update Basic --css ".card { color: red; }"
      shrike type update 1234567890 --name "Renamed"
      echo '{"fields":["A","B","C"],"templates":[...]}' | \\
        shrike type update Basic --json-input
    """
    client = ctx.obj["client"]
    nt_id, _nt_name = _resolve_note_type(client, identifier)

    if json_input:
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON input: {e}") from e
        data["id"] = nt_id
        note_types = [data]
    else:
        nt_obj: dict = {"id": nt_id}
        if name:
            nt_obj["name"] = name
        if css_text is not None:
            nt_obj["css"] = css_text

        if len(nt_obj) == 1:
            raise click.UsageError("Nothing to update. Use --name, --css, or --json-input.")
        note_types = [nt_obj]

    with output.spinner("Updating note type…"):
        result = client.upsert_note_types(note_types)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    for r in result.results:
        if r.status == "updated":
            # Note-type names are collection-authored → escaped.
            output.console.print(
                f"[yellow]~[/yellow] Updated note type [cyan]{output.esc(r.name or '')}[/cyan]"
                f" ([green]#{r.id}[/green])"
            )
        elif r.status == "error":
            output.error(r.error or "Unknown error")


@type_group.command("delete", short_help="Delete note types")
@output_options
@click.argument("identifiers", nargs=-1, required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
@click.pass_context
def type_delete(ctx: click.Context, identifiers: tuple[str, ...], yes: bool) -> None:
    """Delete note types by name or ID.

    A note type can only be deleted if no notes use it.

    \b
    Example:
      shrike type delete Basic
      shrike type delete 1779649378945 1779649378946 --yes
    """
    client = ctx.obj["client"]
    resolved = [_resolve_note_type(client, ident) for ident in identifiers]

    if not yes:
        desc = ", ".join(f"{name} (#{nt_id})" for nt_id, name in resolved)
        if not click.confirm(f"Delete {len(resolved)} note type(s)? {desc}\nThis cannot be undone"):
            output.console.print("Cancelled.")
            return

    with output.spinner("Deleting note type(s)…"):
        result = client.delete_note_types([nt_id for nt_id, _ in resolved])

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    for r in result.results:
        if r.status == "deleted":
            # Note-type names are collection-authored → escaped.
            name = output.esc(r.name or "")
            output.console.print(f"Deleted note type [cyan]{name}[/cyan] ([green]#{r.id}[/green])")
        elif r.status == "not_found":
            output.console.print(f"[dim]Not found: #{r.id}[/dim]")
        elif r.status == "error":
            output.console.print(
                f"[bold red]![/bold red] [cyan]{output.esc(r.name or '')}[/cyan]"
                f" ([green]#{r.id}[/green]):"
                f" [red]{output.esc(r.error or 'Unknown error')}[/red]"
            )
