from __future__ import annotations

import json
import sys

import click

from shrike.cli import output


def _parse_template(value: str) -> dict:
    """Parse a NAME:FRONT:BACK template argument.

    The delimiter is ':', but template HTML often contains ':' so we split
    on the first two occurrences only.
    """
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise click.BadParameter(
            f"Invalid template format: {value!r}. "
            "Use NAME:FRONT_HTML:BACK_HTML"
        )
    return {"name": parts[0], "front": parts[1], "back": parts[2]}


@click.group("type", short_help="Manage note types")
def type_group():
    """Create, list, show, and update note type definitions."""
    pass


@type_group.command("list", short_help="List note types")
@click.pass_context
def type_list(ctx):
    """List all note types with their fields."""
    client = ctx.obj["client"]
    result = client.collection_info(include=["note_types"])

    if ctx.obj["json"]:
        output.emit_json(result.get("note_types", []))
        return

    note_types = result.get("note_types", [])
    if not note_types:
        click.echo(click.style("No note types found.", dim=True))
        return

    rows = [
        [nt["name"], nt.get("type", "standard"), ", ".join(nt.get("fields", []))]
        for nt in note_types
    ]
    output.table(["Name", "Type", "Fields"], rows)
    click.echo()


@type_group.command("show", short_help="Show note type details")
@click.argument("name")
@click.pass_context
def type_show(ctx, name):
    """Show the full definition of a note type, including templates and CSS."""
    client = ctx.obj["client"]
    result = client.collection_info(
        include=["note_types"],
        note_type_details=[name],
    )

    if ctx.obj["json"]:
        note_types = result.get("note_types", [])
        match = next((nt for nt in note_types if nt["name"] == name), None)
        output.emit_json(match)
        return

    note_types = result.get("note_types", [])
    match = next((nt for nt in note_types if nt["name"] == name), None)

    if not match:
        raise click.ClickException(f"Note type '{name}' not found.")

    if match.get("templates") is None:
        raise click.ClickException(
            f"Note type '{name}' not found. Available types: "
            + ", ".join(nt["name"] for nt in note_types)
        )

    output.note_type_detail(match)


@type_group.command("create", short_help="Create a new note type")
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
    "--json-input", is_flag=True,
    help="Read a JSON note type definition from stdin.",
)
@click.pass_context
def type_create(ctx, name, field, template, css_text, cloze, json_input):
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
            raise click.ClickException(f"Invalid JSON input: {e}")
        if isinstance(data, list):
            note_types = data
        else:
            note_types = [data]
    else:
        if not name:
            raise click.ClickException("--name is required.")
        if not field:
            raise click.ClickException("At least one --field is required.")
        if not template:
            raise click.ClickException("At least one --template is required.")

        nt_obj: dict = {
            "name": name,
            "fields": list(field),
            "templates": [_parse_template(t) for t in template],
            "css": css_text or "",
        }
        if cloze:
            nt_obj["is_cloze"] = True
        note_types = [nt_obj]

    result = client.upsert_note_types(note_types)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    for r in result.get("results", []):
        status = r.get("status")
        if status == "created":
            output.success(
                f"Created note type '{r.get('name', '')}' "
                f"(ID: {click.style(str(r.get('id', '')), **output.ID_STYLE)})"
            )
        elif status == "error":
            output.error(r.get("error", "Unknown error"))


@type_group.command("update", short_help="Update a note type")
@click.argument("note_type_id", type=int)
@click.option("--name", help="New name for the note type.")
@click.option("--css", "css_text", help="New CSS styling.")
@click.option(
    "--json-input", is_flag=True,
    help="Read a full JSON note type definition from stdin (merged with ID).",
)
@click.pass_context
def type_update(ctx, note_type_id, name, css_text, json_input):
    """Update an existing note type by ID.

    \b
    Examples:
      shrike type update 1234567890 --name "Renamed"
      shrike type update 1234567890 --css ".card { color: red; }"
      echo '{"fields":["A","B","C"],"templates":[...]}' | \\
        shrike type update 1234567890 --json-input
    """
    client = ctx.obj["client"]

    if json_input:
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON input: {e}")
        data["id"] = note_type_id
        note_types = [data]
    else:
        nt_obj: dict = {"id": note_type_id}
        if name:
            nt_obj["name"] = name
        if css_text is not None:
            nt_obj["css"] = css_text

        if len(nt_obj) == 1:
            raise click.ClickException(
                "Nothing to update. Use --name, --css, or --json-input."
            )
        note_types = [nt_obj]

    result = client.upsert_note_types(note_types)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    for r in result.get("results", []):
        status = r.get("status")
        if status == "updated":
            output.success(
                f"Updated note type '{r.get('name', '')}' "
                f"(ID: {click.style(str(r.get('id', '')), **output.ID_STYLE)})"
            )
        elif status == "error":
            output.error(r.get("error", "Unknown error"))
