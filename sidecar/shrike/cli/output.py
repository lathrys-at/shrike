from __future__ import annotations

import json
from typing import Any

import click


# -- Colors and styles --

HEADER = {"bold": True}
DIM = {"dim": True}
ID_STYLE = {"fg": "cyan"}
NAME_STYLE = {"fg": "green"}
TAG_STYLE = {"fg": "yellow"}
ERROR_STYLE = {"fg": "red", "bold": True}
SUCCESS_STYLE = {"fg": "green"}
LABEL_STYLE = {"fg": "blue", "bold": True}


def emit_json(data: Any) -> None:
    """Print data as formatted JSON and exit."""
    click.echo(json.dumps(data, indent=2, ensure_ascii=False))


def table(headers: list[str], rows: list[list[str]], max_col_width: int = 50) -> None:
    """Print an aligned table with styled headers."""
    if not rows:
        click.echo(click.style("  (none)", **DIM))
        return

    # Calculate column widths
    all_rows = [headers] + rows
    widths = [
        min(max(len(str(row[i])) for row in all_rows), max_col_width)
        for i in range(len(headers))
    ]

    # Header
    header_line = "  ".join(
        click.style(h.ljust(widths[i]), **HEADER) for i, h in enumerate(headers)
    )
    click.echo(header_line)
    click.echo(click.style("  ".join("─" * w for w in widths), **DIM))

    # Rows
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            text = str(cell)
            if len(text) > widths[i]:
                text = text[: widths[i] - 1] + "…"
            cells.append(text.ljust(widths[i]))
        click.echo("  ".join(cells))


def section(title: str) -> None:
    """Print a section header."""
    click.echo()
    click.echo(click.style(f"  {title}", **LABEL_STYLE))
    click.echo(click.style("  " + "─" * len(title), **DIM))


def kv(label: str, value: Any, indent: int = 4) -> None:
    """Print a key-value pair."""
    prefix = " " * indent
    click.echo(f"{prefix}{click.style(label + ':', **DIM)} {value}")


def note_summary_row(note: dict) -> list[str]:
    """Format a note dict as a table row."""
    tags = ", ".join(note.get("tags", []))
    modified = note.get("modified", "")
    if "T" in modified:
        modified = modified.split("T")[0]
    return [
        str(note["id"]),
        note.get("note_type", ""),
        note.get("deck", ""),
        tags,
        modified,
    ]


def note_detail(note: dict) -> None:
    """Render a full note with all its fields."""
    click.echo()
    click.echo(
        f"  {click.style('Note', **HEADER)} {click.style(str(note['id']), **ID_STYLE)}"
    )
    kv("Type", note.get("note_type", ""))
    kv("Deck", note.get("deck", ""))
    if note.get("tags"):
        tags = " ".join(click.style(t, **TAG_STYLE) for t in note["tags"])
        kv("Tags", tags)
    kv("Modified", note.get("modified", ""))

    content = note.get("content", {})
    if content:
        click.echo()
        for field_name, value in content.items():
            click.echo(f"    {click.style(field_name, **LABEL_STYLE)}")
            # Indent field content
            for line in str(value).splitlines():
                click.echo(f"      {line}")
    click.echo()


def note_type_detail(nt: dict) -> None:
    """Render a full note type definition."""
    click.echo()
    click.echo(
        f"  {click.style('Note Type', **HEADER)} "
        f"{click.style(nt['name'], **NAME_STYLE)}"
    )
    kv("ID", nt.get("id", ""))
    kv("Type", nt.get("type", "standard"))
    kv("Fields", ", ".join(nt.get("fields", [])))

    templates = nt.get("templates", [])
    if templates:
        click.echo()
        click.echo(f"    {click.style('Templates', **LABEL_STYLE)}")
        for tmpl in templates:
            click.echo(f"      {click.style(tmpl['name'], **HEADER)}")
            click.echo(f"        Front: {click.style(tmpl['front'], **DIM)}")
            click.echo(f"        Back:  {click.style(tmpl['back'], **DIM)}")

    css = nt.get("css")
    if css is not None:
        click.echo()
        click.echo(f"    {click.style('CSS', **LABEL_STYLE)}")
        for line in css.splitlines():
            click.echo(f"      {click.style(line, **DIM)}")

    click.echo()


def result_status(results: list[dict]) -> None:
    """Render a list of upsert/delete results."""
    for r in results:
        status = r.get("status", "unknown")
        if status == "created":
            icon = click.style("+", **SUCCESS_STYLE)
            msg = f"Created note {click.style(str(r['id']), **ID_STYLE)}"
        elif status == "updated":
            icon = click.style("~", fg="yellow")
            msg = f"Updated note {click.style(str(r['id']), **ID_STYLE)}"
        elif status == "error":
            icon = click.style("!", **ERROR_STYLE)
            msg = click.style(r.get("error", "Unknown error"), **ERROR_STYLE)
        else:
            icon = " "
            msg = str(r)
        click.echo(f"  {icon} {msg}")


def error(message: str) -> None:
    """Print an error message."""
    click.echo(click.style(f"Error: {message}", **ERROR_STYLE), err=True)


def success(message: str) -> None:
    """Print a success message."""
    click.echo(click.style(message, **SUCCESS_STYLE))
