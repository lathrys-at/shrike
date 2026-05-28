from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)
_pretty = True


class NoteIDType(click.ParamType):
    """Click parameter type that accepts note/type IDs with an optional ``#`` prefix."""

    name = "ID"

    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            value = value.lstrip("#")
        try:
            return int(value)
        except ValueError:
            self.fail(f"{value!r} is not a valid ID", param, ctx)


NOTE_ID = NoteIDType()


def _append_template_field(lines: list[str], label: str, value: str) -> None:
    """Append a template field — inline if single-line, indented block if multiline."""
    parts = value.splitlines()
    if len(parts) <= 1:
        lines.append(f"    [dim]{label}:[/dim] {value}")
    else:
        lines.append(f"    [dim]{label}:[/dim]")
        for p in parts:
            lines.append(f"      {p}")


def set_pretty(enabled: bool) -> None:
    """Switch the module-level consoles between styled and plain output."""
    global console, err_console, _pretty  # noqa: PLW0603
    _pretty = enabled
    if not enabled:
        console = Console(no_color=True, highlight=False)
        err_console = Console(stderr=True, no_color=True, highlight=False)


@contextmanager
def spinner(message: str) -> Generator[None, None, None]:
    """Show a dots spinner while work happens. No-op when pretty is off."""
    if _pretty:
        with console.status(message, spinner="dots"):
            yield
    else:
        yield


def _merge_json(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if value:
        ctx.obj["json"] = True
        ctx.obj["pretty"] = False
        set_pretty(False)


def _merge_pretty(ctx: click.Context, _param: click.Parameter, value: bool | None) -> None:
    if value is not None:
        if value and ctx.obj.get("json"):
            raise click.UsageError("--pretty and --json are mutually exclusive.")
        ctx.obj["pretty"] = value
        set_pretty(value)


def output_options(fn: Any) -> Any:
    """Add ``--json`` and ``--pretty/--no-pretty`` to a command.

    Values are merged into ``ctx.obj`` via callbacks so the command
    function's signature doesn't change.  This lets the same flags
    appear on both the root group and each leaf command — users can
    write either ``shrike --json info`` or ``shrike info --json``.
    """
    fn = click.option(
        "--pretty/--no-pretty",
        default=None,
        callback=_merge_pretty,
        expose_value=False,
        is_eager=True,
        help="Styled output (default: --pretty).",
    )(fn)
    fn = click.option(
        "--json",
        "json_flag",
        is_flag=True,
        default=False,
        callback=_merge_json,
        expose_value=False,
        is_eager=True,
        help="Output raw JSON instead of formatted text.",
    )(fn)
    return fn


def emit_json(data: Any) -> None:
    """Print data as formatted JSON."""
    console.print_json(json.dumps(data, ensure_ascii=False))


def table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a table with dim headers, flush left."""
    if not rows:
        console.print("[dim](none)[/dim]")
        return

    t = Table(show_edge=False, box=None, pad_edge=False, padding=(0, 2, 0, 0))
    for h in headers:
        t.add_column(h, header_style="dim underline", style="")
    for row in rows:
        t.add_row(*row)
    console.print(t)


def section(title: str) -> None:
    """Print a section header."""
    console.print()
    console.print(f"[bold]{title}[/bold]")


def kv(label: str, value: Any, indent: int = 0) -> None:
    """Print a key-value pair."""
    prefix = " " * indent
    console.print(f"{prefix}[dim]{label}:[/dim] {value}")


def note_summary_row(note: dict[str, Any]) -> list[str]:
    """Format a note dict as a table row."""
    tags = ", ".join(note.get("tags", []))
    modified = note.get("modified", "")
    if "T" in modified:
        modified = modified.split("T")[0]
    return [
        f"[green]#{note['id']}[/green]",
        note.get("note_type", ""),
        note.get("deck", ""),
        tags,
        modified,
    ]


def note_type_table(note_types: list[dict[str, Any]], col_path: str) -> None:
    """Render a note types table with header. Shared by 'type list' and 'info --types'."""
    console.print(f"[dim]Showing {len(note_types)} note type(s) in [cyan]{col_path}[/cyan][/dim]")
    console.print()
    rows = [
        [
            f"[green]#{nt.get('id', '')}[/green]",
            f"[cyan]{nt['name']}[/cyan]",
            f"[dim]{nt.get('type', 'standard')}[/dim]",
            ", ".join(nt.get("fields", [])),
        ]
        for nt in note_types
    ]
    table(["ID", "Name", "Kind", "Fields"], rows)


def note_detail(note: dict[str, Any], *, subtitle: str | None = None) -> None:
    """Render a full note with all its fields."""
    header = Text()
    header.append("Note ", style="bold")
    header.append(f"#{note['id']}", style="green")
    if subtitle:
        header.append(f"  {subtitle}", style="dim")

    lines: list[str] = []
    lines.append(f"[dim]Type:[/dim] [cyan]{note.get('note_type', '')}[/cyan]")
    lines.append(f"[dim]Deck:[/dim] [cyan]{note.get('deck', '')}[/cyan]")
    if note.get("tags"):
        tags = " ".join(f"[yellow]{t}[/yellow]" for t in note["tags"])
        lines.append(f"[dim]Tags:[/dim] {tags}")
    lines.append(f"[dim]Modified:[/dim] {note.get('modified', '')}")

    content = note.get("content", {})
    if content:
        lines.append("")
        for field_name, value in content.items():
            lines.append(f"[cyan]{field_name}[/cyan]")
            for line in str(value).splitlines():
                lines.append(f"  {line}")

    console.print(
        Panel(
            "\n".join(lines),
            title=header,
            title_align="left",
            border_style="dim",
            padding=(0, 2),
        )
    )


def note_type_detail(nt: dict[str, Any]) -> None:
    """Render a full note type definition."""
    header = Text()
    header.append("Note Type ", style="bold")
    header.append(nt["name"], style="cyan")

    lines: list[str] = []
    lines.append(f"[dim]ID:[/dim] [green]#{nt.get('id', '')}[/green]")
    lines.append(f"[dim]Type:[/dim] {nt.get('type', 'standard')}")
    lines.append(f"[dim]Fields:[/dim] {', '.join(nt.get('fields', []))}")

    templates = nt.get("templates", [])
    if templates:
        lines.append("")
        lines.append("[bold]Templates[/bold]")
        for tmpl in templates:
            lines.append(f"  [cyan]{tmpl['name']}[/cyan]")
            _append_template_field(lines, "Front", tmpl["front"])
            _append_template_field(lines, "Back", tmpl["back"])

    css = nt.get("css")
    if css is not None:
        lines.append("")
        lines.append("[bold]CSS[/bold]")
        for css_line in css.splitlines():
            lines.append(f"  {css_line}")

    console.print(
        Panel(
            "\n".join(lines),
            title=header,
            title_align="left",
            border_style="dim",
            padding=(0, 2),
        )
    )


def result_status(results: list[dict[str, Any]]) -> None:
    """Render a list of upsert/delete results."""
    for r in results:
        status = r.get("status", "unknown")
        if status == "created":
            console.print(f"[green]+[/green] Created note [green]#{r['id']}[/green]")
        elif status == "updated":
            console.print(f"[yellow]~[/yellow] Updated note [green]#{r['id']}[/green]")
        elif status == "error":
            console.print(f"[bold red]![/bold red] [red]{r.get('error', 'Unknown error')}[/red]")
        else:
            console.print(str(r))


def error(message: str) -> None:
    """Print an error message to stderr."""
    err_console.print(f"[bold red]Error:[/bold red] {message}")


def success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]{message}[/green]")
