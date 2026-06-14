from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import click
from pydantic import BaseModel
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from shrike.schemas import Note, NoteTypeInfo, SearchMatch, UpsertNoteResult

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)
_pretty = True


def esc(value: Any) -> str:
    """Escape untrusted content so Rich renders it literally, never as markup.

    Note field/tag/deck/snippet text and media filenames are authored by anyone
    who can write the collection (Anki sync, a shared/imported .apkg, an MCP
    upsert) — so they MUST NOT reach ``console.print`` as live markup. A
    bracketed value would otherwise restyle the terminal (spoofing) or, if
    malformed (a stray ``[/tag]``), raise ``rich.errors.MarkupError`` and crash
    the whole command. Wrap every collection-authored datum that lands in a
    styled string with this; leave Shrike's own ``[style]`` tags unescaped.
    """
    return escape(str(value))


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
    """Append a template field — inline if single-line, indented block if multiline.

    ``value`` is collection-authored (template HTML) → escaped so it can't inject
    markup. ``label`` is Shrike's own copy and stays as live markup.
    """
    parts = value.splitlines()
    if len(parts) <= 1:
        lines.append(f"    [dim]{label}:[/dim] {esc(value)}")
    else:
        lines.append(f"    [dim]{label}:[/dim]")
        for p in parts:
            lines.append(f"      {esc(p)}")


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


def _to_jsonable(data: Any) -> Any:
    """Recursively convert Pydantic models to JSON-ready dicts, dropping nulls.

    ``exclude_none`` keeps CLI ``--json`` output close to the pre-typed shape
    (only set keys appear), independent of the MCP wire which carries nulls.
    """
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json", exclude_none=True)
    if isinstance(data, list):
        return [_to_jsonable(item) for item in data]
    return data


def emit_json(data: Any) -> None:
    """Print data as formatted JSON. Accepts Pydantic models, lists, or plain data."""
    console.print_json(json.dumps(_to_jsonable(data), ensure_ascii=False))


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


def note_summary_row(note: Note) -> list[str]:
    """Format a note as a table row.

    Note type/deck/tags are collection-authored → escaped so a bracketed value
    is shown literally and can neither restyle the terminal nor crash the render.
    """
    tags = ", ".join(esc(t) for t in note.tags)
    modified = note.modified
    if "T" in modified:
        modified = modified.split("T")[0]
    return [
        f"[green]#{note.id}[/green]",
        esc(note.note_type),
        esc(note.deck),
        tags,
        esc(modified),
    ]


def note_type_table(note_types: list[NoteTypeInfo], col_path: str) -> None:
    """Render a note types table with header. Shared by 'type list' and 'info --types'.

    Note-type names/kinds/fields are collection-authored → escaped.
    """
    console.print(
        f"[dim]Showing {len(note_types)} note type(s) in [cyan]{esc(col_path)}[/cyan][/dim]"
    )
    console.print()
    rows = [
        [
            f"[green]#{nt.id}[/green]",
            f"[cyan]{esc(nt.name)}[/cyan]",
            f"[dim]{esc(nt.type)}[/dim]",
            ", ".join(esc(f) for f in nt.fields),
        ]
        for nt in note_types
    ]
    table(["ID", "Name", "Kind", "Fields"], rows)


def note_detail(note: Note | SearchMatch, *, subtitle: str | None = None) -> None:
    """Render a full note with all its fields."""
    header = Text()
    header.append("Note ", style="bold")
    header.append(f"#{note.id}", style="green")
    if subtitle:
        header.append(f"  {subtitle}", style="dim")

    # Type/deck/tags/field names + values + modified are collection-authored →
    # escaped so bracketed content renders literally (no terminal spoof, no crash).
    lines: list[str] = []
    lines.append(f"[dim]Type:[/dim] [cyan]{esc(note.note_type)}[/cyan]")
    lines.append(f"[dim]Deck:[/dim] [cyan]{esc(note.deck)}[/cyan]")
    if note.tags:
        tags = " ".join(f"[yellow]{esc(t)}[/yellow]" for t in note.tags)
        lines.append(f"[dim]Tags:[/dim] {tags}")
    lines.append(f"[dim]Modified:[/dim] {esc(note.modified)}")

    if note.content:
        lines.append("")
        for field_name, value in note.content.items():
            lines.append(f"[cyan]{esc(field_name)}[/cyan]")
            for line in str(value).splitlines():
                lines.append(f"  {esc(line)}")

    console.print(
        Panel(
            "\n".join(lines),
            title=header,
            title_align="left",
            border_style="dim",
            padding=(0, 2),
        )
    )


def note_type_detail(nt: NoteTypeInfo) -> None:
    """Render a full note type definition."""
    header = Text()
    header.append("Note Type ", style="bold")
    header.append(nt.name, style="cyan")

    # Type/field names/metadata/template names + CSS are collection-authored →
    # escaped (template front/back via _append_template_field).
    lines: list[str] = []
    lines.append(f"[dim]ID:[/dim] [green]#{nt.id}[/green]")
    lines.append(f"[dim]Type:[/dim] {esc(nt.type)}")
    lines.append(f"[dim]Fields:[/dim] {', '.join(esc(f) for f in nt.fields)}")

    if nt.detail is not None:
        if nt.detail.fields:
            lines.append("")
            lines.append("[bold]Fields[/bold]")
            for fd in nt.detail.fields:
                meta = f"{esc(fd.font)} {esc(fd.size)}px"
                if fd.description:
                    meta += f' · "{esc(fd.description)}"'
                lines.append(f"  [cyan]{esc(fd.name)}[/cyan] [dim]{meta}[/dim]")

        lines.append("")
        lines.append("[bold]Templates[/bold]")
        for tmpl in nt.detail.templates:
            lines.append(f"  [cyan]{esc(tmpl.name)}[/cyan]")
            _append_template_field(lines, "Front", tmpl.front)
            _append_template_field(lines, "Back", tmpl.back)

        lines.append("")
        lines.append("[bold]CSS[/bold]")
        for css_line in nt.detail.css.splitlines():
            lines.append(f"  {esc(css_line)}")

    console.print(
        Panel(
            "\n".join(lines),
            title=header,
            title_align="left",
            border_style="dim",
            padding=(0, 2),
        )
    )


def result_status(results: list[UpsertNoteResult]) -> None:
    """Render a list of upsert results."""
    for r in results:
        if r.status == "created":
            console.print(f"[green]+[/green] Created note [green]#{r.id}[/green]")
        elif r.status == "updated":
            console.print(f"[yellow]~[/yellow] Updated note [green]#{r.id}[/green]")
        elif r.status == "error":
            console.print(f"[bold red]![/bold red] [red]{esc(r.error or 'Unknown error')}[/red]")
        else:
            console.print(str(r))


def error(message: str) -> None:
    """Print an error message to stderr.

    ``message`` is data (often server/echoed error text that may carry
    collection-authored content) → escaped so it can't inject markup or crash.
    """
    err_console.print(f"[bold red]Error:[/bold red] {esc(message)}")


def success(message: str) -> None:
    """Print a success message.

    ``message`` is treated as data and escaped (callers pass plain text, never
    Shrike's own markup).
    """
    console.print(f"[green]{esc(message)}[/green]")
