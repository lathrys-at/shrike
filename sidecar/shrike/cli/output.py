from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console()
err_console = Console(stderr=True)


def emit_json(data: Any) -> None:
    """Print data as formatted JSON."""
    console.print_json(json.dumps(data, ensure_ascii=False))


def table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a table with styled headers."""
    if not rows:
        console.print("  [dim](none)[/dim]")
        return

    t = Table(show_edge=False, pad_edge=False, box=None, padding=(0, 2))
    for h in headers:
        t.add_column(h, style="bold")
    for row in rows:
        t.add_row(*row)
    console.print(t)


def section(title: str) -> None:
    """Print a section header."""
    console.print()
    console.print(f"  [bold blue]{title}[/bold blue]")
    console.print(f"  [dim]{'─' * len(title)}[/dim]")


def kv(label: str, value: Any, indent: int = 4) -> None:
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
        str(note["id"]),
        note.get("note_type", ""),
        note.get("deck", ""),
        tags,
        modified,
    ]


def note_detail(note: dict[str, Any]) -> None:
    """Render a full note with all its fields."""
    header = Text()
    header.append("Note ", style="bold")
    header.append(str(note["id"]), style="cyan")

    lines: list[str] = []
    lines.append(f"[dim]Type:[/dim] {note.get('note_type', '')}")
    lines.append(f"[dim]Deck:[/dim] {note.get('deck', '')}")
    if note.get("tags"):
        tags = " ".join(f"[yellow]{t}[/yellow]" for t in note["tags"])
        lines.append(f"[dim]Tags:[/dim] {tags}")
    lines.append(f"[dim]Modified:[/dim] {note.get('modified', '')}")

    content = note.get("content", {})
    if content:
        lines.append("")
        for field_name, value in content.items():
            lines.append(f"[bold blue]{field_name}[/bold blue]")
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
    header.append(nt["name"], style="green")

    lines: list[str] = []
    lines.append(f"[dim]ID:[/dim] {nt.get('id', '')}")
    lines.append(f"[dim]Type:[/dim] {nt.get('type', 'standard')}")
    lines.append(f"[dim]Fields:[/dim] {', '.join(nt.get('fields', []))}")

    templates = nt.get("templates", [])
    if templates:
        lines.append("")
        lines.append("[bold blue]Templates[/bold blue]")
        for tmpl in templates:
            lines.append(f"  [bold]{tmpl['name']}[/bold]")
            lines.append(f"    [dim]Front:[/dim] {tmpl['front']}")
            lines.append(f"    [dim]Back:[/dim]  {tmpl['back']}")

    console.print(
        Panel(
            "\n".join(lines),
            title=header,
            title_align="left",
            border_style="dim",
            padding=(0, 2),
        )
    )

    css = nt.get("css")
    if css is not None:
        console.print(Syntax(css, "css", theme="monokai", padding=1))


def result_status(results: list[dict[str, Any]]) -> None:
    """Render a list of upsert/delete results."""
    for r in results:
        status = r.get("status", "unknown")
        if status == "created":
            console.print(f"  [green]+[/green] Created note [cyan]{r['id']}[/cyan]")
        elif status == "updated":
            console.print(f"  [yellow]~[/yellow] Updated note [cyan]{r['id']}[/cyan]")
        elif status == "error":
            console.print(f"  [bold red]![/bold red] [red]{r.get('error', 'Unknown error')}[/red]")
        else:
            console.print(f"    {r}")


def error(message: str) -> None:
    """Print an error message to stderr."""
    err_console.print(f"[bold red]Error:[/bold red] {message}")


def success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]{message}[/green]")
