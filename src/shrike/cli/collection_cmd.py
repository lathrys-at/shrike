from __future__ import annotations

import click

from shrike.cli import output
from shrike.cli.config import resolve_collection
from shrike.cli.output import output_options
from shrike.schemas import CollectionPruneResponse


@click.group("collection", short_help="Collection-wide query and maintenance")
def collection() -> None:
    """Collection-wide operations: raw query and maintenance."""


def _render_preview(r: CollectionPruneResponse) -> int:
    """Print what would be / was removed; return the total item count."""
    total = 0
    if r.unused_tags is not None:
        total += r.unused_tags.removed
        output.console.print(f"[yellow]{r.unused_tags.removed}[/yellow] unused tag(s)")
        for t in r.unused_tags.tags:
            output.console.print(f"  [yellow]{t}[/yellow]")
    if r.empty_notes is not None:
        n = len(r.empty_notes.removed)
        total += n
        output.console.print(f"[yellow]{n}[/yellow] empty note(s)")
        for nid in r.empty_notes.removed:
            output.console.print(f"  [green]#{nid}[/green]")
    if r.empty_cards is not None:
        total += r.empty_cards.cards_removed
        deleted = r.empty_cards.notes_deleted
        suffix = f" ({len(deleted)} note(s) deleted)" if deleted else ""
        output.console.print(
            f"[yellow]{r.empty_cards.cards_removed}[/yellow] empty card(s){suffix}"
        )
    return total


@collection.command("prune", short_help="Remove unused tags, empty notes, and empty cards")
@output_options
@click.option("--unused-tags", is_flag=True, help="Remove tag-registry names no note uses.")
@click.option("--empty-notes", is_flag=True, help="Delete notes whose every field is blank.")
@click.option("--empty-cards", is_flag=True, help="Remove cards that render empty.")
@click.option("--apply", "apply_", is_flag=True, help="Apply the changes (default: preview only).")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_context
def prune(
    ctx: click.Context,
    unused_tags: bool,
    empty_notes: bool,
    empty_cards: bool,
    apply_: bool,
    yes: bool,
) -> None:
    """Tidy up the collection: unused tags, empty notes, and empty cards.

    Select cleanups with --unused-tags / --empty-notes / --empty-cards; with
    none selected, all three run. By default this only previews what would be
    removed. Pass --apply to actually remove (it previews, asks for
    confirmation, then applies); --yes skips the prompt.

    An empty note has every field blank, where a field is blank only if it has
    no text and no media — an image- or audio-only note is kept.

    \b
    Examples:
      shrike collection prune                        # preview everything
      shrike collection prune --unused-tags --apply  # clear unused tags
      shrike collection prune --apply --yes          # prune all, no prompt
    """
    client = ctx.obj["client"]
    selected: dict[str, bool] = {
        "unused_tags": unused_tags,
        "empty_notes": empty_notes,
        "empty_cards": empty_cards,
    }

    # JSON mode is non-interactive: --apply applies, otherwise preview.
    if ctx.obj["json"]:
        result = client.prune(dry_run=not apply_, **selected)
        output.emit_json(result)
        return

    with output.spinner("Scanning…"):
        preview = client.prune(dry_run=True, **selected)

    total = _render_preview(preview)
    if total == 0:
        output.console.print("[dim]Nothing to prune.[/dim]")
        return

    if not apply_:
        output.console.print("[dim]Preview only — pass --apply to remove.[/dim]")
        return
    if not yes and not click.confirm("Remove these?"):
        output.console.print("Cancelled.")
        return

    with output.spinner("Pruning…"):
        result = client.prune(dry_run=False, **selected)
    output.console.print("Pruned.")
    _render_preview(result)


@collection.command("query", short_help="Find notes with a raw Anki search expression")
@output_options
@click.argument("expression")
@click.option("--brief", is_flag=True, help="Show only IDs and metadata, not field content.")
@click.option("--limit", type=int, default=50, help="Max notes to return (default 50).")
@click.pass_context
def query(ctx: click.Context, expression: str, brief: bool, limit: int) -> None:
    """Find notes matching a raw Anki search EXPRESSION.

    The power-user escape hatch: EXPRESSION is passed straight to Anki's search
    engine, so the full language works (is:due, prop:ivl>=30, added:, rated:,
    flag:, OR, -, parentheses). For meaning/text search use 'note search'; for
    plain deck/tag/type filters use 'note list'.

    \b
    Examples:
      shrike collection query "is:due prop:ivl>=30"
      shrike collection query "added:7 -tag:done" --brief
      shrike collection query "deck:Japanese (tag:verb OR tag:adj)" --limit 100
    """
    client = ctx.obj["client"]

    with output.spinner("Searching…"):
        result = client.query(expression, fields="meta" if brief else "full", limit=limit)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    notes = result.notes
    if not notes:
        output.console.print("[dim]No notes found.[/dim]")
        return

    col_path = resolve_collection(ctx.obj["config"]) or "collection"
    count = f"{len(notes)} of {result.total}" if result.total > len(notes) else str(result.total)
    output.console.print(
        f"[dim]Showing {count} note(s) matching [cyan]{expression}[/cyan] "
        f"from [cyan]{col_path}[/cyan][/dim]"
    )
    output.console.print()

    if brief or not any(n.content for n in notes):
        rows = [output.note_summary_row(n) for n in notes]
        output.table(["ID", "Type", "Deck", "Tags", "Modified"], rows)
    else:
        for n in notes:
            output.note_detail(n)

    output.console.print()
