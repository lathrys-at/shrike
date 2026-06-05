from __future__ import annotations

import click

from shrike.cli import output
from shrike.cli.output import output_options
from shrike.schemas import CollectionPruneResponse


@click.group("collection", short_help="Collection-wide maintenance")
def collection() -> None:
    """Collection-wide maintenance operations."""


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
