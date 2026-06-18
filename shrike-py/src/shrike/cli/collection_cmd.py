from __future__ import annotations

import click

from shrike.cli import output
from shrike.cli.export_cmd import export
from shrike.cli.groups import OrderedGroup
from shrike.cli.import_cmd import import_cmd
from shrike.cli.info_cmd import info
from shrike.cli.media_cmd import media
from shrike.cli.output import output_options
from shrike.cli.tag_cmd import tag
from shrike.schemas import CollectionCheckResponse, CollectionPruneResponse


@click.group("collection", cls=OrderedGroup, short_help="Collection-wide operations")
def collection() -> None:
    """Collection-wide operations: info, import/export, maintenance, tags, media."""


# Rehomed under `collection` (#683): info/export/import/tag/media are collection-
# scoped, so they live beneath the collection group rather than at the top level.
collection.add_command(info)
collection.add_command(export)
collection.add_command(import_cmd)
collection.add_command(tag)
collection.add_command(media)


def _render_preview(r: CollectionPruneResponse) -> int:
    """Print what would be / was removed; return the total item count."""
    total = 0
    if r.unused_tags is not None:
        total += r.unused_tags.removed
        output.console.print(f"[yellow]{r.unused_tags.removed}[/yellow] unused tag(s)")
        # Tag names are collection-authored → escaped.
        for t in r.unused_tags.tags:
            output.console.print(f"  [yellow]{output.esc(t)}[/yellow]")
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
    if r.unused_media is not None:
        total += r.unused_media.removed
        output.console.print(f"[yellow]{r.unused_media.removed}[/yellow] unused media file(s)")
        # Media filenames are collection-authored → escaped.
        for f in r.unused_media.files:
            output.console.print(f"  [cyan]{output.esc(f)}[/cyan]")
    return total


@collection.command("prune", short_help="Remove unused tags, empty notes/cards, and unused media")
@output_options
@click.option("--unused-tags", is_flag=True, help="Remove tag-registry names no note uses.")
@click.option("--empty-notes", is_flag=True, help="Delete notes whose every field is blank.")
@click.option("--empty-cards", is_flag=True, help="Remove cards that render empty.")
@click.option("--unused-media", is_flag=True, help="Trash media files no note references.")
@click.option("--dry-run", is_flag=True, help="Preview the cleanups without applying them.")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_context
def prune(
    ctx: click.Context,
    unused_tags: bool,
    empty_notes: bool,
    empty_cards: bool,
    unused_media: bool,
    dry_run: bool,
    yes: bool,
) -> None:
    """Tidy up the collection: unused tags, empty notes/cards, and unused media.

    Select cleanups with --unused-tags / --empty-notes / --empty-cards /
    --unused-media; with none selected, all run. By default this previews what
    would be removed, asks for confirmation, then applies; --dry-run only
    previews, --yes skips the prompt.

    An empty note has every field blank, where a field is blank only if it has
    no text and no media — an image- or audio-only note is kept. Unused media
    goes to Anki's recoverable trash (see 'shrike collection check' to inspect
    media issues without pruning).

    \b
    Examples:
      shrike collection prune --dry-run            # preview everything
      shrike collection prune --unused-tags        # clear unused tags
      shrike collection prune --unused-media        # trash orphaned media
      shrike collection prune --yes                # prune all, no prompt
    """
    client = ctx.obj["client"]
    selected: dict[str, bool] = {
        "unused_tags": unused_tags,
        "empty_notes": empty_notes,
        "empty_cards": empty_cards,
        "unused_media": unused_media,
    }

    # JSON mode is non-interactive: --dry-run previews, otherwise apply directly.
    if ctx.obj["json"]:
        result = client.prune(dry_run=dry_run, **selected)
        output.emit_json(result)
        return

    with output.spinner("Scanning…"):
        preview = client.prune(dry_run=True, **selected)

    total = _render_preview(preview)
    if total == 0:
        output.console.print("[dim]Nothing to prune.[/dim]")
        return

    if dry_run:
        return
    if not yes and not click.confirm("Remove these?"):
        output.console.print("Cancelled.")
        return

    with output.spinner("Pruning…"):
        result = client.prune(dry_run=False, **selected)
    output.console.print("Pruned.")
    _render_preview(result)


def _render_check(r: CollectionCheckResponse) -> None:
    # Media dir + filenames are collection-authored → escaped so a bracket-bearing
    # name renders literally rather than crashing the render.
    issues = bool(r.unused or r.missing or r.have_trash)
    output.console.print(f"[dim]Media folder:[/dim] [cyan]{output.esc(r.media_dir)}[/cyan]")
    if r.missing:
        output.console.print(f"[bold red]{len(r.missing)}[/bold red] missing media file(s):")
        for f in r.missing:
            output.console.print(f"  [red]{output.esc(f)}[/red]")
        if r.missing_media_notes:
            ids = ", ".join(f"#{n}" for n in r.missing_media_notes)
            output.console.print(f"  [dim]referenced by notes:[/dim] [green]{ids}[/green]")
    if r.unused:
        output.console.print(
            f"[yellow]{len(r.unused)}[/yellow] unused media file(s) "
            "[dim](shrike collection prune --unused-media)[/dim]:"
        )
        for f in r.unused:
            output.console.print(f"  [cyan]{output.esc(f)}[/cyan]")
    if r.have_trash:
        output.console.print("[dim]Anki's media trash is non-empty.[/dim]")
    if not issues:
        output.console.print("[dim]No media issues found.[/dim]")


@collection.command("check", short_help="Report media-integrity issues (read-only)")
@output_options
@click.pass_context
def check(ctx: click.Context) -> None:
    """Report collection media issues without changing anything.

    Lists unused media (on disk but unreferenced — prune candidates), missing
    media (referenced by notes but absent), and whether Anki's media trash holds
    anything. Read-only; use 'shrike collection prune --unused-media' to remove
    unused files.
    """
    client = ctx.obj["client"]
    with output.spinner("Checking…"):
        result = client.collection_check()

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    _render_check(result)
    if result.missing:
        ctx.exit(1)


@collection.command("reload", short_help="Re-open the collection from disk")
@output_options
@click.pass_context
def reload(ctx: click.Context) -> None:
    """Close and re-open the collection without restarting the daemon.

    Picks up changes made to the collection file on disk underneath the daemon
    (a restored backup, a file-level sync or swap) and re-checks the search index
    for drift, rebuilding it in the background if the collection changed.
    """
    client = ctx.obj["client"]
    with output.spinner("Reloading…"):
        result = client.reload()

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    output.console.print(f"Reloaded collection [dim](col_mod={result.col_mod})[/dim].")
    if result.rebuilding:
        output.console.print("[dim]Collection changed — rebuilding the search index…[/dim]")
