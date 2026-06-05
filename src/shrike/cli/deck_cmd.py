from __future__ import annotations

import click

from shrike.cli import output
from shrike.cli.output import output_options
from shrike.schemas import DeckInfo, UpsertDecksResponse


def _match_deck(decks: list[DeckInfo], ref: str) -> DeckInfo | None:
    """Find a deck by name, numeric id, or '#'-prefixed id.

    '#<id>' matches by id only; a bare integer is tried as an id first, then as
    a name; anything else is a name (mirrors the server's deck-ref resolution).
    """
    if ref.startswith("#") and ref[1:].isdigit():
        did = int(ref[1:])
        return next((d for d in decks if d.id == did), None)
    if ref.isdigit():
        did = int(ref)
        by_id = next((d for d in decks if d.id == did), None)
        return by_id or next((d for d in decks if d.name == ref), None)
    return next((d for d in decks if d.name == ref), None)


@click.group("deck", short_help="Manage decks")
def deck() -> None:
    """Create, rename, and delete decks.

    Deck deletion requires the deck to be empty: move its notes elsewhere first
    (e.g. 'shrike note update <id> --deck …', or rename one deck onto another to
    merge), then delete the now-empty deck.
    """


def _render_upsert(ctx: click.Context, result: UpsertDecksResponse) -> None:
    if ctx.obj["json"]:
        output.emit_json(result)
        return
    for r in result.results:
        if r.status == "created":
            output.console.print(f"[green]+[/green] Created deck [cyan]{r.name}[/cyan]")
        elif r.status == "updated":
            output.console.print(f"[yellow]~[/yellow] Updated deck [cyan]{r.name}[/cyan]")
        else:
            output.console.print(f"[bold red]![/bold red] [red]{r.error}[/red]")


@deck.command("create", short_help="Create a deck")
@output_options
@click.argument("name")
@click.pass_context
def deck_create(ctx: click.Context, name: str) -> None:
    """Create an empty deck. Use "::" for nesting (e.g. "Japanese::Vocabulary").

    No-op if the deck already exists.

    \b
    Examples:
      shrike deck create "Japanese::Vocabulary"
    """
    client = ctx.obj["client"]
    with output.spinner("Creating deck…"):
        result = client.upsert_decks([{"name": name}])
    _render_upsert(ctx, result)


@deck.command("rename", short_help="Rename or reparent a deck")
@output_options
@click.argument("old")
@click.argument("new")
@click.pass_context
def deck_rename(ctx: click.Context, old: str, new: str) -> None:
    """Rename or reparent a deck.

    Decks do not merge: renaming onto a name another deck already uses is an
    error. To consolidate, move the notes (e.g. 'note update <id> --deck NEW').

    \b
    Examples:
      shrike deck rename "Japanese::Vocabulary" "Japanese::Vocab"
      shrike deck rename "Misc::French" "French"      # reparent to top level
    """
    client = ctx.obj["client"]
    decks = client.collection_info(include=["decks"]).decks or []
    match = _match_deck(decks, old)
    if match is None:
        raise click.ClickException(f"Deck not found: {old}")

    with output.spinner("Renaming deck…"):
        result = client.upsert_decks([{"id": match.id, "name": new}])
    _render_upsert(ctx, result)


@deck.command("delete", short_help="Delete empty decks")
@output_options
@click.argument("names", nargs=-1, required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
@click.pass_context
def deck_delete(ctx: click.Context, names: tuple[str, ...], yes: bool) -> None:
    """Delete one or more decks. Each must already be empty.

    \b
    Examples:
      shrike deck delete "Old Deck"
      shrike deck delete "A" "B" --yes
    """
    if not yes:
        listed = ", ".join(names)
        if not click.confirm(f"Delete {len(names)} deck(s)? {listed}"):
            output.console.print("Cancelled.")
            return

    client = ctx.obj["client"]
    with output.spinner("Deleting decks…"):
        result = client.delete_decks(list(names))

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    for name in result.deleted:
        output.console.print(f"[red]-[/red] Deleted deck [cyan]{name}[/cyan]")
    for name in result.not_empty:
        output.console.print(
            f"[bold red]![/bold red] [cyan]{name}[/cyan] is not empty — move its notes out first"
        )
    for name in result.not_found:
        output.console.print(f"[bold red]![/bold red] Not found: [cyan]{name}[/cyan]")

    if result.not_empty or result.not_found:
        ctx.exit(1)
