"""``shrike search`` — retrieval: semantic + substring search, raw Anki query, coverage.

The group is a default-command group (``SearchGroup``): ``shrike search <query>``
runs the default ``search`` command (semantic + exact-substring retrieval, RRF-
fused), while ``shrike search query`` (raw Anki search expression) and ``shrike
search coverage`` (the cross-modal coverage matrix) are named subcommands.
"""

from __future__ import annotations

from typing import Any

import click

from shrike.cli import output
from shrike.cli.config import resolve_collection
from shrike.cli.groups import SearchGroup
from shrike.cli.output import NOTE_ID, output_options
from shrike.schemas import CoverageCell, CoverageMatrix, CoverageRow, SearchMatch

# The cross-modal coverage cells, styled for the matrix (#235): native is the
# strong (green) form, via-derived-text the weaker (yellow) reachability,
# unavailable a dim dash so the matrix reads at a glance.
_COVERAGE_CELL_STYLE: dict[CoverageCell, str] = {
    CoverageCell.NATIVE: "[green]native[/green]",
    CoverageCell.VIA_DERIVED_TEXT: "[yellow]via text[/yellow]",
    CoverageCell.UNAVAILABLE: "[dim]—[/dim]",
}

# The modalities the matrix is scoped to (mirrors profiles.MODALITIES; the CLI
# renders from the wire model, which is fixed at these three).
_COVERAGE_MODALITIES = ("text", "image", "audio")


@click.group("search", cls=SearchGroup, short_help="Search and query notes")
def search() -> None:
    """Search the collection and inspect retrieval.

    \b
      shrike search "electron transport chain"   # semantic + substring search
      shrike search query "is:due prop:ivl>=30"  # raw Anki search expression
      shrike search coverage                      # cross-modal coverage matrix

    Plain deck/tag/type filters live under 'shrike note list'.
    """


def _search_match_badges(m: SearchMatch) -> str:
    """The ` · `-joined evidence badges for one search match (pretty output).

    Provenance (#182) surfaces only the signals the other badges don't already imply — `text` is
    covered by the score, `exact` by the `match:` field list — so the new, otherwise-invisible facet
    (a non-text modality like `image`, or a future lexical signal `fuzzy`/`tag`) shows on its own.
    """
    bits = []
    facet = [p.signal for p in m.provenance if p.signal not in ("text", "exact")]
    if facet:
        bits.append(", ".join(facet))
    if m.score is not None:
        bits.append(f"{m.score:.2f}")
    if m.substring is not None:
        bits.append("match: " + ", ".join(m.substring.matched_fields))
    return " · ".join(bits)


@search.command("search", short_help="Semantic + substring search over notes", hidden=True)
@output_options
@click.argument("queries", nargs=-1)
@click.option(
    "--similar-to",
    multiple=True,
    type=NOTE_ID,
    metavar="ID",
    help="Find notes similar to this note ID.",
)
@click.option("--top-k", type=int, default=10, help="Results per query (default: 10).")
@click.option(
    "--threshold", type=float, default=0.5, help="Minimum similarity score (default: 0.5)."
)
@click.option("--deck", help="Restrict search to this deck.")
@click.option(
    "--tags",
    multiple=True,
    help="Restrict search to notes with these tags (repeatable, comma-separated).",
    callback=lambda ctx, param, value: tuple(  # split a,b and -t a -t b alike
        part.strip() for v in value for part in v.split(",") if part.strip()
    ),
)
@click.option("--brief", is_flag=True, help="Show only IDs and scores, not full note content.")
@click.pass_context
def search_run(
    ctx: click.Context,
    queries: tuple[str, ...],
    similar_to: tuple[int, ...],
    top_k: int,
    threshold: float,
    deck: str | None,
    tags: tuple[str, ...],
    brief: bool,
) -> None:
    """Semantic similarity + exact-substring search over the collection.

    \b
    Examples:
      shrike search "electron transport chain"
      shrike search --similar-to 170000123
      shrike search "mitochondria" --deck Biochemistry
    """
    if not queries and not similar_to:
        raise click.UsageError("Provide query strings and/or --similar-to note IDs.")

    client = ctx.obj["client"]

    kwargs: dict[str, Any] = {"top_k": top_k, "threshold": threshold}
    if queries:
        kwargs["queries"] = list(queries)
    if similar_to:
        kwargs["ids"] = list(similar_to)
    if deck:
        kwargs["deck"] = deck
    if tags:
        kwargs["tags"] = list(tags)

    with output.spinner("Searching notes…"):
        result = client.search_notes(**kwargs)

    if ctx.obj["json"]:
        output.emit_json(result)
        return

    # A message can accompany results (e.g. semantic ranking unavailable, exact
    # matches still shown), so print it but don't suppress the results below.
    if result.message:
        output.console.print(f"[dim]{output.esc(result.message)}[/dim]")

    if not result.results or not any(g.matches for g in result.results):
        if not result.message:
            output.console.print("[dim]No results.[/dim]")
        return

    for group in result.results:
        # The query string and deck/snippet (collection-authored) are escaped so a
        # bracketed value renders literally — no terminal spoof, no MarkupError crash.
        output.console.print(f"\nResults for: [cyan]{output.esc(group.source)}[/cyan]")
        for m in group.matches:
            badges = _search_match_badges(m)
            if brief:
                tag = f"\\[{badges}] " if badges else ""
                output.console.print(
                    f"  {tag}[green]#{m.id}[/green] ([cyan]{output.esc(m.deck)}[/cyan])"
                )
                # The window a literal (substring) or near-miss (fuzzy) hit matched, so a
                # text/audio card's match is legible at a glance.
                snippet = (m.substring and m.substring.snippet) or (m.fuzzy and m.fuzzy.snippet)
                if snippet:
                    output.console.print(f"      [dim]{output.esc(snippet)}[/dim]")
            else:
                output.note_detail(m, subtitle=f"[{badges}]" if badges else None)

    output.console.print()


@search.command("query", short_help="Find notes with a raw Anki search expression")
@output_options
@click.argument("expression")
@click.option("--brief", is_flag=True, help="Show only IDs and metadata, not field content.")
@click.option("--limit", type=int, default=50, help="Max notes to return (default 50).")
@click.pass_context
def query(ctx: click.Context, expression: str, brief: bool, limit: int) -> None:
    """Find notes matching a raw Anki search EXPRESSION.

    The power-user escape hatch: EXPRESSION is passed straight to Anki's search
    engine, so the full language works (is:due, prop:ivl>=30, added:, rated:,
    flag:, OR, -, parentheses). For meaning/text search use 'shrike search'; for
    plain deck/tag/type filters use 'shrike note list'.

    \b
    Examples:
      shrike search query "is:due prop:ivl>=30"
      shrike search query "added:7 -tag:done" --brief
      shrike search query "deck:Japanese (tag:verb OR tag:adj)" --limit 100
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
    # The query expression + collection path can contain brackets → escaped.
    output.console.print(
        f"[dim]Showing {count} note(s) matching [cyan]{output.esc(expression)}[/cyan] "
        f"from [cyan]{output.esc(col_path)}[/cyan][/dim]"
    )
    output.console.print()

    if brief or not any(n.content for n in notes):
        rows = [output.note_summary_row(n) for n in notes]
        output.table(["ID", "Type", "Deck", "Tags", "Modified"], rows)
    else:
        for n in notes:
            output.note_detail(n)

    output.console.print()


def _render_coverage(coverage: CoverageMatrix) -> None:
    """Render the cross-modal coverage matrix (#235) as a query×target table."""
    output.section("Coverage (query → target)")
    rows: list[list[str]] = []
    for q in _COVERAGE_MODALITIES:
        row: CoverageRow = getattr(coverage, q)
        cells = [_COVERAGE_CELL_STYLE[getattr(row, t)] for t in _COVERAGE_MODALITIES]
        rows.append([q, *cells])
    output.table(["query \\ target", *_COVERAGE_MODALITIES], rows)


@search.command("coverage", short_help="Show the cross-modal coverage matrix")
@output_options
@click.pass_context
def coverage(ctx: click.Context) -> None:
    """Show how each (query → target) modality pair is reachable.

    For each pair the cell is native (one embedding space covers both), via text
    (a recognizer derives text from the target into the text space), or
    unavailable. Reflects the server's configured embedders + recognizers, so
    e.g. text→audio reads "via text" only when ASR reaches it.
    """
    from shrike.client import ShrikeClient

    url = ctx.obj["url"]
    client = ShrikeClient(url, autostart=False)
    with output.spinner("Checking coverage…"):
        status = client.server_status()

    if status is None:
        raise click.ClickException("Server is not running or not responding.")

    if ctx.obj["json"]:
        output.emit_json(status.coverage)
        return

    if status.coverage is None:
        output.console.print("[dim]No coverage information available.[/dim]")
        return

    _render_coverage(status.coverage)
