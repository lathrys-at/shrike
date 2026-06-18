"""Shared status-block renderers.

The same Embedding / Index / Derived-text / Recognition blocks render in three
places — ``shrike server status`` (the full report), ``shrike server embedding
status``, and ``shrike server index status`` — so they live here once, keyed off
the typed wire models, rather than being duplicated across the three command
modules (which is how they drifted before).

The shape: the section header IS the identity (``Index``, ``Embedding
[text]``), and ``Status:`` moves onto its own indented line with the rest of the
per-block fields. Every renderer takes a model from ``shrike.schemas`` and writes
to ``output.console`` — no JSON branch here (the ``--json`` callers emit the raw
model upstream and never reach these)."""

from __future__ import annotations

from shrike.cli import output
from shrike.schemas import (
    DerivedStatus,
    EmbeddingStatus,
    IndexStatus,
    RecognitionEngineStatus,
)


def render_embedding_spaces(spaces: list[EmbeddingStatus]) -> None:
    """One ``Embedding [<modalities>]`` block per configured space.

    The modalities move up into the header to identify the space; the per-space
    status/model/provider/batching sit indented below. A space with no reported
    modalities (an older backend, or a down space) renders a bare ``Embedding``
    header."""
    if not spaces:
        # No space reported at all (older payload) — a single "not configured".
        output.console.print("[bold]Embedding[/bold]")
        output.kv("Status", "[dim]not configured[/dim]", indent=2)
        return
    for emb in spaces:
        modalities = getattr(emb, "modalities", None)
        # The `[modalities]` is literal text, not Rich markup — escape it so
        # `[text]` isn't parsed (and swallowed) as a style tag.
        suffix = output.esc(f" [{', '.join(modalities)}]") if modalities else ""
        output.console.print(f"[bold]Embedding{suffix}[/bold]")
        _render_embedding_fields(emb)


def _render_embedding_fields(emb: EmbeddingStatus) -> None:
    """The per-space body — ``Status:`` on its own line, then the rest."""
    if emb.state == "running" and emb.available:
        output.kv("Status", "[green]available[/green]", indent=2)
        if emb.url:
            output.kv("URL", f"[cyan]{emb.url}[/cyan]", indent=2)
        if emb.pid:
            output.kv("PID", f"[cyan]{emb.pid}[/cyan]", indent=2)
        if emb.model:
            output.kv("Model", f"[cyan]{emb.model}[/cyan]", indent=2)
        if emb.provider:
            output.kv("Provider", emb.provider.replace("ExecutionProvider", ""), indent=2)
        if emb.batch:
            output.kv("Batching", emb.batch, indent=2)
    else:
        labels = {
            "running": "[dim]unavailable[/dim]",
            "failed": "[red]failed to start[/red]",
            "stopped": "[dim]stopped[/dim]",
            "not_configured": "[dim]not configured[/dim]",
        }
        output.kv("Status", labels.get(emb.state, "[dim]not configured[/dim]"), indent=2)


def render_index(idx: IndexStatus) -> None:
    """The ``Index`` block — ``Status:`` on its own line, then the per-modality
    sub-index breakdown, then the shared col_mod/activation/path."""
    output.console.print("[bold]Index[/bold]")
    if idx.state == "ready":
        output.kv("Status", "[green]ready[/green]", indent=2)
    elif idx.state == "building":
        output.kv("Status", "[yellow]building[/yellow]", indent=2)
        output.kv("Progress", f"{idx.progress.indexed} / {idx.progress.total} notes", indent=2)
    elif idx.state == "error":
        output.kv("Status", "[red]error[/red]", indent=2)
        output.kv("Error", idx.error, indent=2)
    else:
        output.kv("Status", "[dim]unavailable[/dim]", indent=2)

    # Per-modality sub-index breakdown: one row per sub-index with its
    # own vectors/dimensions. Falls back to the aggregate size/ndim when the
    # server didn't report the breakdown (older payload, or no index built).
    if idx.modalities:
        for m in idx.modalities:
            dims = str(m.ndim) if m.ndim is not None else "?"
            output.kv(
                f"Vectors ({m.modality})",
                f"[green]{m.size}[/green] · {dims} dims",
                indent=2,
            )
    elif idx.state == "ready":
        output.kv("Vectors", f"[green]{idx.size}[/green]", indent=2)
        output.kv("Dimensions", str(idx.ndim if idx.ndim is not None else "?"), indent=2)

    if idx.col_mod is not None:
        output.kv("Collection mod", str(idx.col_mod), indent=2)
    if idx.activation:
        # Per-modality activation-gate calibration: the typical best-match a query beats.
        for modality, s in idx.activation.items():
            output.kv(
                f"Activation ({modality})",
                f"μ={s['mean']:.3f} σ={s['std']:.3f} (n={int(s['n'])})",
                indent=2,
            )
    if idx.path:
        output.kv("Path", f"[cyan]{idx.path}[/cyan]", indent=2)


def render_derived(der: DerivedStatus) -> None:
    """The ``Derived text`` block — ``Status:`` on its own line."""
    output.console.print("[bold]Derived text[/bold]")
    if not der.fts5:
        output.kv("Status", "[dim]unavailable (no SQLite FTS5)[/dim]", indent=2)
        return
    if der.state == "ready":
        output.kv("Status", "[green]ready[/green]", indent=2)
        output.kv("Rows", f"[green]{der.size}[/green]", indent=2)
    elif der.state == "building":
        output.kv("Status", "[yellow]building[/yellow]", indent=2)
    elif der.state == "error":
        output.kv("Status", "[red]error[/red]", indent=2)
    else:
        output.kv("Status", "[dim]unavailable[/dim]", indent=2)
    if der.col_mod is not None:
        output.kv("Collection mod", str(der.col_mod), indent=2)


def render_recognition(recognition: dict[str, RecognitionEngineStatus]) -> None:
    """The ``Recognition`` block — one entry per attached engine,
    keyed by source; ``Status:`` on its own line. An empty map renders a single
    ``Recognition`` header with a ``none`` status."""
    if not recognition:
        output.console.print("[bold]Recognition[/bold]")
        output.kv("Status", "[dim]none (no recognizer attached)[/dim]", indent=2)
        return
    for source in sorted(recognition):
        eng = recognition[source]
        # `[source]` is literal — escape so Rich doesn't parse it as markup.
        output.console.print(f"[bold]Recognition {output.esc(f'[{source}]')}[/bold]")
        if eng.state == "ready":
            output.kv("Status", f"[green]ready[/green] ([cyan]{eng.backend}[/cyan])", indent=2)
        elif eng.state == "error":
            output.kv("Status", f"[red]error[/red] ([cyan]{eng.backend}[/cyan])", indent=2)
        else:
            output.kv("Status", f"[dim]{eng.state}[/dim]", indent=2)
        if eng.fingerprint:
            output.kv("Fingerprint", f"[cyan]{eng.fingerprint}[/cyan]", indent=2)
