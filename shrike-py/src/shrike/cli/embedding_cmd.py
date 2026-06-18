from __future__ import annotations

import click

from shrike.cli import output, status_render
from shrike.cli.config import resolve_embedding_profile
from shrike.cli.groups import OrderedGroup
from shrike.cli.index_cmd import _poll_progress
from shrike.cli.output import output_options
from shrike.client import ShrikeClient
from shrike.harness.engines.embedding.runtime import BACKEND_ALIASES, SUPPORTED_BACKENDS
from shrike.schemas import EmbeddingStatus


@click.group("embedding", cls=OrderedGroup, short_help="Manage the embedding service")
def embedding() -> None:
    """Start, stop, and inspect the embedding service used for semantic search."""


@embedding.command("status", short_help="Show embedding service status")
@output_options
@click.pass_context
def embedding_status(ctx: click.Context) -> None:
    """Show the current state of the embedding service.

    Reports one entry per configured embedding space: a multi-space
    profile shows each space keyed by its modalities.

    \b
    Examples:
      shrike server embedding status
      shrike --json server embedding status
    """
    client: ShrikeClient = ctx.obj["client"]
    json_out: bool = ctx.obj["json"]

    # Pull the full status so every space is reported, not just the primary. The
    # shared renderer keeps this identical to the `server status` Embedding block.
    with output.spinner("Checking embedding service…"):
        full = client.server_status()

    if full is None:
        raise click.ClickException("Server is not responding.")

    spaces = full.embedding_spaces or [full.embedding]

    if json_out:
        # The full per-space list; a single-space server emits a one-element list.
        output.emit_json([s.model_dump(exclude_none=True) for s in spaces])
        return

    status_render.render_embedding_spaces(spaces)


@embedding.command("start", short_help="Start the embedding service")
@output_options
@click.option(
    "--embedding-backend",
    "backend",
    type=click.Choice([*SUPPORTED_BACKENDS, *BACKEND_ALIASES], case_sensitive=False),
    help="Embedding backend: 'llama' or 'onnx' (needs the 'onnx' extra). Default: llama.",
)
@click.option(
    "--embedding-model",
    "model",
    type=click.Path(),
    help="Path to the embedding model: a GGUF file (llama) or an ONNX model directory (onnx).",
)
@click.option(
    "--llama-server", "llama_server", type=click.Path(), help="Path to llama-server binary."
)
@click.option(
    "--embedding-port", "port", type=int, help="Port for the embedding server (default: 8373)."
)
@click.option(
    "--embedding-context-size", "context_size", type=int, help="Context size for embedding model."
)
@click.option(
    "--embedding-threads", "threads", type=int, help="CPU threads for embedding inference."
)
@click.option("--embedding-gpu-layers", "gpu_layers", type=int, help="Layers to offload to GPU.")
@click.option(
    "--embedding-pooling",
    "pooling",
    type=click.Choice(["mean", "last", "cls", "none"], case_sensitive=False),
    help="llama-server pooling type. Set 'last' for last-token models "
    "(Jina v5, Qwen3-Embedding) whose GGUF omits it.",
)
@click.option(
    "--embedding-arg",
    "extra_args",
    multiple=True,
    help="Extra llama-server flag passed through verbatim, repeatable and "
    "shlex-split (e.g. --embedding-arg='--flash-attn'). Runtime-only flags; "
    "Shrike-owned flags are rejected. (llama backend only.)",
)
@click.option(
    "--embedding-onnx-provider",
    "onnx_providers",
    multiple=True,
    help="onnxruntime execution provider(s), repeatable, in priority order. "
    "Default: CPUExecutionProvider. (onnx backend only.)",
)
@click.option(
    "--embedding-batch-size",
    "batch_size",
    type=click.IntRange(min=1),
    help="Cap the embedding batch size (any backend), >= 1. Default: as large as safe.",
)
@click.option(
    "--background", is_flag=True, help="Return immediately without waiting for any index rebuild."
)
@click.pass_context
def embedding_start(
    ctx: click.Context,
    backend: str | None,
    model: str | None,
    llama_server: str | None,
    port: int | None,
    context_size: int | None,
    threads: int | None,
    gpu_layers: int | None,
    pooling: str | None,
    extra_args: tuple[str, ...],
    onnx_providers: tuple[str, ...],
    batch_size: int | None,
    background: bool,
) -> None:
    """Start the embedding service on a running server.

    Resolves the model and llama-server settings via the same config → env →
    flag cascade as `shrike server start`. If the embedding model changed (or
    the index is stale), an index rebuild is triggered automatically.

    \b
    Examples:
      shrike server embedding start
      shrike server embedding start --embedding-model ~/models/embed.gguf
      shrike server embedding start --background
    """
    config = ctx.obj["config"]
    client: ShrikeClient = ctx.obj["client"]
    json_out: bool = ctx.obj["json"]

    # v2-first like `server start`: a config declaring embedders:/managed:
    # is the only home for these settings — the legacy flags/env are rejected/
    # ignored under it; a legacy config runs the old cascade unchanged.
    from shrike.harness.profiles import ProfileError

    try:
        resolved = resolve_embedding_profile(
            config,
            {
                "backend": backend,
                "model": model,
                "port": port,
                "context_size": context_size,
                "threads": threads,
                "gpu_layers": gpu_layers,
                "pooling": pooling,
                "extra_args": list(extra_args) or None,
                "llama_server": llama_server,
                "onnx_providers": list(onnx_providers) or None,
                "batch_size": batch_size,
            },
        )
    except ProfileError as e:
        raise click.ClickException(str(e)) from e

    # Under a v2 config send NO overrides: the daemon booted with --config and
    # owns the resolution — "start what the config says". (The bridged params
    # include keys like endpoint that the legacy override body doesn't carry.)
    from shrike.harness.profiles import parse_capabilities

    if not parse_capabilities(config).legacy:
        resolved = {}

    with output.spinner("Starting embedding service…"):
        data = client.embedding_start(**resolved)

    if data.status == "already_running":
        if json_out:
            output.emit_json(data)
        else:
            output.console.print("[dim]Embedding service is already running.[/dim]")
        return

    idx = data.index

    if idx.state == "building" and not background:
        _poll_progress(client, idx.progress.total, json_out=json_out)
        if not json_out:
            _render_embedding(client.embedding_status())
        return

    if json_out:
        output.emit_json(data)
    else:
        output.success("Embedding service started.")
        _render_embedding(data.embedding)
        if idx.state == "building":
            output.console.print("[dim]Index rebuild started in the background.[/dim]")


@embedding.command("stop", short_help="Stop the embedding service")
@output_options
@click.pass_context
def embedding_stop(ctx: click.Context) -> None:
    """Stop the embedding service on a running server.

    The Shrike server and the Anki collection stay up; semantic search becomes
    unavailable until the service is started again. Useful for llama-server
    upgrades, model swaps, or freeing GPU/RAM during maintenance.

    \b
    Examples:
      shrike server embedding stop
      shrike --json server embedding stop
    """
    client: ShrikeClient = ctx.obj["client"]
    json_out: bool = ctx.obj["json"]

    with output.spinner("Stopping embedding service…"):
        data = client.embedding_stop()

    if json_out:
        output.emit_json(data)
        return

    if data.status == "stopped":
        output.success("Embedding service stopped.")
    else:
        output.console.print("[dim]Embedding service is not running.[/dim]")


def _render_embedding(emb: EmbeddingStatus) -> None:
    """Render ONE embedding space's block (the `embedding start` confirmation).

    Delegates to the shared per-space renderer so the start-confirmation block
    matches the `server status` / `server embedding status` Embedding blocks.
    `start` only knows the space it just started, so it renders one."""
    status_render.render_embedding_spaces([emb])
