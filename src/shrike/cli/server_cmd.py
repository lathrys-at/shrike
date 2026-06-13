from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from pathlib import Path

import click

from shrike.cli import output
from shrike.cli.config import (
    DEFAULT_CONFIG_PATH,
    embedding_args,
    index_args,
    locking_args,
    resolve_cache_dir,
    resolve_collection,
    resolve_embedding_profile,
    resolve_index_save,
    resolve_locking,
    resolve_recognition,
    resolve_transport,
    save_config,
    transport_args,
)
from shrike.cli.output import output_options
from shrike.client import ShrikeClient
from shrike.daemon import (
    META_FILE,
    STATE_DIR,
    cleanup_state,
    is_server_alive,
    read_server_meta,
    stop_server,
)
from shrike.embedding import BACKEND_ALIASES, SUPPORTED_BACKENDS
from shrike.log import DEFAULT_LOG_DIR, get_log_file, parse_log_line, style_log_line
from shrike.schemas import ServerStatus


def _render_status(status: ServerStatus) -> None:
    """Render a responding server's status (the GET /status report)."""
    output.console.print("[bold green]Server is running[/bold green]")
    output.kv("URL", f"[cyan]{status.url}[/cyan]")
    output.kv("PID", f"[cyan]{status.pid}[/cyan]")
    output.kv("Collection", f"[cyan]{status.collection}[/cyan]")
    output.kv("Log level", status.log_level)
    if status.log:
        output.kv("Log", f"[cyan]{status.log}[/cyan]")
    if status.uptime:
        output.kv("Uptime", status.uptime)
    if status.locking == "cooperative":
        held = "[green]held[/green]" if status.collection_held else "[dim]released (idle)[/dim]"
        output.kv("Locking", f"cooperative · collection {held}")

    emb = status.embedding
    if emb.state == "running" and emb.available:
        output.kv("Embedding", "[green]available[/green]")
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
        if emb.modalities:
            output.kv("Modalities", ", ".join(emb.modalities), indent=2)
    else:
        labels = {
            "running": "[dim]unavailable[/dim]",
            "failed": "[red]failed to start[/red]",
            "stopped": "[dim]stopped[/dim]",
        }
        output.kv("Embedding", labels.get(emb.state, "[dim]not configured[/dim]"))

    idx = status.index
    if idx.state == "ready":
        output.kv("Index", "[green]ready[/green]")
        output.kv("Vectors", f"[green]{idx.size}[/green]", indent=2)
        output.kv("Dimensions", str(idx.ndim if idx.ndim is not None else "?"), indent=2)
    elif idx.state == "building":
        output.kv("Index", "[yellow]building[/yellow]")
        output.kv("Progress", f"{idx.progress.indexed} / {idx.progress.total} notes", indent=2)
    elif idx.state == "error":
        output.kv("Index", "[red]error[/red]")
        output.kv("Error", idx.error, indent=2)
    else:
        output.kv("Index", "[dim]unavailable[/dim]")
    if idx.col_mod is not None:
        output.kv("Collection mod", str(idx.col_mod), indent=2)
    if idx.activation:
        # Per-modality activation-gate calibration (#201b): the typical best-match a query beats.
        for modality, s in idx.activation.items():
            output.kv(
                f"Activation ({modality})",
                f"μ={s['mean']:.3f} σ={s['std']:.3f} (n={int(s['n'])})",
                indent=2,
            )
    if idx.path:
        output.kv("Path", f"[cyan]{idx.path}[/cyan]", indent=2)

    # Derived-text store (#98): the FTS5 trigram sidecar behind substring/fuzzy lexical search.
    der = status.derived
    if not der.fts5:
        output.kv("Derived text", "[dim]unavailable (no SQLite FTS5)[/dim]")
    elif der.state == "ready":
        output.kv("Derived text", "[green]ready[/green]")
        output.kv("Rows", f"[green]{der.size}[/green]", indent=2)
    elif der.state == "building":
        output.kv("Derived text", "[yellow]building[/yellow]")
    elif der.state == "error":
        output.kv("Derived text", "[red]error[/red]")
    else:
        output.kv("Derived text", "[dim]unavailable[/dim]")
    if der.fts5 and der.col_mod is not None:
        output.kv("Collection mod", str(der.col_mod), indent=2)

    # Recognition (#228): OCR/ASR over note media. Shown only when configured.
    rec = status.recognition
    if rec.state == "ready":
        output.kv("Recognition", f"[green]ready[/green] ([cyan]{rec.backend}[/cyan])")
    elif rec.state == "error":
        output.kv("Recognition", "[red]error[/red]")

    # The modality coverage matrix (#498/#235): what semantic search can reach.
    if status.coverage is not None:
        served = [m for m, on in status.coverage.items() if on]
        output.kv(
            "Semantic coverage",
            ", ".join(served) if served else "[dim]none (embedding off)[/dim]",
        )

    # Multi-collection routing (#68): per-collection rows. The figures above
    # (embedding/index/derived) are the DEFAULT collection's — the one the
    # operational commands (embedding/index/recognition, collection reload) act
    # on; tool calls route per --profile. Shown only when more than one
    # collection is known.
    if status.collections:
        output.section("Collections")
        rows = []
        for c in status.collections:
            marker = "[green]*[/green]" if c.is_default else " "
            if not c.active:
                state = "[dim]idle (not opened)[/dim]"
            elif c.held is False:
                state = "[dim]released (idle)[/dim]"
            else:
                state = "[green]open[/green]"
            idx_state = c.index_state or "[dim]-[/dim]"
            tags = []
            if not c.registered:
                tags.append("[dim](boot)[/dim]")
            rows.append(
                [
                    marker,
                    f"[cyan]{c.name}[/cyan] {' '.join(tags)}".rstrip(),
                    state,
                    f"index: {idx_state}",
                    f"[cyan]{c.path}[/cyan]",
                ]
            )
        output.table(["", "Profile", "Lock", "Index", "Collection"], rows)
        output.console.print(
            "[dim]* active default — operational commands act on it; "
            "route a tool call elsewhere with [/dim][cyan]--profile[/cyan][dim].[/dim]"
        )


def _wait_for_server(
    url: str, timeout: float = 15.0, *, show_spinner: bool = True
) -> ServerStatus | None:
    """Poll until the daemon responds to /status. Returns the status or None."""
    client = ShrikeClient(url, autostart=False)
    deadline = time.monotonic() + timeout

    with output.spinner("Starting server…") if show_spinner else contextlib.nullcontext():
        while time.monotonic() < deadline:
            status = client.server_status()
            if status is not None:
                return status
            time.sleep(0.2)
    return None


@click.group("server", short_help="Manage the Shrike daemon")
def server() -> None:
    """Start, stop, and check the status of the Shrike MCP server."""


@server.command("start", short_help="Start the MCP server")
@output_options
@click.option(
    "--collection",
    type=click.Path(),
    help="Path to the Anki collection file (collection.anki2).",
)
@click.option("--port", type=int, help="Port to listen on (default: 8372).")
@click.option("--host", help="Host to bind to (default: 127.0.0.1).")
@click.option(
    "--allow-remote",
    is_flag=True,
    help="Permit binding to a non-loopback host. Endpoints are unauthenticated, "
    "so this exposes the full collection API to the network.",
)
@click.option(
    "--allowed-host",
    "allowed_host",
    multiple=True,
    help="Additional Host header to trust beyond loopback (repeatable) — e.g. a "
    "reverse-proxy or Tailscale hostname. 'host:port' matches exactly; bare host, any port.",
)
@click.option(
    "--allowed-origin",
    "allowed_origin",
    multiple=True,
    help="Additional Origin header to trust beyond loopback (repeatable). Most "
    "native MCP clients send none (always allowed); add only if a browser client 403s.",
)
@click.option(
    "--no-dns-rebinding-protection",
    is_flag=True,
    help="Disable Host/Origin validation entirely, on any bind. For deployments "
    "where the network is the trust boundary (behind a reverse proxy, on a VPN/tailnet, "
    "firewalled). Endpoints stay unauthenticated.",
)
@click.option("--foreground", is_flag=True, help="Run in the foreground instead of daemonizing.")
@click.option(
    "--log-dir",
    type=click.Path(),
    help="Directory for log files (default: platform-specific).",
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    help="Log level (default: info).",
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    help="Directory for the vector index and other caches (default: platform-specific).",
)
@click.option(
    "--index-save-delay",
    type=float,
    help="Seconds of idle after the last index change before flushing to disk (default: 60).",
)
@click.option(
    "--index-save-threshold",
    type=int,
    help="Unsaved index changes that force an immediate flush (default: 100).",
)
@click.option(
    "--cooperative-lock",
    is_flag=True,
    help="Release the collection lock when idle and re-open on demand, so an idle "
    "daemon doesn't block launching Anki (opt-in; default holds the lock for the "
    "daemon's lifetime).",
)
@click.option(
    "--lock-hold-seconds",
    type=float,
    help="In cooperative mode, seconds to hold the collection after the last "
    "operation before releasing it (default: 5).",
)
@click.option(
    "--llama-server",
    type=click.Path(),
    help="Path to llama-server binary (default: LLAMA_SERVER_PATH env or PATH lookup).",
)
@click.option(
    "--embedding-backend",
    type=click.Choice([*SUPPORTED_BACKENDS, *BACKEND_ALIASES], case_sensitive=False),
    help="Embedding backend: 'llama' (llama-server, GGUF/MLX) or 'onnx' (in-process "
    "onnxruntime; needs the 'onnx' optional dependency). Default: llama.",
)
@click.option(
    "--embedding-model",
    type=click.Path(),
    help="Path to the embedding model: a GGUF file (llama backend) or an ONNX model "
    "directory with model.onnx + tokenizer.json (onnx backend). Enables embedding.",
)
@click.option("--embedding-port", type=int, help="Port for the embedding server (default: 8373).")
@click.option(
    "--embedding-context-size",
    type=int,
    help="Context size for the embedding model. For onnx it sets the token-truncation "
    "length (default 256); above the model's positional limit is on you (no clamp).",
)
@click.option(
    "--embedding-threads", type=int, help="Number of CPU threads for embedding inference."
)
@click.option("--embedding-gpu-layers", type=int, help="Number of layers to offload to GPU.")
@click.option(
    "--embedding-pooling",
    type=click.Choice(["mean", "last", "cls", "none"], case_sensitive=False),
    help="llama-server pooling type. Set 'last' for last-token models "
    "(Jina v5, Qwen3-Embedding) whose GGUF omits it.",
)
@click.option(
    "--embedding-arg",
    "embedding_arg",
    multiple=True,
    help="Extra llama-server flag passed through verbatim, repeatable and "
    "shlex-split (e.g. --embedding-arg='--flash-attn'). Runtime-only flags; "
    "Shrike-owned flags are rejected. (llama backend only.)",
)
@click.option(
    "--embedding-onnx-provider",
    "embedding_onnx_provider",
    multiple=True,
    help="onnxruntime execution provider(s), repeatable, in priority order "
    "(e.g. CUDAExecutionProvider). Default: CPUExecutionProvider. (onnx backend only.)",
)
@click.option(
    "--embedding-batch-size",
    "embedding_batch_size",
    type=click.IntRange(min=1),
    help="Cap the embedding batch size (any backend), >= 1. Default: as large as a startup "
    "self-check proves safe. A batch-variant model (e.g. int8 ONNX) is embedded serially.",
)
@click.option(
    "--no-embedding",
    is_flag=True,
    help="Start the server without the embedding service even if a model is configured "
    "(start it later with 'shrike embedding start').",
)
@click.option(
    "--ocr-backend",
    type=click.Choice(["apple"]),
    default=None,
    help="OCR backend for recognizing text in note media (#228). Off by default. "
    "'apple' (macOS Vision) is no longer compiled into the server build — platform "
    "engines are mobile-only (docs/distribution.md); selecting it degrades "
    "recognition to an error state (the replacement is #502).",
)
@click.option(
    "--save-config",
    "save_config_flag",
    is_flag=True,
    help="Persist the resolved flags to the config file. Without this, 'server start' "
    "never writes config — it stays under your control and start always reflects the "
    "flags you pass.",
)
@click.pass_context
def server_start(
    ctx: click.Context,
    collection: str | None,
    port: int | None,
    host: str | None,
    allow_remote: bool,
    allowed_host: tuple[str, ...],
    allowed_origin: tuple[str, ...],
    no_dns_rebinding_protection: bool,
    foreground: bool,
    log_dir: str | None,
    log_level: str | None,
    cache_dir: str | None,
    index_save_delay: float | None,
    index_save_threshold: int | None,
    cooperative_lock: bool,
    lock_hold_seconds: float | None,
    llama_server: str | None,
    embedding_backend: str | None,
    embedding_model: str | None,
    embedding_port: int | None,
    embedding_context_size: int | None,
    embedding_threads: int | None,
    embedding_gpu_layers: int | None,
    embedding_pooling: str | None,
    embedding_arg: tuple[str, ...],
    embedding_onnx_provider: tuple[str, ...],
    embedding_batch_size: int | None,
    no_embedding: bool,
    ocr_backend: str | None,
    save_config_flag: bool,
) -> None:
    """Start the Shrike MCP server as a background daemon.

    The collection path can be set via --collection, the config file,
    or the SHRIKE_COLLECTION environment variable.
    """
    config = ctx.obj["config"]

    # Resolve collection path
    collection_path = resolve_collection(config, collection)
    if not collection_path:
        raise click.ClickException(
            "No collection path specified.\n\n"
            "Provide one with:\n"
            "  --collection /path/to/collection.anki2\n"
            "  SHRIKE_COLLECTION environment variable\n"
            "  'collection' key in config file"
        )

    # Ensure collection parent directory exists
    collection_dir = Path(collection_path).parent
    collection_dir.mkdir(parents=True, exist_ok=True)

    # Resolve server settings
    server_host = host or config.get("server", {}).get("host", "127.0.0.1")
    server_port = port or config.get("server", {}).get("port", 8372)
    url = f"http://{server_host}:{server_port}/mcp"

    # Resolve logging settings
    log_config = config.get("logging", {})
    resolved_log_dir = str(
        Path(log_dir or log_config.get("dir") or str(DEFAULT_LOG_DIR)).expanduser()
    )
    resolved_log_level = log_level or log_config.get("level", "info")

    # Resolve embedding settings once for both the spawned server args and the
    # config we persist — v2-first (#498): a config declaring embedders:/managed:
    # resolves against the build's compiled features and rejects the legacy
    # flags; otherwise the legacy config → env → flag cascade runs unchanged.
    from shrike.profiles import ProfileError

    try:
        resolved_embedding = resolve_embedding_profile(
            config,
            {
                "backend": embedding_backend,
                "model": embedding_model,
                "port": embedding_port,
                "context_size": embedding_context_size,
                "threads": embedding_threads,
                "gpu_layers": embedding_gpu_layers,
                "pooling": embedding_pooling,
                "extra_args": list(embedding_arg) or None,
                "llama_server": llama_server,
                "onnx_providers": list(embedding_onnx_provider) or None,
                "batch_size": embedding_batch_size,
            },
        )
    except ProfileError as e:
        raise click.ClickException(str(e)) from e
    # A v2 config rides --config to the daemon (#498): structured entries
    # (remote endpoints, api_key_env) have no flag spelling, so the daemon
    # resolves embedders:/recognizers:/managed: from the file itself. The
    # legacy flags it replaces are rejected under v2 (resolve_embedding_profile
    # above; --ocr-backend here — the v2 recognizers map owns recognition).
    is_v2 = any(config.get(k) is not None for k in ("embedders", "recognizers", "managed"))
    if is_v2:
        if ocr_backend:
            raise click.ClickException(
                "--ocr-backend is incompatible with a config declaring the v2 sections — "
                "recognition is declared in recognizers: (docs/distribution.md)"
            )
        config_file_path = str(ctx.obj["config_path"] or DEFAULT_CONFIG_PATH)
        embedding_cli_args = ["--config", config_file_path] + (
            ["--no-embedding"] if no_embedding else []
        )
        recognition_cli_args = []
    else:
        embedding_cli_args = embedding_args(resolved_embedding, no_embedding=no_embedding)
        resolved_recognition = resolve_recognition(config, ocr_backend=ocr_backend)
        recognition_cli_args = (
            ["--ocr-backend", resolved_recognition["ocr"]] if resolved_recognition["ocr"] else []
        )
    remote_args = ["--allow-remote"] if allow_remote else []

    # Resolve transport-security additions (config → env → flags) for the spawned
    # server args and the config we persist.
    resolved_transport = resolve_transport(
        config,
        allowed_hosts=list(allowed_host) or None,
        allowed_origins=list(allowed_origin) or None,
        no_dns_rebinding_protection=no_dns_rebinding_protection or None,
    )
    transport_cli_args = transport_args(resolved_transport)

    # Resolve cache dir + index-flush tuning (config → env → flags) for the
    # spawned server args and the config we persist.
    resolved_cache_dir = resolve_cache_dir(config, cache_dir)
    resolved_index_save = resolve_index_save(
        config, save_delay=index_save_delay, save_threshold=index_save_threshold
    )
    cache_args = ["--cache-dir", resolved_cache_dir] if resolved_cache_dir else []
    index_save_args = index_args(resolved_index_save)

    # Resolve cooperative-locking settings (config → env → flags).
    resolved_locking = resolve_locking(
        config, cooperative=cooperative_lock or None, hold_seconds=lock_hold_seconds
    )
    locking_cli_args = locking_args(resolved_locking)

    # Check if already running (via lock, not PID)
    if is_server_alive():
        meta = read_server_meta()
        existing_url = meta.get("url", "unknown") if meta else "unknown"
        existing_pid = meta.get("pid", "unknown") if meta else "unknown"
        raise click.ClickException(
            f"Server is already running (PID {existing_pid})\n"
            f"  URL: {existing_url}\n\n"
            "Stop it first with: shrike server stop"
        )

    json_out: bool = ctx.obj["json"]

    # Persist the resolved flags only when asked. `server start` is otherwise a
    # no-write operation: the config file stays user-managed, and start always
    # reflects exactly the flags it was given (see #56).
    if save_config_flag:
        config_path = ctx.obj["config_path"]
        config["collection"] = collection_path
        config["server"]["host"] = server_host
        config["server"]["port"] = server_port
        if allow_remote:
            config["server"]["allow_remote"] = True
        if resolved_transport["allowed_hosts"]:
            config["server"]["allowed_hosts"] = resolved_transport["allowed_hosts"]
        if resolved_transport["allowed_origins"]:
            config["server"]["allowed_origins"] = resolved_transport["allowed_origins"]
        if resolved_transport["no_dns_rebinding_protection"]:
            config["server"]["no_dns_rebinding_protection"] = True
        if resolved_locking["cooperative"]:
            config["server"]["cooperative_lock"] = True
        if resolved_locking["hold_seconds"] is not None:
            config["server"]["lock_hold_seconds"] = resolved_locking["hold_seconds"]
        # Remember embedding settings so `shrike embedding start` works later.
        # Skipped for a v2 config (#498): the embedders:/managed: sections pass
        # through save_config verbatim — writing the bridged params back as a
        # legacy embedding: section would create the forbidden v2+legacy mix.
        if not any(config.get(k) is not None for k in ("embedders", "recognizers", "managed")):
            for key, value in resolved_embedding.items():
                if value is not None:
                    config.setdefault("embedding", {})[key] = value
        if resolved_cache_dir:
            config["cache_dir"] = resolved_cache_dir
        for key, value in resolved_index_save.items():
            if value is not None:
                config.setdefault("index", {})[key] = value
        saved = save_config(config, config_path)
        if not json_out:
            output.console.print(f"  [dim]Config saved to {saved}[/dim]")

    if foreground:
        output.console.print(f"Starting server in foreground on {server_host}:{server_port}")
        output.console.print(f"Collection: {collection_path}")
        output.console.print(f"Log level: {resolved_log_level}")
        output.console.print("Press Ctrl+C to stop.\n")
        sys.argv = [
            "shrike-server",
            "--collection",
            collection_path,
            "--port",
            str(server_port),
            "--host",
            server_host,
            "--log-dir",
            resolved_log_dir,
            "--log-level",
            resolved_log_level,
            "--foreground",
            *remote_args,
            *transport_cli_args,
            *cache_args,
            *index_save_args,
            *locking_cli_args,
            *embedding_cli_args,
            *recognition_cli_args,
        ]
        from shrike.server import main

        main()
        return

    # Daemon mode
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    bootstrap_log = Path(resolved_log_dir)
    bootstrap_log.mkdir(parents=True, exist_ok=True)

    # Close our copy of the bootstrap log handle right after spawn — the child
    # keeps its own dup'd fd, so this no longer leaks for the CLI's lifetime.
    with open(bootstrap_log / "shrike-bootstrap.log", "a") as bootstrap_log_file:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "shrike.server",
                "--collection",
                collection_path,
                "--port",
                str(server_port),
                "--host",
                server_host,
                "--log-dir",
                resolved_log_dir,
                "--log-level",
                resolved_log_level,
                *remote_args,
                *transport_cli_args,
                *cache_args,
                *index_save_args,
                *locking_cli_args,
                *embedding_cli_args,
                *recognition_cli_args,
            ],
            stdout=bootstrap_log_file,
            stderr=bootstrap_log_file,
            start_new_session=True,
        )

    log_file = get_log_file(config, log_dir_override=resolved_log_dir)
    status = _wait_for_server(url)
    if status is not None:
        status.log = str(log_file)
        if json_out:
            output.emit_json(status)
        else:
            _render_status(status)
    else:
        if proc.poll() is not None:
            cleanup_state()
            raise click.ClickException(
                f"Server process exited with code {proc.returncode}.\nCheck log: {log_file}"
            )
        if json_out:
            output.emit_json(
                {
                    "started": True,
                    "pid": proc.pid,
                    "url": url,
                    "responding": False,
                    "log": str(log_file),
                }
            )
        else:
            output.console.print(
                "[yellow]Server started but not yet responding."
                " Check log for details:[/yellow]\n"
                f"  {log_file}"
            )


@server.command("stop", short_help="Stop the running server")
@output_options
@click.pass_context
def server_stop(ctx: click.Context) -> None:
    """Stop the Shrike MCP server daemon."""
    json_out: bool = ctx.obj["json"]

    if not is_server_alive():
        if json_out:
            output.emit_json({"stopped": False, "reason": "not running"})
        else:
            output.console.print("[dim]Server is not running.[/dim]")
            if META_FILE.exists():
                cleanup_state()
                output.console.print("[dim](cleaned up stale state)[/dim]")
        return

    with output.spinner("Stopping server…"):
        result = stop_server()

    if json_out:
        output.emit_json(result)
    else:
        if result.get("stopped"):
            if result.get("forced"):
                output.console.print("[yellow]Graceful shutdown timed out, forced kill.[/yellow]")
            output.success("Server stopped.")
        else:
            output.console.print(f"[dim]{result.get('reason', 'unknown')}[/dim]")


@server.command("status", short_help="Show server status")
@output_options
@click.pass_context
def server_status_cmd(ctx: click.Context) -> None:
    """Check whether the Shrike MCP server is running."""
    url = ctx.obj["url"]
    json_out: bool = ctx.obj["json"]
    client = ShrikeClient(url, autostart=False)
    with output.spinner("Checking server…"):
        status = client.server_status()

    # Responsive: the server answered /status with its full self-report.
    if status is not None:
        status.log = str(Path(status.log_dir) / "shrike.log")
        if json_out:
            output.emit_json(status)
        else:
            _render_status(status)
        return

    # Not responsive: connection state is the client's call, from the daemon lock.
    if is_server_alive():
        meta = read_server_meta() or {}
        if json_out:
            output.emit_json(
                {
                    "running": True,
                    "responsive": False,
                    "pid": meta.get("pid"),
                    "url": meta.get("url"),
                }
            )
        else:
            output.console.print("[bold yellow]Server is running but not responding[/bold yellow]")
            if meta.get("url"):
                output.kv("URL", f"[cyan]{meta['url']}[/cyan]")
            if meta.get("pid"):
                output.kv("PID", f"[cyan]{meta['pid']}[/cyan]")
        return

    if META_FILE.exists():
        cleanup_state()
    if json_out:
        output.emit_json({"running": False})
    else:
        output.console.print("[dim]Server is not running.[/dim]")
    ctx.exit(1)


@server.command("logs", short_help="View server logs")
@output_options
@click.option("--follow", "-f", is_flag=True, help="Follow the log output (like tail -f).")
@click.option("--lines", "-n", type=int, default=50, help="Number of lines to show (default: 50).")
@click.option(
    "--process",
    "-p",
    type=click.Choice(["shrike", "llama"], case_sensitive=False),
    default="shrike",
    help="Which process log to view (default: shrike).",
)
@click.option("--stdin", "read_stdin", is_flag=True, help="Read log lines from stdin.")
@click.pass_context
def server_logs(
    ctx: click.Context,
    follow: bool,
    lines: int,
    process: str,
    read_stdin: bool,
) -> None:
    """View the server log output.

    \b
    Examples:
      shrike server logs
      shrike server logs -f
      shrike server logs -n 100
      shrike --json server logs
      shrike --no-pretty server logs
      cat shrike.log | shrike server logs --stdin
    """
    json_out: bool = ctx.obj["json"]
    pretty: bool = ctx.obj["pretty"]

    if json_out and follow:
        raise click.ClickException("--json and --follow cannot be used together.")

    if read_stdin:
        input_lines = sys.stdin.read().splitlines()
        if json_out:
            _emit_json(input_lines)
        else:
            for line in input_lines:
                _emit_line(line, pretty=pretty)
        return

    # Reading from log file
    config = ctx.obj["config"]
    meta = read_server_meta()

    log_dir_path = None
    if meta:
        log_dir_path = meta.get("log_dir")
    if not log_dir_path:
        log_dir_path = config.get("logging", {}).get("dir")

    log_file = get_log_file(config, log_dir_override=log_dir_path, process_name=process)

    if not log_file.exists():
        raise click.ClickException(
            f"Log file not found: {log_file}\n"
            "Is the server running? Start it with: shrike server start"
        )

    if json_out:
        all_lines = log_file.read_text(encoding="utf-8").splitlines()
        _emit_json(all_lines[-lines:])
    elif follow:
        _tail_follow(log_file, lines, pretty=pretty)
    else:
        all_lines = log_file.read_text(encoding="utf-8").splitlines()
        for line in all_lines[-lines:]:
            _emit_line(line, pretty=pretty)


def _emit_line(line: str, *, pretty: bool) -> None:
    """Print a single log line — styled or plain."""
    if pretty:
        styled = style_log_line(line)
        if styled is not None:
            output.console.print(styled, highlight=False)
    else:
        stripped = line.strip()
        if stripped:
            click.echo(stripped)


def _emit_json(lines: list[str]) -> None:
    """Parse log lines and emit as a JSON object with a ``messages`` key."""
    records: list[dict[str, str]] = []
    for line in lines:
        parsed = parse_log_line(line)
        if parsed is not None:
            records.append(parsed)
    output.emit_json({"messages": records})


def _tail_follow(path: Path, initial_lines: int, *, pretty: bool) -> None:
    """Print the last n lines then follow new output."""
    import select

    try:
        fh = open(path, encoding="utf-8")  # noqa: SIM115
    except OSError as err:
        raise click.ClickException(f"Cannot read log file: {err}") from err

    try:
        content = fh.read()
        existing = content.splitlines()
        for line in existing[-initial_lines:]:
            _emit_line(line, pretty=pretty)

        output.console.print("[dim]--- following (Ctrl+C to stop) ---[/dim]")
        while True:
            if hasattr(fh, "fileno"):
                try:
                    select.select([fh], [], [], 0.5)
                except (ValueError, OSError):
                    time.sleep(0.5)
            else:
                time.sleep(0.5)

            new_data = fh.read()
            if new_data:
                for line in new_data.splitlines():
                    _emit_line(line, pretty=pretty)
    except KeyboardInterrupt:
        output.console.print()
    finally:
        fh.close()
