from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import click

from shrike.cli import output
from shrike.cli.client import ShrikeClient
from shrike.cli.config import resolve_collection, save_config
from shrike.cli.output import output_options
from shrike.daemon import (
    META_FILE,
    STATE_DIR,
    cleanup_state,
    is_server_alive,
    read_server_meta,
    server_status,
    stop_server,
)
from shrike.log import DEFAULT_LOG_DIR, get_log_file, parse_log_line, style_log_line


def _embedding_args(config: dict[str, Any]) -> list[str]:
    """Build CLI args for the embedding service from config."""
    emb = config.get("embedding", {})
    args: list[str] = []
    model = emb.get("model")
    if model:
        args.extend(["--embedding-model", str(model)])
        if emb.get("port"):
            args.extend(["--embedding-port", str(emb["port"])])
        if emb.get("context_size"):
            args.extend(["--embedding-context-size", str(emb["context_size"])])
        if emb.get("threads"):
            args.extend(["--embedding-threads", str(emb["threads"])])
        if emb.get("gpu_layers"):
            args.extend(["--embedding-gpu-layers", str(emb["gpu_layers"])])
    return args


def _render_status(status: dict[str, Any]) -> None:
    """Render the unified server status block used by both start and status."""
    if not status.get("running", False):
        output.console.print("[dim]Server is not running.[/dim]")
        return

    if status.get("responsive") is False:
        output.console.print("[bold yellow]Server is running but not responding[/bold yellow]")
    else:
        output.console.print("[bold green]Server is running[/bold green]")
    if status.get("url"):
        output.kv("URL", f"[cyan]{status['url']}[/cyan]")
    if status.get("pid"):
        output.kv("PID", f"[cyan]{status['pid']}[/cyan]")
    if status.get("collection"):
        output.kv("Collection", f"[cyan]{status['collection']}[/cyan]")
    if status.get("log_level"):
        output.kv("Log level", status["log_level"])
    if status.get("log"):
        output.kv("Log", f"[cyan]{status['log']}[/cyan]")
    if status.get("uptime"):
        output.kv("Uptime", status["uptime"])
    emb = status.get("embedding")
    if emb:
        if emb.get("available"):
            output.kv("Embedding", "[green]available[/green]")
            if emb.get("url"):
                output.kv("URL", f"[cyan]{emb['url']}[/cyan]", indent=2)
            if emb.get("pid"):
                output.kv("PID", f"[cyan]{emb['pid']}[/cyan]", indent=2)
            if emb.get("model"):
                output.kv("Model", f"[cyan]{emb['model']}[/cyan]", indent=2)
        else:
            output.kv("Embedding", "[dim]unavailable[/dim]")


def _wait_for_server(
    url: str, timeout: float = 15.0, *, show_spinner: bool = True
) -> dict[str, Any] | None:
    """Poll until the daemon responds to /status. Returns the status dict or None."""
    client = ShrikeClient(url, autostart=False)
    deadline = time.monotonic() + timeout

    with output.spinner("Starting server…") if show_spinner else contextlib.nullcontext():
        while time.monotonic() < deadline:
            status = client.server_status()
            if status is not None:
                return status
            time.sleep(0.2)
    return None


def ensure_server(config: dict[str, Any]) -> str:
    """Start the daemon if it isn't already running. Returns the server URL.

    Uses collection path, host, port, and logging settings from *config*.
    Raises ``click.ClickException`` if the server cannot be started
    (e.g. no collection path configured).
    """
    from shrike.cli.config import resolve_collection, resolve_url

    url = resolve_url(config)

    if is_server_alive():
        meta = read_server_meta()
        if meta:
            return str(meta.get("url", url))
        return url

    # Clean up any stale state from a crashed server
    cleanup_state()

    collection_path = resolve_collection(config)
    if not collection_path:
        raise click.ClickException(
            "Cannot auto-start server: no collection path configured.\n\n"
            "Provide one with:\n"
            "  shrike server start --collection /path/to/collection.anki2\n"
            "  SHRIKE_COLLECTION environment variable\n"
            "  'collection' key in config file"
        )

    collection_dir = Path(collection_path).parent
    collection_dir.mkdir(parents=True, exist_ok=True)

    server_config = config.get("server", {})
    server_host = server_config.get("host", "127.0.0.1")
    server_port = server_config.get("port", 8372)
    url = f"http://{server_host}:{server_port}/mcp"

    log_config = config.get("logging", {})
    resolved_log_dir = str(Path(log_config.get("dir") or str(DEFAULT_LOG_DIR)).expanduser())
    resolved_log_level = log_config.get("level", "info")

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    bootstrap_log = Path(resolved_log_dir)
    bootstrap_log.mkdir(parents=True, exist_ok=True)
    bootstrap_log_file = open(bootstrap_log / "shrike-bootstrap.log", "a")  # noqa: SIM115

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
            *_embedding_args(config),
        ],
        stdout=bootstrap_log_file,
        stderr=bootstrap_log_file,
        start_new_session=True,
    )

    if _wait_for_server(url, show_spinner=False) is None and proc.poll() is not None:
        cleanup_state()
        log_file = get_log_file(config, log_dir_override=resolved_log_dir)
        raise click.ClickException(
            f"Auto-started server exited with code {proc.returncode}.\nCheck log: {log_file}"
        )

    return url


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
@click.pass_context
def server_start(
    ctx: click.Context,
    collection: str | None,
    port: int | None,
    host: str | None,
    foreground: bool,
    log_dir: str | None,
    log_level: str | None,
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
        ]
        from shrike.server import main

        main()
        return

    # Daemon mode
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    bootstrap_log = Path(resolved_log_dir)
    bootstrap_log.mkdir(parents=True, exist_ok=True)
    bootstrap_log_file = open(bootstrap_log / "shrike-bootstrap.log", "a")  # noqa: SIM115

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
            *_embedding_args(config),
        ],
        stdout=bootstrap_log_file,
        stderr=bootstrap_log_file,
        start_new_session=True,
    )

    json_out: bool = ctx.obj["json"]

    # Save config if it doesn't exist yet
    config_path = ctx.obj.get("config_path")
    if config_path and not config_path.exists():
        config["collection"] = collection_path
        config["server"]["host"] = server_host
        config["server"]["port"] = server_port
        saved = save_config(config, config_path)
        if not json_out:
            output.console.print(f"  [dim]Config saved to {saved}[/dim]")

    log_file = get_log_file(config, log_dir_override=resolved_log_dir)
    status = _wait_for_server(url)
    if status is not None:
        if json_out:
            status["log"] = str(log_file)
            output.emit_json(status)
        else:
            status["log"] = str(log_file)
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
    client = ShrikeClient(url, autostart=False)
    with output.spinner("Checking server…"):
        status = client.server_status()

        if status is None:
            status = server_status()
            if status["running"]:
                status["responsive"] = False

    if ctx.obj["json"]:
        output.emit_json(status)
        if not status.get("running"):
            ctx.exit(1)
        return

    if status.get("log_dir"):
        status["log"] = str(Path(status["log_dir"]) / "shrike.log")

    _render_status(status)

    if not status.get("running"):
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
