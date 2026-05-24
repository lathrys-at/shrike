from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from shrike.cli import output
from shrike.cli.client import ShrikeClient
from shrike.cli.config import resolve_collection, save_config
from shrike.log import DEFAULT_LOG_DIR, get_log_file, parse_log_line, style_log_line

STATE_DIR = Path("~/.local/state/shrike").expanduser()
PID_FILE = STATE_DIR / "server.pid"
META_FILE = STATE_DIR / "server.json"


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        return pid if _is_process_alive(pid) else None
    except (ValueError, OSError):
        return None


def _read_meta() -> dict[str, Any] | None:
    if not META_FILE.exists():
        return None
    try:
        result: dict[str, Any] = json.loads(META_FILE.read_text())
        return result
    except (json.JSONDecodeError, OSError):
        return None


def _cleanup_state() -> None:
    for f in (PID_FILE, META_FILE):
        with contextlib.suppress(OSError):
            f.unlink(missing_ok=True)


def _wait_for_server(url: str, timeout: float = 15.0) -> bool:
    client = ShrikeClient(url)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.ping():
            return True
        time.sleep(0.2)
    return False


@click.group("server", short_help="Manage the Shrike daemon")
def server() -> None:
    """Start, stop, and check the status of the Shrike MCP server."""


@server.command("start", short_help="Start the MCP server")
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
    help="Directory for log files (default: ~/.local/state/shrike/logs).",
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

    # Check if already running
    existing_pid = _read_pid()
    if existing_pid is not None:
        meta = _read_meta()
        existing_url = meta.get("url", "unknown") if meta else "unknown"
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

    # The server process handles its own log files via RotatingFileHandler.
    # We still capture stderr to a bootstrap log in case the process crashes
    # before logging is configured.
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
        ],
        stdout=bootstrap_log_file,
        stderr=bootstrap_log_file,
        start_new_session=True,
    )

    # Write PID and metadata
    PID_FILE.write_text(str(proc.pid))
    META_FILE.write_text(
        json.dumps(
            {
                "pid": proc.pid,
                "url": url,
                "host": server_host,
                "port": server_port,
                "collection": collection_path,
                "log_dir": resolved_log_dir,
                "log_level": resolved_log_level,
                "started": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )

    # Save config if it doesn't exist yet
    config_path = ctx.obj.get("config_path")
    if config_path and not config_path.exists():
        config["collection"] = collection_path
        config["server"]["host"] = server_host
        config["server"]["port"] = server_port
        saved = save_config(config, config_path)
        output.console.print(f"  [dim]Config saved to {saved}[/dim]")

    # Wait for server to come up
    output.console.print(f"Starting server (PID {proc.pid})...")

    log_file = get_log_file(config, log_dir_override=resolved_log_dir)
    if _wait_for_server(url):
        output.success(f"Server running at {url}")
        output.kv("Collection", collection_path, indent=2)
        output.kv("Log", str(log_file), indent=2)
        output.kv("Level", resolved_log_level, indent=2)
    else:
        if proc.poll() is not None:
            _cleanup_state()
            raise click.ClickException(
                f"Server process exited with code {proc.returncode}.\nCheck log: {log_file}"
            )
        output.console.print(
            "[yellow]Server started but not yet responding. Check log for details:[/yellow]\n"
            f"  {log_file}"
        )


@server.command("stop", short_help="Stop the running server")
@click.pass_context
def server_stop(ctx: click.Context) -> None:
    """Stop the Shrike MCP server daemon."""
    pid = _read_pid()
    if pid is None:
        meta = _read_meta()
        if meta:
            _cleanup_state()
            output.console.print("Server is not running (cleaned up stale state).")
        else:
            output.console.print("Server is not running.")
        return

    output.console.print(f"Stopping server (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _cleanup_state()
        output.success("Server already stopped.")
        return

    # Wait for graceful shutdown
    for _ in range(50):  # 5 seconds
        if not _is_process_alive(pid):
            break
        time.sleep(0.1)
    else:
        output.console.print("[yellow]Graceful shutdown timed out, forcing...[/yellow]")
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)

    _cleanup_state()
    output.success("Server stopped.")


@server.command("status", short_help="Show server status")
@click.pass_context
def server_status(ctx: click.Context) -> None:
    """Check whether the Shrike MCP server is running."""
    pid = _read_pid()
    meta = _read_meta()

    if pid is None:
        if meta:
            _cleanup_state()
        output.console.print("[dim]Server is not running.[/dim]")
        ctx.exit(1)
        return

    output.console.print("[bold green]Server is running[/bold green]")
    if meta:
        output.kv("URL", meta.get("url", "unknown"), indent=2)
        output.kv("PID", meta.get("pid", pid), indent=2)
        output.kv("Collection", meta.get("collection", "unknown"), indent=2)
        output.kv("Log level", meta.get("log_level", "info"), indent=2)

        # Show log file path
        log_dir = meta.get("log_dir")
        if log_dir:
            output.kv("Log", str(Path(log_dir) / "shrike.log"), indent=2)

        started = meta.get("started", "")
        if started:
            try:
                start_dt = datetime.fromisoformat(started)
                delta = datetime.now(UTC) - start_dt
                hours, remainder = divmod(int(delta.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                if hours:
                    uptime = f"{hours}h {minutes}m"
                elif minutes:
                    uptime = f"{minutes}m {seconds}s"
                else:
                    uptime = f"{seconds}s"
                output.kv("Uptime", uptime, indent=2)
            except ValueError:
                pass


@server.command("logs", short_help="View server logs")
@click.option("--follow", "-f", is_flag=True, help="Follow the log output (like tail -f).")
@click.option("--lines", "-n", type=int, default=50, help="Number of lines to show (default: 50).")
@click.option(
    "--process",
    "-p",
    type=click.Choice(["shrike", "llama"], case_sensitive=False),
    default="shrike",
    help="Which process log to view (default: shrike).",
)
@click.option("--json-output", "--json", "json_out", is_flag=True, help="Output as JSON array.")
@click.option("--pretty/--no-pretty", default=True, help="Styled output (default: --pretty).")
@click.pass_context
def server_logs(
    ctx: click.Context,
    follow: bool,
    lines: int,
    process: str,
    json_out: bool,
    pretty: bool,
) -> None:
    """View the server log output.

    Reads from the log file by default, or from stdin if piped.

    \b
    Examples:
      shrike server logs
      shrike server logs -f
      shrike server logs -n 100
      shrike server logs --json
      shrike server logs --no-pretty
      cat ~/.local/state/shrike/logs/shrike.log | shrike server logs --json
    """
    if json_out and follow:
        raise click.ClickException("--json and --follow cannot be used together.")

    reading_stdin = not sys.stdin.isatty()

    if reading_stdin:
        input_lines = sys.stdin.read().splitlines()
        if json_out:
            _emit_json(input_lines)
        else:
            for line in input_lines:
                _emit_line(line, pretty=pretty)
        return

    # Reading from log file
    config = ctx.obj["config"]
    meta = _read_meta()

    log_dir = None
    if meta:
        log_dir = meta.get("log_dir")
    if not log_dir:
        log_dir = config.get("logging", {}).get("dir")

    log_file = get_log_file(config, log_dir_override=log_dir, process_name=process)

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
    """Parse log lines and emit as a JSON array."""
    records: list[dict[str, str]] = []
    for line in lines:
        parsed = parse_log_line(line)
        if parsed is not None:
            records.append(parsed)
    output.emit_json(records)


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
