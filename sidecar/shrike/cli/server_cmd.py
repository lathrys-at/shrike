from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click

from shrike.cli import output
from shrike.cli.client import ShrikeClient
from shrike.cli.config import resolve_collection, save_config

STATE_DIR = Path("~/.local/state/shrike").expanduser()
PID_FILE = STATE_DIR / "server.pid"
META_FILE = STATE_DIR / "server.json"
LOG_FILE = STATE_DIR / "server.log"


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


def _read_meta() -> dict | None:
    if not META_FILE.exists():
        return None
    try:
        return json.loads(META_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _cleanup_state():
    for f in (PID_FILE, META_FILE):
        try:
            f.unlink(missing_ok=True)
        except OSError:
            pass


def _wait_for_server(url: str, timeout: float = 15.0) -> bool:
    client = ShrikeClient(url)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.ping():
            return True
        time.sleep(0.2)
    return False


@click.group("server", short_help="Manage the Shrike daemon")
def server():
    """Start, stop, and check the status of the Shrike MCP server."""
    pass


@server.command("start", short_help="Start the MCP server")
@click.option(
    "--collection",
    type=click.Path(),
    help="Path to the Anki collection file (collection.anki2).",
)
@click.option("--port", type=int, help="Port to listen on (default: 8372).")
@click.option("--host", help="Host to bind to (default: 127.0.0.1).")
@click.option(
    "--foreground", is_flag=True, help="Run in the foreground instead of daemonizing."
)
@click.pass_context
def server_start(ctx, collection, port, host, foreground):
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
        click.echo(
            f"Starting server in foreground on {server_host}:{server_port}"
        )
        click.echo(f"Collection: {collection_path}")
        click.echo("Press Ctrl+C to stop.\n")
        # Import and run directly
        sys.argv = [
            "shrike-server",
            "--collection", collection_path,
            "--port", str(server_port),
            "--host", server_host,
        ]
        from shrike.server import main
        main()
        return

    # Daemon mode
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_FILE, "a")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "shrike.server",
            "--collection", collection_path,
            "--port", str(server_port),
            "--host", server_host,
        ],
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )

    # Write PID and metadata
    PID_FILE.write_text(str(proc.pid))
    META_FILE.write_text(json.dumps({
        "pid": proc.pid,
        "url": url,
        "host": server_host,
        "port": server_port,
        "collection": collection_path,
        "started": datetime.now(timezone.utc).isoformat(),
        "log": str(LOG_FILE),
    }, indent=2))

    # Save config if it doesn't exist yet
    config_path = ctx.obj.get("config_path")
    if config_path and not config_path.exists():
        config["collection"] = collection_path
        config["server"]["host"] = server_host
        config["server"]["port"] = server_port
        saved = save_config(config, config_path)
        click.echo(click.style(f"  Config saved to {saved}", dim=True))

    # Wait for server to come up
    click.echo(f"Starting server (PID {proc.pid})...")

    if _wait_for_server(url):
        output.success(f"Server running at {url}")
        output.kv("Collection", collection_path, indent=2)
        output.kv("Log", str(LOG_FILE), indent=2)
    else:
        # Check if process died
        if proc.poll() is not None:
            _cleanup_state()
            raise click.ClickException(
                f"Server process exited with code {proc.returncode}.\n"
                f"Check log: {LOG_FILE}"
            )
        click.echo(click.style(
            "Server started but not yet responding. Check log for details:\n"
            f"  {LOG_FILE}",
            fg="yellow",
        ))


@server.command("stop", short_help="Stop the running server")
@click.pass_context
def server_stop(ctx):
    """Stop the Shrike MCP server daemon."""
    pid = _read_pid()
    if pid is None:
        meta = _read_meta()
        if meta:
            _cleanup_state()
            click.echo("Server is not running (cleaned up stale state).")
        else:
            click.echo("Server is not running.")
        return

    click.echo(f"Stopping server (PID {pid})...")
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
        # Force kill
        click.echo(click.style("Graceful shutdown timed out, forcing...", fg="yellow"))
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    _cleanup_state()
    output.success("Server stopped.")


@server.command("status", short_help="Show server status")
@click.pass_context
def server_status(ctx):
    """Check whether the Shrike MCP server is running."""
    pid = _read_pid()
    meta = _read_meta()

    if pid is None:
        if meta:
            _cleanup_state()
        click.echo(click.style("Server is not running.", dim=True))
        ctx.exit(1)
        return

    click.echo(click.style("Server is running", fg="green", bold=True))
    if meta:
        output.kv("URL", meta.get("url", "unknown"), indent=2)
        output.kv("PID", meta.get("pid", pid), indent=2)
        output.kv("Collection", meta.get("collection", "unknown"), indent=2)
        started = meta.get("started", "")
        if started:
            try:
                start_dt = datetime.fromisoformat(started)
                delta = datetime.now(timezone.utc) - start_dt
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
        output.kv("Log", meta.get("log", "unknown"), indent=2)
