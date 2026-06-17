"""Daemon lifecycle management using file locks.

The server holds an exclusive file lock (server.lock) for its entire lifetime.
When the server exits — cleanly or via crash/kill — the OS releases the lock.
Clients detect server liveness by attempting to acquire the same lock:
  - If acquisition succeeds: the server is dead (lock released). Clean up stale
    state and optionally restart.
  - If acquisition fails (locked): the server is alive.

This eliminates PID recycling issues entirely. The PID file is kept as a
convenience for diagnostics, but liveness is determined by the lock alone.

The metadata file (server.json) stores connection details so clients know where
to reach the running server without hardcoding ports.

Shutdown uses HTTP first (POST /shutdown on the running server), which works
on all platforms. Signal-based shutdown (SIGTERM/SIGKILL) is a fallback for
when the HTTP endpoint is unresponsive.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from shrike.platform.paths import state_dir

logger = logging.getLogger("shrike.daemon")

_DEFAULT_STATE_DIR = state_dir()


def _lock_file(sd: Path) -> Path:
    return sd / "server.lock"


def _pid_file(sd: Path) -> Path:
    return sd / "server.pid"


def _meta_file(sd: Path) -> Path:
    return sd / "server.json"


# Module-level aliases for the default state dir (used by CLI commands)
STATE_DIR = _DEFAULT_STATE_DIR
LOCK_FILE = _lock_file(STATE_DIR)
PID_FILE = _pid_file(STATE_DIR)
META_FILE = _meta_file(STATE_DIR)


class ServerLock:
    """Held by the running server process to advertise liveness.

    Usage in the server:
        lock = ServerLock()
        lock.acquire(meta={...})  # blocks if another server is running
        ...  # run server
        lock.release()            # called on clean shutdown (also released by OS on crash)

    Pass state_dir to override the directory for lock/pid/meta files (used by tests).
    """

    def __init__(self, state_dir_override: Path | None = None) -> None:
        self._state_dir = state_dir_override or _DEFAULT_STATE_DIR
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(_lock_file(self._state_dir)), timeout=0)
        self._acquired = False

    def acquire(self, meta: dict[str, Any]) -> None:
        """Acquire the server lock. Raises AlreadyRunningError if held."""
        try:
            self._lock.acquire()
        except Timeout as err:
            existing = read_server_meta(self._state_dir)
            pid = existing.get("pid", "?") if existing else "?"
            raise AlreadyRunningError(f"Another server is already running (PID {pid})") from err

        self._acquired = True
        _pid_file(self._state_dir).write_text(str(os.getpid()))
        _meta_file(self._state_dir).write_text(json.dumps(meta, indent=2))

    def release(self) -> None:
        """Release the lock and clean up state files."""
        if self._acquired:
            cleanup_state(self._state_dir)
            self._lock.release()
            self._acquired = False

    def __enter__(self) -> ServerLock:
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


class AlreadyRunningError(Exception):
    """Raised when attempting to start a server that's already running."""


def is_server_alive(sd: Path | None = None) -> bool:
    """Check if a server currently holds the lock.

    Attempts a non-blocking lock acquisition. If it succeeds, the server
    is dead (we release immediately). If it fails, the server is alive.
    """
    sd = sd or _DEFAULT_STATE_DIR
    sd.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(_lock_file(sd)), timeout=0)
    try:
        lock.acquire()
        lock.release()
        return False
    except Timeout:
        return True


def read_server_meta(sd: Path | None = None) -> dict[str, Any] | None:
    """Read the server metadata file, or None if missing/corrupt."""
    mf = _meta_file(sd or _DEFAULT_STATE_DIR)
    if not mf.exists():
        return None
    try:
        result: dict[str, Any] = json.loads(mf.read_text())
        return result
    except (json.JSONDecodeError, OSError):
        return None


def read_pid(sd: Path | None = None) -> int | None:
    """Read the PID from the PID file, or None if missing/invalid."""
    pf = _pid_file(sd or _DEFAULT_STATE_DIR)
    if not pf.exists():
        return None
    try:
        return int(pf.read_text().strip())
    except (ValueError, OSError):
        return None


def cleanup_state(sd: Path | None = None) -> None:
    """Remove PID and metadata files (not the lock file — that's managed by filelock)."""
    sd = sd or _DEFAULT_STATE_DIR
    for f in (_pid_file(sd), _meta_file(sd)):
        with contextlib.suppress(OSError):
            f.unlink(missing_ok=True)


def _request_http_shutdown(url: str) -> bool:
    """POST /shutdown to the server. Returns True if accepted."""
    import httpx

    shutdown_url = url.rsplit("/", 1)[0] + "/shutdown"
    try:
        resp = httpx.post(shutdown_url, timeout=5.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def _force_kill(pid: int) -> None:
    """Terminate a process using platform-appropriate hard kill."""
    import signal

    if sys.platform == "win32":
        os.kill(pid, signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGKILL)


def _signal_term(pid: int) -> bool:
    """Send SIGTERM on Unix. No-op on Windows (returns False)."""
    if sys.platform == "win32":
        return False
    import signal

    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return False


def stop_server(timeout: float = 5.0) -> dict[str, Any]:
    """Stop a running server. Returns status dict.

    Shutdown strategy (cross-platform):
      1. HTTP POST /shutdown — clean, works on all platforms
      2. SIGTERM (Unix only) — fallback if HTTP fails
      3. SIGKILL / TerminateProcess — last resort if server is hung
    """
    if not is_server_alive():
        if META_FILE.exists() or PID_FILE.exists():
            cleanup_state()
            return {"stopped": False, "reason": "not running (cleaned stale state)"}
        return {"stopped": False, "reason": "not running"}

    meta = read_server_meta()
    pid = read_pid()
    url = meta.get("url") if meta else None

    # 1. Try HTTP shutdown (works on all platforms)
    if url and _request_http_shutdown(url):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not is_server_alive():
                cleanup_state()
                return {"stopped": True, "pid": pid, "forced": False}
            time.sleep(0.1)
        logger.warning("HTTP shutdown accepted but server did not exit within %ss", timeout)

    # 2. Try SIGTERM (Unix only, no-op on Windows)
    if pid is not None and _signal_term(pid):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not is_server_alive():
                cleanup_state()
                return {"stopped": True, "pid": pid, "forced": False}
            time.sleep(0.1)
        logger.warning("SIGTERM sent but server did not exit within %ss", timeout)

    # 3. Force kill as last resort
    if pid is not None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            _force_kill(pid)
        time.sleep(0.5)

    cleanup_state()
    return {"stopped": True, "pid": pid, "forced": True}


def server_status() -> dict[str, Any]:
    """Get the current server status."""
    alive = is_server_alive()
    meta = read_server_meta()
    pid = read_pid()

    if not alive:
        if meta or PID_FILE.exists():
            cleanup_state()
        return {"running": False}

    result: dict[str, Any] = {"running": True, "pid": pid}
    if meta:
        result["url"] = meta.get("url")
        result["collection"] = meta.get("collection")
        result["log_level"] = meta.get("log_level")
        result["log_dir"] = meta.get("log_dir")
        result["started"] = meta.get("started")

        started = meta.get("started", "")
        if started:
            with contextlib.suppress(ValueError):
                start_dt = datetime.fromisoformat(started)
                delta = datetime.now(UTC) - start_dt
                hours, remainder = divmod(int(delta.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                if hours:
                    result["uptime"] = f"{hours}h {minutes}m"
                elif minutes:
                    result["uptime"] = f"{minutes}m {seconds}s"
                else:
                    result["uptime"] = f"{seconds}s"

    return result
