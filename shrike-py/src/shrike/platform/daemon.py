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
import hashlib
import json
import logging
import os
import stat
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


def control_socket_path(sd: Path) -> Path:
    """The control-plane Unix socket path for a daemon rooted at state dir ``sd``.

    AF_UNIX paths are length-limited (~104 bytes on macOS, 108 on Linux), and a
    state dir under a deep temp/profile path would overflow that — so the socket
    does NOT live in the state dir. It lives in a short, user-private runtime dir
    (``$XDG_RUNTIME_DIR`` when set, else a ``0700`` per-uid dir under ``/tmp``),
    under a name derived (lexically, so existence-independent) from ``sd``. The
    name is deterministic so a restart and ``cleanup_state`` recompute the same
    path to unlink a stale socket; the full path is also recorded in server.json
    for client discovery. POSIX only — Windows uses a loopback-TCP control plane.
    """
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg and os.path.isdir(xdg):
        runtime = Path(xdg)  # short and user-private (0700) by the XDG spec
    else:
        runtime = Path("/tmp") / f"shrike-{os.getuid()}"
        runtime.mkdir(mode=0o700, exist_ok=True)
        info = runtime.lstat()
        # Refuse a pre-existing dir we don't own or that's a symlink — the classic
        # /tmp hijack: an attacker who owns the path could read the socket.
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError(f"refusing unsafe control-socket dir {runtime} (wrong owner/type)")
        if stat.S_IMODE(info.st_mode) != 0o700:
            os.chmod(runtime, 0o700)
    digest = hashlib.sha1(os.path.abspath(str(sd)).encode()).hexdigest()[:10]
    return runtime / f"shrike-{digest}.sock"


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


def control_channel(meta: dict[str, Any] | None) -> tuple[str, str | None]:
    """Resolve the control-plane ``(base_url, uds_path)`` from server metadata.

    The privileged control routes (shutdown/reload/index/embedding, full status)
    live on a separate always-local listener; ``server.json`` records its address
    under ``control``. Returns the base URL to issue control requests against and
    the Unix-socket path (``None`` for the loopback-TCP fallback). Falls back to
    the *data* plane's base URL when the metadata predates the plane split, so a
    stale daemon mid-upgrade still stops cleanly.
    """
    control = meta.get("control") if meta else None
    if isinstance(control, dict):
        uds = control.get("uds")
        if uds:
            # httpx routes by the transport's socket; the base host is a synthetic
            # authority the UDS control guard ignores.
            return "http://localhost", str(uds)
        url = control.get("url")
        if url:
            return str(url).rstrip("/"), None
    data_url = meta.get("url") if meta else None
    if data_url:
        return str(data_url).rsplit("/", 1)[0], None
    return "", None


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
    """Remove PID/metadata files and the control socket (not the lock file —
    that's managed by filelock). Removing a stale control socket here keeps a
    crashed daemon's socket from lingering."""
    sd = sd or _DEFAULT_STATE_DIR
    files = [_pid_file(sd), _meta_file(sd)]
    # The control socket is POSIX-only and lives outside the state dir (a short
    # runtime dir); Windows has none. Prefer the path the daemon *recorded* in
    # server.json (authoritative even if XDG_RUNTIME_DIR differs in this process),
    # read before the meta file is unlinked below; recompute only as a fallback,
    # and never let that recompute (which can raise on a hijacked runtime dir)
    # abort cleanup.
    if sys.platform != "win32":
        sock: str | None = None
        meta = read_server_meta(sd)
        if meta:
            _, sock = control_channel(meta)
        if sock is None:
            with contextlib.suppress(Exception):
                sock = str(control_socket_path(sd))
        if sock is not None:
            files.append(Path(sock))
    for f in files:
        with contextlib.suppress(OSError):
            f.unlink(missing_ok=True)


def _request_http_shutdown(meta: dict[str, Any] | None) -> bool:
    """POST /shutdown on the control plane. Returns True if accepted.

    /shutdown is a privileged control route, so it goes over the control channel
    (a Unix socket, or the loopback-TCP fallback) resolved from ``meta`` — falling
    back to the data base URL for a pre-split daemon.
    """
    import httpx

    base, uds = control_channel(meta)
    if not base:
        return False
    shutdown_url = f"{base}/shutdown"
    try:
        if uds is not None:
            with httpx.Client(transport=httpx.HTTPTransport(uds=uds)) as client:
                resp = client.post(shutdown_url, timeout=5.0)
        else:
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


def stop_server(timeout: float = 5.0, sd: Path | None = None) -> dict[str, Any]:
    """Stop a running server. Returns status dict.

    ``sd`` targets a daemon at a non-default state dir (its lock/meta/control all
    live there); ``None`` is the platform default.

    Shutdown strategy (cross-platform):
      1. HTTP POST /shutdown — clean, works on all platforms
      2. SIGTERM (Unix only) — fallback if HTTP fails
      3. SIGKILL / TerminateProcess — last resort if server is hung
    """
    resolved = sd or _DEFAULT_STATE_DIR
    if not is_server_alive(sd):
        if _meta_file(resolved).exists() or _pid_file(resolved).exists():
            cleanup_state(sd)
            return {"stopped": False, "reason": "not running (cleaned stale state)"}
        return {"stopped": False, "reason": "not running"}

    meta = read_server_meta(sd)
    pid = read_pid(sd)

    # 1. Try HTTP shutdown over the control plane (works on all platforms)
    if _request_http_shutdown(meta):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not is_server_alive(sd):
                cleanup_state(sd)
                return {"stopped": True, "pid": pid, "forced": False}
            time.sleep(0.1)
        logger.warning("HTTP shutdown accepted but server did not exit within %ss", timeout)

    # 2. Try SIGTERM (Unix only, no-op on Windows)
    if pid is not None and _signal_term(pid):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not is_server_alive(sd):
                cleanup_state(sd)
                return {"stopped": True, "pid": pid, "forced": False}
            time.sleep(0.1)
        logger.warning("SIGTERM sent but server did not exit within %ss", timeout)

    # 3. Force kill as last resort
    if pid is not None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            _force_kill(pid)
        time.sleep(0.5)

    cleanup_state(sd)
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
