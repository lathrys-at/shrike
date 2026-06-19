from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import ipaddress
import logging
import os
import signal
import socket
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from starlette.applications import Starlette

import shrike_native
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools.base import Tool
from mcp.server.transport_security import (
    TransportSecurityMiddleware,
    TransportSecuritySettings,
)
from pydantic import ValidationError

from shrike.api.tools import ToolInputError, register_tools
from shrike.harness.collection import DEFAULT_LOCK_HOLD
from shrike.harness.engines.embedding.runtime import (
    BACKEND_ALIASES,
    DEFAULT_BACKEND,
    SUPPORTED_BACKENDS,
    EmbeddingRuntime,
)
from shrike.harness.harness import CollectionManager, Harness, HarnessParams, KernelConfigError
from shrike.platform.daemon import AlreadyRunningError, ServerLock, control_socket_path
from shrike.platform.driven_runtime import DrivenRuntime
from shrike.platform.log import configure_logging
from shrike.platform.paths import cache_dir, state_dir
from shrike.platform.pathsafety import (
    is_loopback,
    server_is_purely_local,
    validate_path_root,
)
from shrike.schemas import WIRE_PROTOCOL_VERSION, ActionError, ActionErrorCode
from shrike.server._mcp_perf import install_validator_cache

# The kernel saver's built-in flush tuning: the --index-save-* help names the
# defaults the flags would override. Sourced from the kernel.
DEFAULT_SAVE_DELAY = float(shrike_native.INDEX_SAVE_DELAY_DEFAULT)
DEFAULT_SAVE_THRESHOLD = int(shrike_native.INDEX_SAVE_THRESHOLD_DEFAULT)

# The actions-over-HTTP wire-version header. Every /actions/* response echoes
# the server's WIRE_PROTOCOL_VERSION; a request may carry the same header to
# assert it speaks the same fabric (a mismatch is refused, the minimum handshake
# a separately-shipped client needs). /status already reports the version in its
# body — this is the per-call header form.
WIRE_VERSION_HEADER = "X-Shrike-Wire-Version"

logger = logging.getLogger("shrike.server")


def _make_image_resolver(
    media_dir: str,
) -> tuple[Callable[[str], bytes | None], Callable[[str], bool]]:
    """A ``(read, exists)`` pair over the media dir for the index's image resolver.

    Closes over the (lock-free, path-derived) media dir so the index can read image bytes on its
    own embed thread without touching the Anki collection. Both sanitize to a basename inside the
    dir (``_safe_media_name``), so a name can only ever resolve inside the media folder. ``exists``
    is a cheap stat (no byte read) the index folds into the per-note hash, so an image stored after
    its note re-embeds on reconcile instead of being skipped.
    """
    from shrike.harness.collection import _safe_media_name

    def _path(name: str) -> str | None:
        safe = _safe_media_name(name)
        return os.path.join(media_dir, safe) if safe else None

    def _read(name: str) -> bytes | None:
        path = _path(name)
        if path is None:
            return None
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            return None

    def _exists(name: str) -> bool:
        path = _path(name)
        return path is not None and os.path.isfile(path)

    return _read, _exists


# Host/Origin values accepted when the server is bound to a loopback address.
# The port is wildcarded (`:*`) so any port the user picked is allowed; matching
# is exact on the host part, which is what stops DNS-rebinding (a page on
# ``evil.com`` resolving to 127.0.0.1 still sends ``Host: evil.com``).
_LOOPBACK_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOOPBACK_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


def _host_header_form(host: str) -> str:
    """The Host-header spelling of a bind host (IPv6 bracketed), for the allow-list.

    ``is_loopback`` accepts *any* 127/8 address (or an expanded ``::1``), so a bind
    Shrike happily accepts as loopback (e.g. ``--host 127.0.0.2``) may not be one of
    the fixed ``_LOOPBACK_HOSTS`` literals. A client then sends ``Host: 127.0.0.2:PORT``
    and the guard rejects it (HTTP 421) — the server is reachable but answers nothing.
    Folding the actual bind host into the allow-list closes that self-brick.

    The Host header carries an IPv6 literal in brackets (``Host: [::1]:8372``), so an
    IPv6 bind host is canonicalized (``ipaddress``) and bracketed; an IPv4 address or a
    plain name is passed through. The port is wildcarded (``:*``) to match every entry.
    """
    bare = host.strip("[]")
    try:
        addr = ipaddress.ip_address(bare)
    except ValueError:
        return f"{host}:*"  # a name (e.g. "localhost") — pass through verbatim
    if addr.version == 6:
        return f"[{addr.compressed}]:*"
    return f"{addr.compressed}:*"


def _positive_int(raw: str) -> int:
    """argparse type: a >= 1 integer (e.g. --embedding-batch-size)."""
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1 (got {value})")
    return value


def _build_transport_security(
    host: str,
    *,
    allowed_hosts: list[str] | None = None,
    allowed_origins: list[str] | None = None,
    disable: bool = False,
) -> TransportSecuritySettings | None:
    """DNS-rebinding / CSRF protection settings, or ``None`` to disable the guard.

    The guard is *independent* of the bind address — binding loopback vs. a
    network interface only decides who can reach the port, not which Host/Origin
    headers are trusted. The guard defends against a browser on the same machine
    (or a DNS-rebinding page) scripting requests; the deployment's own boundary
    (a reverse proxy, a VPN/tailnet, a firewall) is a separate, often sufficient,
    layer of trust.

    Returns ``None`` (no validation) when:
    - ``disable`` is set — the operator declares the network is the trust boundary
      (Shrike behind Caddy / on a tailnet / firewalled); or
    - the bind is non-loopback and no explicit ``allowed_hosts``/``allowed_origins``
      were given — a deliberately network-bound server has no fixed Host set to
      validate against (the ``--allow-remote`` behaviour).

    Otherwise returns settings allow-listing the loopback Host/Origin values (when
    bound to loopback) plus any explicit additions — so a loopback server behind a
    local proxy can trust the proxy's forwarded hostname without opening up. The
    *actual* bind host is always folded in too: ``is_loopback`` accepts any 127/8
    address, so a bind Shrike happily accepts (e.g. ``--host 127.0.0.2``) that isn't
    one of the fixed loopback literals must still answer its own ``Host`` header
    rather than self-brick with HTTP 421.
    """
    extra_hosts = allowed_hosts or []
    extra_origins = allowed_origins or []

    if disable:
        return None
    if not _is_loopback(host) and not extra_hosts and not extra_origins:
        return None

    base_hosts = list(_LOOPBACK_HOSTS) if _is_loopback(host) else []
    base_origins = list(_LOOPBACK_ORIGINS) if _is_loopback(host) else []
    # Fold the actual loopback bind host into the allow-list when it isn't already a
    # fixed literal (e.g. 127.0.0.2) — a loopback bind the server accepts must stay
    # reachable. Non-loopback binds stay fail-closed: only the explicit allow-list
    # is trusted, never the bind interface itself.
    if _is_loopback(host):
        bind_host = _host_header_form(host)
        if bind_host not in base_hosts:
            base_hosts.append(bind_host)
            base_origins.append(f"http://{bind_host}")
    allow_hosts = base_hosts + extra_hosts
    if not allow_hosts:
        # Rebinding protection on with an empty Host allow-list rejects *every*
        # request's Host (421) — the server is reachable but answers nothing. Only
        # happens on a non-loopback bind given allowed_origins but no allowed_hosts;
        # surface the footgun at startup rather than as a silently-bricked server.
        logger.warning(
            "Transport guard is on but no allowed Host values are configured "
            "(non-loopback bind with --allowed-origin but no --allowed-host): every "
            "request will be rejected with HTTP 421. Add --allowed-host or pass "
            "--no-dns-rebinding-protection."
        )
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allow_hosts,
        allowed_origins=base_origins + extra_origins,
    )


# The server-local path-safety mechanism is shared across capabilities
# (store_media, export, import) — the generic helpers live in shrike.pathsafety.
# These module-level aliases keep the server.py names that call sites and tests
# import; the per-capability *policy* (which root list, which gate) stays at the
# call sites.
_is_loopback = is_loopback
_server_is_purely_local = server_is_purely_local
_validate_media_path_root = validate_path_root


# POST /embedding/start overrides that choose WHAT the embedding backend
# executes or loads: the binary (llama_server → arbitrary subprocess), the model
# file it deserializes, the verbatim subprocess args (extra_args), the onnx
# execution-provider shared libraries, and the backend kind that selects among
# them. On a purely-local server the loopback caller IS the operator (the same
# trust as editing the config or running the binary directly), so these stay
# settable — the CLI's `shrike embedding start --llama-server/--embedding-model`
# flow rides this. On any server that is NOT purely-local the loopback peer may
# be a reverse proxy / tailnet fronting a remote client, so an override here is
# refused and the daemon falls back to its boot-configured embedding settings —
# a remote caller can trigger a start but never redirect the spawned process.
# The remaining body keys
# (port/context_size/threads/gpu_layers/pooling/batch_size) are runtime knobs,
# not execution-shaping, and pass through on any bind.
_EXEC_SHAPING_OVERRIDES = ("backend", "model", "llama_server", "extra_args", "onnx_providers")


def _rejected_exec_overrides(overrides: dict[str, Any], *, purely_local: bool) -> list[str]:
    """Execution-shaping override keys a non-purely-local server must refuse.

    Empty when the server is purely-local (the loopback caller is the trusted
    operator) or the body carried none of :data:`_EXEC_SHAPING_OVERRIDES`. A
    non-empty result is the set to reject — the daemon then starts with its
    boot-configured settings rather than the caller's chosen binary/model/args.
    """
    if purely_local:
        return []
    return [k for k in _EXEC_SHAPING_OVERRIDES if k in overrides]


def _make_guard(
    security: TransportSecuritySettings | None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """A per-plane route decorator: validate Host/Origin under ``security`` (a
    no-op when ``None``) and emit the one-INFO-line-per-served-call.

    Each listener carries its own settings — the data plane honors the operator's
    exposure flags, the control plane is pinned local — so the middleware is built
    once per plane here and closed over, rather than shared.
    """
    security_mw = TransportSecurityMiddleware(security)

    def guard(
        handler: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(handler)
        async def wrapped(request: Any) -> Any:
            # is_post=False: validate Host/Origin only. Content-Type is not
            # enforced here because several endpoints are intentionally bodyless
            # POSTs (/shutdown, /index/rebuild, /embedding/stop). A no-op when
            # security is None.
            started = time.perf_counter()
            path = request.url.path
            rejection = await security_mw.validate_request(request, is_post=False)
            if rejection is not None:
                logger.warning(
                    "%s %s rejected by Host/Origin guard (%d) from %s",
                    request.method,
                    path,
                    rejection.status_code,
                    request.client,
                )
                return rejection
            response = await handler(request)
            elapsed_ms = (time.perf_counter() - started) * 1000
            # Every served route logs at INFO — including /status polls. The one
            # exception is the actions edge: when an action reaches its impl, the
            # _safe_tool wrapper ALREADY emits the canonical one-INFO-line-per-call,
            # so the transport line here would be a SECOND INFO line for the same
            # call. Demote it to DEBUG for /actions/* (the handler logs its own INFO
            # line for the early-return errors that never reach a tool).
            level = logging.DEBUG if path.startswith("/actions/") else logging.INFO
            logger.log(
                level,
                "%s %s -> %d (%.0fms)",
                request.method,
                path,
                response.status_code,
                elapsed_ms,
            )
            return response

        return wrapped

    return guard


class _ControlListener:
    """Where the privileged control plane listens — always local, never widened
    by ``--allow-remote``.

    A Unix-domain socket on POSIX (``<state_dir>/control.sock``, filesystem-gated,
    off the network entirely); a loopback-only TCP port on Windows, where asyncio
    has no Unix-socket support. The address is recorded in ``server.json`` so the
    CLI/client can reach it; the data plane's exposure flags never touch this
    listener.
    """

    def __init__(self, state_dir: Path) -> None:
        self.uds: str | None = None
        self.host: str | None = None
        self.port: int | None = None
        self._sock: socket.socket | None = None
        if sys.platform == "win32":
            # No asyncio UDS on Windows: pre-bind an ephemeral loopback socket so
            # the chosen port is known before server.json is written, and hand the
            # bound socket to uvicorn (no re-bind race).
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            self.host, self.port = sock.getsockname()[:2]
            self._sock = sock
        else:
            # A short, user-private path (AF_UNIX is length-limited, and the state
            # dir can be deep); shared with daemon.cleanup_state so a restart finds
            # the same socket. A stale socket from a crashed daemon would make
            # bind() fail; the daemon lock guarantees no live peer owns it, so
            # remove it.
            uds = control_socket_path(state_dir)
            with contextlib.suppress(OSError):
                uds.unlink()
            self.uds = str(uds)

    @property
    def meta(self) -> dict[str, Any]:
        """The ``control`` block recorded in ``server.json`` for client discovery."""
        if self.uds is not None:
            return {"uds": self.uds}
        return {"url": f"http://{self.host}:{self.port}"}

    @property
    def operator_gated(self) -> bool:
        """Whether the transport restricts callers to the daemon's operator.

        A Unix-domain socket is ``0600`` inside a ``0700`` runtime dir, so only the
        daemon's own uid can connect — the caller IS the operator. The loopback-TCP
        fallback (Windows, which lacks asyncio UDS) is reachable by ANY local user
        on the host, so a caller there is *not* provably the operator. This gates
        whether ``/embedding/start`` trusts caller-supplied execution params
        (binary/model/args): a non-operator-gated transport must refuse them, since
        a hostile local user could otherwise drive the daemon to spawn an arbitrary
        binary as the daemon's user.
        """
        return self.uds is not None

    @property
    def security(self) -> TransportSecuritySettings | None:
        """The control listener's Host/Origin policy — fixed local, never widened.

        A Unix-domain socket is filesystem-gated and unreachable by a browser
        (there is no ``fetch`` to a Unix socket), so DNS-rebinding/Host validation
        adds nothing — skip it (``None``). The loopback-TCP fallback, by contrast,
        *is* reachable by a same-host browser at ``127.0.0.1:<port>``, so it keeps
        the loopback-only guard. Independent of the data plane's settings either
        way — the data plane's exposure flags never reach here.
        """
        if self.uds is not None:
            return None
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(_LOOPBACK_HOSTS),
            allowed_origins=list(_LOOPBACK_ORIGINS),
        )

    def uvicorn_config_kwargs(self) -> dict[str, Any]:
        """uvicorn ``Config`` kwargs binding this listener (``uds=`` on POSIX)."""
        if self.uds is not None:
            return {"uds": self.uds}
        # The pre-bound socket is passed to serve(sockets=...); Config still needs
        # a host/port for its own bookkeeping/logging.
        return {"host": self.host, "port": self.port}

    def serve_kwargs(self) -> dict[str, Any]:
        """uvicorn ``Server.serve`` kwargs (the pre-bound Windows socket)."""
        if self._sock is not None:
            return {"sockets": [self._sock]}
        return {}

    async def harden(self, server: Any) -> None:
        """Best-effort tighten the UDS to owner-only once uvicorn has bound it.

        uvicorn creates the socket file during startup; wait for that, then chmod
        0o600. A no-op on the TCP fallback. Defense-in-depth on top of the state
        dir's own (user-owned) permissions.

        Gives up if the server stops/never binds (``should_exit`` or a bounded
        deadline) rather than spinning forever — a control-startup that exits early
        leaves ``started`` False, and an unbounded wait here would wedge the gather
        (and the whole daemon).
        """
        if self.uds is None:
            return
        deadline = time.monotonic() + 10.0
        while not getattr(server, "started", False):
            if getattr(server, "should_exit", False) or time.monotonic() > deadline:
                return
            await asyncio.sleep(0.01)
        with contextlib.suppress(OSError):
            os.chmod(self.uds, 0o600)

    def cleanup(self) -> None:
        """Remove the UDS path on shutdown; close the pre-bound TCP socket."""
        if self.uds is not None:
            with contextlib.suppress(OSError):
                os.unlink(self.uds)
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()


def create_mcp(
    *,
    host: str,
    port: int,
    transport_security: TransportSecuritySettings | None,
) -> FastMCP:
    """Build a fresh FastMCP app.

    Constructed per-process inside ``main()`` rather than as an import-time
    global so the server is testable and re-usable in-process. ``host``/``port``
    and ``transport_security`` are passed at construction so the MCP endpoint's
    DNS-rebinding protection matches the address actually bound.

    ``transport_security is None`` means "no guard" everywhere else in the host
    (the custom routes' ``TransportSecurityMiddleware(None)`` leaves protection
    off). But FastMCP *silently re-enables* the guard on ``/mcp`` for a loopback
    host when handed ``None`` (mcp ``fastmcp/server.py`` auto-enables for
    127.0.0.1/localhost/::1) — so ``--no-dns-rebinding-protection`` would be
    honored on the custom routes yet ignored on ``/mcp``. Pass FastMCP an
    *explicit* protection-disabled settings instead of ``None`` so ``/mcp`` and
    the custom routes agree: when the guard is off, it is off on both.
    """
    if transport_security is None:
        transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return FastMCP(
        "Shrike",
        stateless_http=True,
        json_response=True,
        host=host,
        port=port,
        transport_security=transport_security,
    )


def _register_custom_routes(
    app: FastMCP,
    harness: Harness,
    server_lock: ServerLock,
    *,
    meta: dict[str, Any],
    security: TransportSecuritySettings | None,
    control_security: TransportSecuritySettings | None,
    request_shutdown: Callable[[], None],
    action_tools: dict[str, Tool] | None = None,
    manager: CollectionManager | None = None,
    export_store: Any | None = None,
    control_purely_local: bool = False,
) -> Starlette:
    """Register the custom HTTP routes, split across the data and control planes.

    The privileged *control* routes (shutdown/reload/index/embedding, the full
    ``/status`` diagnostics) are registered on a separate Starlette app this
    function builds and returns — served on the always-local control listener,
    never widened by ``--allow-remote``. The *data* routes (``/actions/{name}``,
    media, export, the minimal ``/health`` liveness) are registered on the FastMCP
    ``app`` alongside ``/mcp``, honoring the operator's exposure flags.

    Both planes bypass the MCP transport middleware, so each route gets Host/Origin
    validation via its plane's ``_make_guard`` — the data guard under ``security``,
    the control guard under ``control_security``. Each handler is parse → harness
    coroutine → JSONResponse; only host-specific work (uptime/pid assembly, the
    media FileResponse, process exit) stays here.

    ``action_tools`` (the ``name -> Tool`` map from :func:`register_tools`) backs
    the actions-over-HTTP data edge; when None the actions route isn't registered.
    ``control_purely_local`` gates whether ``/embedding/start`` trusts
    caller-supplied execution params: True for a filesystem-gated UDS control
    transport (callers confined to the operator), False for the Windows
    loopback-TCP fallback (reachable by other local users) — see
    :attr:`_ControlListener.operator_gated`.
    """
    wrapper = harness.wrapper
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import FileResponse, JSONResponse, Response
    from starlette.routing import Route

    from shrike.harness.collection import CollectionBusyError, _safe_media_name

    data_guard = _make_guard(security)
    control_guard = _make_guard(control_security)
    control_routes: list[Route] = []
    _Handler = Callable[[Request], Awaitable[Response]]

    def _route(plane: str, path: str, methods: list[str]) -> Callable[[_Handler], _Handler]:
        """Register a handler on its plane: the FastMCP data ``app`` (alongside
        ``/mcp``) or the control Starlette app, each behind its own guard."""

        def deco(handler: _Handler) -> _Handler:
            if plane == "control":
                control_routes.append(Route(path, control_guard(handler), methods=methods))
            else:
                app.custom_route(path, methods=methods)(data_guard(handler))
            return handler

        return deco

    @_route("data", "/health", ["GET"])
    async def handle_health(request: Request) -> JSONResponse:
        # The data plane's minimal, unauthenticated liveness probe: enough to
        # confirm the daemon is up and assert the wire version, and NOTHING the
        # full /status carries (paths, PID, model fingerprints, lock state) — that
        # diagnostic surface is control-plane only, so an --allow-remote'd data
        # plane leaks nothing sensitive.
        return JSONResponse({"running": True, "wire_protocol_version": WIRE_PROTOCOL_VERSION})

    @_route("control", "/status", ["GET"])
    async def handle_status(request: Request) -> JSONResponse:
        import contextlib

        status: dict[str, Any] = {
            "running": True,
            "wire_protocol_version": WIRE_PROTOCOL_VERSION,
            "pid": os.getpid(),
            "url": meta.get("url"),
            "collection": meta.get("collection"),
            "log_level": meta.get("log_level"),
            "log_dir": meta.get("log_dir"),
        }

        started = meta.get("started", "")
        if started:
            with contextlib.suppress(ValueError):
                start_dt = datetime.fromisoformat(started)
                delta = datetime.now(UTC) - start_dt
                hours, remainder = divmod(int(delta.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                if hours:
                    status["uptime"] = f"{hours}h {minutes}m"
                elif minutes:
                    status["uptime"] = f"{minutes}m {seconds}s"
                else:
                    status["uptime"] = f"{seconds}s"

        # The core status block (embedding/index/derived/locking); health()
        # probes llama-server over HTTP off the loop inside. These top-level
        # fields describe the DEFAULT collection (which the operational routes
        # act on).
        status.update(await harness.status())

        # Per-collection rows: the boot/default collection plus every registered
        # profile, with each one's held/index/col_mod. Only emitted when the
        # manager knows of more than the single boot collection, so a
        # single-collection daemon's payload carries no `collections` key.
        if manager is not None:
            rows = manager.status_rows()
            if len(rows) > 1:
                status["collections"] = rows

        return JSONResponse(status)

    @_route("data", "/media/{filename:path}", ["GET"])
    async def handle_media(request: Request) -> Response:
        # Serve a media file by name — the model-friendly retrieval path that
        # fetch_media/list_media point at (no base64). Read-only; same Host/Origin
        # guard as the other custom routes. Filename is reduced to a basename
        # inside the media dir (traversal guard); the dir is resolved lock-free.
        safe = _safe_media_name(request.path_params.get("filename", ""))
        if not safe:
            return Response(status_code=404)
        full = os.path.join(wrapper.media_dir, safe)
        if not os.path.isfile(full):
            return Response(status_code=404)
        return FileResponse(full, filename=safe)

    @_route("data", "/export/{token}", ["GET"])
    async def handle_export(request: Request) -> Response:
        # Serve a pending export package by its one-shot token — the download
        # path export_package's `url` points at (no base64). Read-only; same
        # Host/Origin guard. The token is the capability (secrets-random,
        # unguessable); the file is a server-named temp under the cache dir, so
        # there is no traversal surface. Reaped after a successful stream — and
        # on TTL / shutdown by the store regardless.
        if export_store is None:
            return Response(status_code=404)
        token = request.path_params.get("token", "")
        path = export_store.resolve(token)
        if path is None:
            return Response(status_code=404)
        from starlette.background import BackgroundTask

        filename = os.path.basename(path)
        # One-shot: reap AFTER the body is streamed (a background task runs once
        # the response completes — reaping inline would delete the file before
        # FileResponse reads it). A failed/aborted GET leaves it for the TTL
        # sweep / shutdown, so the collection-bearing temp never lingers.
        return FileResponse(
            path,
            filename=filename,
            background=BackgroundTask(export_store.reap, token),
        )

    @_route("control", "/index/rebuild", ["POST"])
    async def handle_index_rebuild(request: Request) -> JSONResponse:
        try:
            return JSONResponse(await harness.rebuild_index())
        except KernelConfigError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @_route("control", "/index/save", ["POST"])
    async def handle_index_save(request: Request) -> JSONResponse:
        return JSONResponse(await harness.save_index())

    @_route("control", "/embedding/start", ["POST"])
    async def handle_embedding_start(request: Request) -> JSONResponse:
        import contextlib

        overrides: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            body = await request.json()
            if isinstance(body, dict):
                for key in (
                    "backend",
                    "model",
                    "port",
                    "context_size",
                    "threads",
                    "gpu_layers",
                    "pooling",
                    "extra_args",
                    "llama_server",
                    "onnx_providers",
                    "batch_size",
                ):
                    if body.get(key) is not None:
                        overrides[key] = body[key]

        # Execution-shaping overrides (binary, model, subprocess args, onnx
        # providers, backend kind) are trusted only when the control transport
        # confines callers to the daemon's operator. A filesystem-gated UDS does
        # (``control_purely_local`` True → accept); the Windows loopback-TCP control
        # listener does NOT — any local user on the host can reach it, so a
        # caller-chosen binary there is the #791 RCE class. There the overrides are
        # refused and the daemon falls back to its boot-configured settings; runtime
        # knobs still pass through.
        rejected = _rejected_exec_overrides(overrides, purely_local=control_purely_local)
        if rejected:
            logger.warning(
                "Refusing /embedding/start overrides %s: the control channel is not confined "
                "to the operator (a loopback-TCP control listener is reachable by other local "
                "users), so the caller-supplied binary/model/args are ignored in favour of the "
                "daemon's boot-configured embedding settings",
                rejected,
            )
            return JSONResponse(
                {
                    "error": (
                        "Execution-shaping parameters ("
                        + ", ".join(_EXEC_SHAPING_OVERRIDES)
                        + ") can't be set via /embedding/start when the control channel isn't "
                        "confined to the daemon's operator (a loopback-TCP control listener is "
                        "reachable by other local users on this host). Configure them at daemon "
                        "startup (--embedding-*/--llama-server/--config) instead."
                    )
                },
                status_code=400,
            )

        try:
            # Starting a backend blocks (model load + health wait); the kernel
            # call runs off the event loop so other requests keep flowing.
            return JSONResponse(await harness.start_embedding(overrides))
        except KernelConfigError as e:
            # Unknown backend / no model / a missing ONNX optional dependency
            # are caller-actionable config errors → 400.
            return JSONResponse({"error": str(e)}, status_code=400)
        except (FileNotFoundError, RuntimeError) as e:
            logger.error("Failed to start embedding service: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    @_route("control", "/embedding/stop", ["POST"])
    async def handle_embedding_stop(request: Request) -> JSONResponse:
        return JSONResponse(await harness.stop_embedding())

    @_route("control", "/reload", ["POST"])
    async def handle_reload(request: Request) -> JSONResponse:
        return JSONResponse(await harness.reload())

    # --- actions-over-HTTP: the UI edge of the action catalog ----------
    # POST /actions/{name}: typed JSON request -> typed JSON response over the
    # SAME named ops the MCP tools bind, served through the SAME _safe_tool path
    # (so error policy + INFO logging + arg coercion are identical), behind the
    # SAME Host/Origin guard. MCP stays the agent edge; this is the UI edge. The
    # body it returns is the structured content the MCP path emits, minus the
    # JSON-RPC envelope. The loopback daemon stays unauthenticated — auth is the
    # relay/proxy edge, never here.
    def _wire_headers() -> dict[str, str]:
        return {WIRE_VERSION_HEADER: str(WIRE_PROTOCOL_VERSION)}

    def _action_error(code: ActionErrorCode, message: str, status: int) -> JSONResponse:
        body = ActionError(code=code, message=message).model_dump(mode="json", by_alias=True)
        return JSONResponse(body, status_code=status, headers=_wire_headers())

    if action_tools is not None:

        @_route("data", "/actions/{name}", ["POST"])
        async def handle_action(request: Request) -> JSONResponse:
            name = request.path_params.get("name", "")

            # Wire-version handshake: an optional request wire-version header must
            # match the server's. A mismatch is refused before the op runs — the
            # minimum a separately-shipped client needs to fail fast on a fabric
            # skew. Absent header = no assertion (today's CLI/programmatic use).
            requested = request.headers.get(WIRE_VERSION_HEADER)
            if requested is not None and requested.strip() != str(WIRE_PROTOCOL_VERSION):
                # The handler owns the one INFO line for the early-return errors
                # that never reach _safe_tool (the route line in _guard is DEBUG
                # for /actions/*).
                logger.info("action %s rejected: wire version %r", name, requested)
                return _action_error(
                    ActionErrorCode.INPUT_ERROR,
                    f"Unsupported wire protocol version {requested!r}; "
                    f"this server speaks {WIRE_PROTOCOL_VERSION}.",
                    400,
                )

            tool = action_tools.get(name)
            if tool is None:
                logger.info("action %s -> unknown_action (404)", name)
                return _action_error(
                    ActionErrorCode.UNKNOWN_ACTION,
                    f"No action named {name!r}.",
                    404,
                )

            # The request body is the tool's arguments object (a JSON object, or
            # absent for a no-arg call). A malformed/non-object body is a caller
            # mistake → input_error, not a 500.
            arguments: dict[str, Any] = {}
            if await request.body():
                try:
                    parsed = await request.json()
                except Exception:
                    logger.info("action %s rejected: malformed JSON body", name)
                    return _action_error(
                        ActionErrorCode.INPUT_ERROR,
                        "Request body must be a JSON object of the action's arguments.",
                        400,
                    )
                if not isinstance(parsed, dict):
                    logger.info("action %s rejected: body is not a JSON object", name)
                    return _action_error(
                        ActionErrorCode.INPUT_ERROR,
                        "Request body must be a JSON object of the action's arguments.",
                        400,
                    )
                arguments = parsed

            try:
                # The same path the MCP tool runs: func_metadata validates +
                # coerces the args against the action's typed signature, then
                # the _safe_tool-wrapped impl runs (its error policy + the one
                # INFO completion line included). convert_result=False returns
                # the raw response model, which we serialize exactly as MCP's
                # structuredContent does (output_model.model_dump by_alias).
                result = await tool.run(arguments, convert_result=False)
            except Exception as exc:
                # Tool.run wraps every failure in ToolError, preserving the
                # original as __cause__; argument-validation failures (a bad or
                # out-of-range arg) surface as a pydantic ValidationError raised
                # before the impl runs, so they aren't a _safe_tool-mapped type.
                cause = exc.__cause__ if exc.__cause__ is not None else exc
                if isinstance(cause, ToolInputError):
                    return _action_error(ActionErrorCode.INPUT_ERROR, str(cause), 400)
                if isinstance(cause, CollectionBusyError | shrike_native.NativeBusyError):
                    # The op never ran (contention) — the caller may retry.
                    return _action_error(
                        ActionErrorCode.COLLECTION_BUSY,
                        "The collection is in use by another process; retry shortly.",
                        409,
                    )
                if isinstance(cause, ValidationError):
                    return _action_error(ActionErrorCode.INPUT_ERROR, str(cause), 400)
                # A genuine bug: _safe_tool already logged it with a traceback
                # (or, for a validation-stage failure, log it here). The wire
                # body carries a FIXED, non-leaking message — never the detail.
                logger.exception("Unhandled error serving action %r", name)
                return _action_error(
                    ActionErrorCode.INTERNAL_ERROR,
                    "The server failed to process this action.",
                    500,
                )

            # Serialize exactly as MCP's structuredContent (FuncMetadata.
            # convert_result): validate the response model, dump json+by_alias.
            # wrap_output is False for every action (each returns a BaseModel),
            # but honor it for completeness so this stays a faithful mirror.
            meta_md = tool.fn_metadata
            payload = {"result": result} if meta_md.wrap_output else result
            assert meta_md.output_model is not None  # every action has a response model
            structured = meta_md.output_model.model_validate(payload).model_dump(
                mode="json", by_alias=True
            )
            return JSONResponse(structured, headers=_wire_headers())

    @_route("control", "/shutdown", ["POST"])
    async def handle_shutdown(request: Request) -> JSONResponse:
        # Graceful exit via uvicorn's own machinery: flag should_exit and return
        # a plain 200. Uvicorn completes in-flight responses — this one
        # included — and closes every connection with a proper FIN before
        # serve() returns; the harness teardown + lock release run on the
        # serve() tail. A process exit (close + BackgroundTask + sleep +
        # os._exit) would race the client's read no matter the grace period: it
        # can turn the un-acked response bytes into a connection reset under a
        # saturated runner, while a graceful close cannot.
        request_shutdown()
        return JSONResponse({"status": "ok", "pid": os.getpid()})

    # The control plane is its own Starlette ASGI app, served on the always-local
    # control listener. The data routes were registered on the FastMCP `app` above.
    return Starlette(routes=control_routes)


def main() -> None:
    # prog is pinned so the help/usage text reads `shrike.server` regardless of
    # entry point (python -m shrike.server via __main__, the //shrike-py/bin launcher, or
    # the foreground CLI) — the package would otherwise surface argv[0] as
    # `__main__.py`.
    parser = argparse.ArgumentParser(prog="shrike.server", description="Shrike MCP server for Anki")
    parser.add_argument(
        "--collection",
        required=True,
        help="Path to the Anki collection file (collection.anki2)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config file declaring the v2 capability sections "
        "(embedders:/recognizers:/managed:, #498). The daemon resolves them "
        "itself — structured entries (remote endpoints, api_key_env) have no "
        "flag spelling. Mutually exclusive with the legacy --embedding-*/"
        "--llama-server/--ocr-backend flags.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8372,
        help="Port to listen on (default: 8372)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Permit binding to a non-loopback host. Every endpoint is "
        "unauthenticated, so this exposes the full collection API to the "
        "network — only use it behind your own auth/network controls.",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="HOST",
        help="Additional Host header value to accept beyond loopback (repeatable). "
        "Use for a reverse-proxy or VPN hostname, e.g. a Caddy domain or a "
        "Tailscale name like host.tailnet.ts.net. A 'host:port' is matched exactly; "
        "a bare host accepts any port.",
    )
    parser.add_argument(
        "--allowed-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="Additional Origin header value to accept beyond loopback (repeatable). "
        "Most native MCP clients send no Origin (which is always allowed); add one "
        "only if a browser-based client is rejected with 403.",
    )
    parser.add_argument(
        "--no-dns-rebinding-protection",
        action="store_true",
        help="Disable Host/Origin validation entirely. For deployments where the "
        "network is the trust boundary — Shrike behind a reverse proxy (Caddy), on a "
        "private VPN/tailnet, or firewalled. Accepts requests with any Host/Origin, "
        "on any bind. Endpoints remain unauthenticated; rely on your network layer.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for log files (default: platform-specific)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Log level (default: info)",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Log to console in addition to file (set by CLI foreground mode)",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Directory for lock/pid/meta state files (default: platform-specific)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for cache files like the vector index (default: platform-specific)",
    )
    parser.add_argument(
        "--cooperative-lock",
        action="store_true",
        help="Release the collection lock when idle and re-open on demand, so an "
        "idle daemon doesn't block launching Anki (opt-in; default holds the lock "
        "for the daemon's lifetime).",
    )
    parser.add_argument(
        "--lock-hold-seconds",
        type=float,
        default=None,
        help="In cooperative mode, seconds to hold the collection after the last "
        f"operation before releasing it (default: {DEFAULT_LOCK_HOLD:g})",
    )
    parser.add_argument(
        "--allow-private-media-fetch",
        action="store_true",
        help="Let store_media fetch URLs that resolve to private/loopback addresses "
        "(off by default — the SSRF guard refuses them). Only for trusted internal "
        "hosts. Also enabled by SHRIKE_MEDIA_ALLOW_PRIVATE_FETCH=1.",
    )
    parser.add_argument(
        "--public-url",
        default=None,
        help="Externally-visible base URL (e.g. https://anki.example.com) used to "
        "build media-file links in fetch_media/list_media. Set this when behind a "
        "reverse proxy; defaults to the bind host. Also read from SHRIKE_PUBLIC_URL.",
    )
    parser.add_argument(
        "--media-path-root",
        action="append",
        default=None,
        metavar="DIR",
        help="Enable store_media's server-local `path` input, confined to files under "
        "DIR (after resolving symlinks). Repeatable for several locations; a path under "
        "any one is allowed. Off (path rejected) when unset. Also read from "
        "SHRIKE_MEDIA_PATH_ROOTS (os.pathsep-separated). Requires a purely-local daemon.",
    )
    parser.add_argument(
        "--export-path-root",
        action="append",
        default=None,
        metavar="DIR",
        help="Enable export_package's server-local `output_path`, confined to files written "
        "under DIR (after resolving symlinks). Repeatable; a path under any one is allowed. "
        "Off (output_path rejected; export still works via the download url) when unset. Also "
        "read from SHRIKE_EXPORT_PATH_ROOTS (os.pathsep-separated). Requires a purely-local "
        "daemon — a write capability distinct from --media-path-root's read.",
    )
    parser.add_argument(
        "--import-path-root",
        action="append",
        default=None,
        metavar="DIR",
        help="Enable import_package's server-local `path` input, confined to files under "
        "DIR (after resolving symlinks). DISTINCT from --media-path-root: import writes into "
        "the collection (a merge), a higher blast radius than a media-file read, so it gets "
        "its own root and never inherits a media root. Repeatable; off (path rejected) when "
        "unset. Also read from SHRIKE_IMPORT_PATH_ROOTS (os.pathsep-separated). Requires a "
        "purely-local daemon.",
    )
    parser.add_argument(
        "--index-save-delay",
        type=float,
        default=None,
        help="Seconds of idle after the last index change before flushing it to "
        f"disk (default: {DEFAULT_SAVE_DELAY:g})",
    )
    parser.add_argument(
        "--index-save-threshold",
        type=int,
        default=None,
        help="Unsaved index changes that force an immediate flush regardless of "
        f"idle time (default: {DEFAULT_SAVE_THRESHOLD})",
    )
    parser.add_argument(
        "--llama-server",
        default=None,
        help="Path to llama-server binary (default: LLAMA_SERVER_PATH env or PATH lookup)",
    )
    parser.add_argument(
        "--embedding-backend",
        default=None,
        choices=[*SUPPORTED_BACKENDS, *BACKEND_ALIASES],
        help="Embedding backend: 'llama' (llama-server subprocess, GGUF/MLX) or "
        "'onnx' (in-process onnxruntime; needs the 'onnx' optional dependency). "
        "Default: llama.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Path to the embedding model: a GGUF file for the llama backend, or an "
        "ONNX model directory (with model.onnx + tokenizer.json) for the onnx "
        "backend. Enables the embedding service.",
    )
    parser.add_argument(
        "--embedding-port",
        type=int,
        default=None,
        help="Port for the embedding server (default: 8373)",
    )
    parser.add_argument(
        "--embedding-context-size",
        type=int,
        default=None,
        help="Context size for embedding model",
    )
    parser.add_argument(
        "--embedding-threads",
        type=int,
        default=None,
        help="Number of CPU threads for embedding inference",
    )
    parser.add_argument(
        "--embedding-gpu-layers",
        type=int,
        default=None,
        help="Number of layers to offload to GPU",
    )
    parser.add_argument(
        "--embedding-pooling",
        default=None,
        choices=["mean", "last", "cls", "none"],
        help="llama-server pooling type. Set 'last' for last-token models "
        "(Jina v5, Qwen3-Embedding) whose GGUF omits it; otherwise the model's "
        "stored default is used.",
    )
    parser.add_argument(
        "--embedding-arg",
        action="append",
        default=None,
        metavar="TOKENS",
        help="Extra llama-server flag(s) to pass through verbatim, repeatable "
        "(e.g. --embedding-arg='--flash-attn'). Each value is shlex-split; "
        "Shrike-owned flags (--model/--host/--port/--embeddings) are rejected. "
        "For runtime-only flags — vector-affecting flags belong in typed "
        "settings like --embedding-pooling. (llama backend only.)",
    )
    parser.add_argument(
        "--embedding-onnx-provider",
        action="append",
        default=None,
        metavar="PROVIDER",
        help="onnxruntime execution provider(s), repeatable, in priority order "
        "(e.g. CUDAExecutionProvider). Default: CPUExecutionProvider. (onnx backend "
        "only.)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=_positive_int,
        default=None,
        help="Cap the embedding batch size (any backend), >= 1. Default: batch as large as a "
        "startup self-check proves safe. A batch-variant model (e.g. int8 ONNX) is always "
        "embedded serially regardless.",
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Don't start the embedding service at boot even if a model is configured "
        "(start it later with 'shrike embedding start')",
    )
    parser.add_argument(
        "--ocr-backend",
        default=None,
        choices=["apple"],
        help="OCR backend for recognizing text in note media (#228). Off by default. "
        "'apple' (macOS Vision) is no longer compiled into the server build — "
        "platform engines are mobile-only (docs/distribution.md); selecting it "
        "degrades recognition to an error state (the replacement is #502).",
    )
    args = parser.parse_args()

    log_dir = configure_logging(
        foreground=args.foreground,
        log_dir_override=args.log_dir,
        log_level_override=args.log_level,
    )

    # Rig native observability into this process's logging: the Rust crates emit
    # tracing events but never log directly — init_logging bridges them onto
    # stdlib `logging` (logger name = the Rust target) and installs the span-trace
    # subscriber behind the exception notes. Must run *after* configure_logging
    # (pyo3-log caches effective levels). A missing native install just means no
    # native logs to bridge.
    with contextlib.suppress(ImportError):
        import shrike_native

        shrike_native.init_logging()

    # The driven runtime's committed threads, installed + started inside _serve()
    # just before the kernel opens (so the set-once seam wins and the loops are
    # parked before the first op), joined on teardown.
    driven = DrivenRuntime()

    # Refuse non-loopback binds unless explicitly opted in: every endpoint is
    # unauthenticated, so a non-loopback host hands the full collection API to
    # anyone on the network. Fail fast, before touching the collection.
    if not _is_loopback(args.host):
        if not args.allow_remote:
            logger.error(
                "Refusing to bind to non-loopback host %s without --allow-remote. "
                "All endpoints are unauthenticated; binding to the network would "
                "expose full collection access (read, write, delete, shutdown) to "
                "anyone who can reach the port.",
                args.host,
            )
            sys.exit(1)
        logger.warning(
            "Binding to non-loopback host %s with --allow-remote: all endpoints are "
            "UNAUTHENTICATED. The entire collection API is reachable by anyone on the "
            "network. Put your own auth/network controls in front of it.",
            args.host,
        )

    # Acquire the daemon lock before touching the collection
    state_dir_override = Path(args.state_dir) if args.state_dir else None
    resolved_state_dir = state_dir_override or state_dir()
    # The privileged control plane's own listener — a Unix-domain socket (POSIX)
    # or a loopback-only TCP port (Windows), built BEFORE server.json is written so
    # its address can be recorded for the CLI/client to discover. Always local; the
    # data-plane exposure flags below never touch it.
    control_listener = _ControlListener(resolved_state_dir)
    server_lock = ServerLock(state_dir_override=state_dir_override)
    server_meta = {
        "pid": None,
        "url": f"http://{args.host}:{args.port}/mcp",
        "host": args.host,
        "port": args.port,
        "control": control_listener.meta,
        "collection": args.collection,
        "log_dir": str(log_dir),
        "log_level": args.log_level or "info",
        "started": datetime.now(UTC).isoformat(),
    }
    try:
        server_lock.acquire(meta=server_meta)
    except AlreadyRunningError as e:
        logger.error("Cannot start: %s", e)
        sys.exit(1)

    hold_seconds = (
        args.lock_hold_seconds if args.lock_hold_seconds is not None else DEFAULT_LOCK_HOLD
    )
    if args.cooperative_lock:
        logger.info(
            "Cooperative locking on: releasing the collection after %.0fs idle", hold_seconds
        )

    # Kernel mode: the kernel owns the collection, the vector index, and the
    # derived ingest; the index files live in the cache dir. The media resolver
    # pair is path-derived (lock-free) and feeds the kernel's image seam for a
    # CLIP-style backend.
    cache_base = Path(args.cache_dir) if args.cache_dir else cache_dir()
    collection_abs = os.path.abspath(args.collection)
    media_base = (
        collection_abs[: -len(".anki2")] if collection_abs.endswith(".anki2") else collection_abs
    )
    _read_img, _img_exists = _make_image_resolver(media_base + ".media")

    # llama-server stays on loopback regardless of the MCP bind host — there is
    # never a reason to expose the embedding backend to the network.
    emb_params: dict[str, Any] = {
        "backend": args.embedding_backend or DEFAULT_BACKEND,
        "model": args.embedding_model,
        "port": args.embedding_port or 8373,
        "context_size": args.embedding_context_size,
        "threads": args.embedding_threads,
        "gpu_layers": args.embedding_gpu_layers,
        "pooling": args.embedding_pooling,
        "extra_args": args.embedding_arg,
        "llama_server": args.llama_server,
        "onnx_providers": args.embedding_onnx_provider,
        "batch_size": args.embedding_batch_size,
        "endpoint": None,
        "api_key_env": None,
        # The capability config carries these; the legacy flag path leaves them
        # at the runtime defaults (text-only, no projectors).
        "modalities": None,
        "mmprojs": None,
    }
    # The v2 recognizers: a list of harness-ready plans. The legacy
    # (no --config) path uses the --ocr-backend flag instead and leaves this
    # empty; a v2 config fills it from the resolved profile.
    recognizer_boot_plans: tuple[Any, ...] = ()
    # The SECONDARY embedding spaces' runtime params: the 2nd+ entries of a
    # multi-space v2 config. Empty for the legacy/flag path and the N=1 case,
    # so the single-space boot is byte-identical (no secondary runtime built).
    secondary_param_sets: tuple[dict[str, Any], ...] = ()
    if args.config:
        # The daemon resolves the v2 capability sections itself: structured
        # entries (remote endpoints, api_key_env) have no flag spelling, so the
        # CLI hands over the config path instead of params. A ProfileError here
        # is a config error — refuse to boot, loudly (the CLI pre-validates, so
        # this is the direct-invocation backstop).
        legacy_flags = [
            name
            for name, value in (
                ("--embedding-backend", args.embedding_backend),
                ("--embedding-model", args.embedding_model),
                ("--embedding-port", args.embedding_port),
                ("--embedding-pooling", args.embedding_pooling),
                ("--embedding-context-size", args.embedding_context_size),
                ("--embedding-threads", args.embedding_threads),
                ("--embedding-gpu-layers", args.embedding_gpu_layers),
                ("--embedding-arg", args.embedding_arg),
                ("--embedding-onnx-provider", args.embedding_onnx_provider),
                ("--embedding-batch-size", args.embedding_batch_size),
                ("--llama-server", args.llama_server),
                ("--ocr-backend", args.ocr_backend),
            )
            if value
        ]
        if legacy_flags:
            parser.error(
                f"--config is mutually exclusive with {', '.join(legacy_flags)} — "
                "the config file is the only home for these (docs/distribution.md)"
            )
        from shrike.cli.config import load_config, resolve_embedding
        from shrike.harness.profiles import (
            ProfileError,
            parse_capabilities,
            plan_to_runtime_params,
            plan_to_runtime_params_set,
            recognizer_plans,
            resolve_profile,
        )

        config_dict = load_config(Path(args.config))
        try:
            caps = parse_capabilities(config_dict)
        except ProfileError as e:
            # A structural parse error (e.g. both v2 and legacy sections) is a
            # genuine config error on either launch path → refuse to boot.
            parser.error(str(e))
        if caps.legacy:
            # A legacy config (no v2 sections) DEGRADES with a warning on BOTH
            # launch paths: the CLI's resolve_embedding_profile short-circuits
            # caps.legacy to this same resolve_embedding, never running the
            # build-validating resolve_profile. Routing the daemon --config path
            # here too is what keeps the two paths consistent — the legacy
            # cascade is warn-and-map, so the intended behavior is degrade, not
            # refuse-boot. A real v2-config error still hits resolve_profile below
            # and refuses.
            for warning in caps.warnings:
                logger.warning("%s", warning)
            try:
                legacy_params = resolve_embedding(config_dict)
            except ValueError as e:
                # A malformed legacy value (e.g. batch_size < 1) is a real config
                # error on either path (the CLI's resolve_embedding raises the same
                # ValueError) → refuse to boot, not degrade.
                parser.error(str(e))
            emb_params.update({k: v for k, v in legacy_params.items() if v is not None})
            emb_params["backend"] = legacy_params.get("backend") or DEFAULT_BACKEND
            logger.info(
                "Legacy embedding config resolved from %s: backend=%s model=%s",
                args.config,
                emb_params["backend"],
                emb_params.get("model"),
            )
        else:
            try:
                plan = resolve_profile(caps, shrike_native.build_features())
            except ProfileError as e:
                parser.error(str(e))
            for warning in plan.warnings:
                logger.warning("%s", warning)
            # The resolved recognizers: describe (and remote OCR) ride the v2
            # config — no flag spelling, so they reach boot from here.
            recognizer_boot_plans = recognizer_plans(plan)
            all_param_sets = plan_to_runtime_params_set(plan)

            def _expand_paths(params: dict[str, Any]) -> dict[str, Any]:
                for key in ("model", "llama_server"):
                    if params.get(key):
                        params[key] = os.path.expanduser(str(params[key]))
                if params.get("mmprojs"):
                    params["mmprojs"] = [os.path.expanduser(str(p)) for p in params["mmprojs"]]
                return params

            # The PRIMARY space drives the index/search path (byte-identical N=1);
            # the 2nd+ entries become SECONDARY spaces, each its own runtime.
            v2_params = _expand_paths(plan_to_runtime_params(plan))
            secondary_param_sets = tuple(_expand_paths(dict(p)) for p in all_param_sets[1:])
            emb_params.update({k: v for k, v in v2_params.items() if v is not None})
            emb_params["backend"] = v2_params.get("backend") or DEFAULT_BACKEND
            logger.info(
                "Capability config resolved from %s: backend=%s model=%s endpoint=%s",
                args.config,
                emb_params["backend"],
                emb_params.get("model"),
                emb_params.get("endpoint"),
            )

    def _runtime_from_params(params: dict[str, Any], *, pid_file: Path | None) -> EmbeddingRuntime:
        return EmbeddingRuntime(
            backend=params["backend"],
            model=params["model"],
            host="127.0.0.1",
            port=params.get("port") or 8373,
            log_dir=log_dir,
            context_size=params.get("context_size"),
            threads=params.get("threads"),
            gpu_layers=params.get("gpu_layers"),
            pooling=params.get("pooling"),
            extra_args=params.get("extra_args"),
            llama_server=params.get("llama_server"),
            pid_file=pid_file,
            onnx_providers=params.get("onnx_providers"),
            batch_size=params.get("batch_size"),
            endpoint=params.get("endpoint"),
            api_key_env=params.get("api_key_env"),
            # A router-managed remote is flagged by the `router` sub-map
            # profiles.py attaches to each shared-router consumer; the flag makes
            # the RemoteBackend derive its fingerprint/dim from the pinned model,
            # not the shared endpoint's /v1/models[0].
            router_managed=params.get("router") is not None,
            **(
                {"modalities": params["modalities"]} if params.get("modalities") is not None else {}
            ),
            **({"mmprojs": params["mmprojs"]} if params.get("mmprojs") is not None else {}),
        )

    runtime = _runtime_from_params(emb_params, pid_file=resolved_state_dir / "embedding.pid")
    # The SECONDARY embedding spaces: each 2nd+ v2 entry is its own runtime,
    # attached to its own kernel embed space. Each needs a default backend kind
    # (a None backend is the no-embedder shape, which secondaries never are).
    # Only a MANAGED llama-server secondary writes a pid file, and it would
    # collide with the primary's, so secondaries get none (the in-process
    # onnx/clip and remote backends — the multi-space shapes — have no subprocess
    # to reap). Empty for the N=1 / legacy path → byte-identical.
    secondary_runtimes = [
        _runtime_from_params({**p, "backend": p.get("backend") or DEFAULT_BACKEND}, pid_file=None)
        for p in secondary_param_sets
    ]

    # The shared llama.cpp ROUTER: when N remote/no-endpoint spaces share one
    # managed server (managed.llama_server.models_dir), profiles.py attaches an
    # identical `router` sub-map to each consumer's params. Build ONE
    # `LlamaServerManager.router(...)` from it — the harness spawns it once,
    # every router-managed RemoteBackend talks to it over loopback, and only the
    # owner stops it. None in every other shape (N=1, single-managed, endpoint,
    # onnx) → the harness path is byte-identical. The consumers carry identical
    # `router` maps (same dir/port/knobs), so the first one found is canonical.
    shared_llama_manager = None
    router_cfg = next(
        (p["router"] for p in (emb_params, *secondary_param_sets) if p.get("router") is not None),
        None,
    )
    if router_cfg is not None:
        shared_llama_manager = shrike_native.LlamaServerManager.router(
            os.path.expanduser(str(router_cfg["models_dir"])),
            host="127.0.0.1",
            port=router_cfg["port"],
            models_max=router_cfg.get("models_max"),
            binary=(
                os.path.expanduser(str(router_cfg["binary"])) if router_cfg.get("binary") else None
            ),
            log_dir=str(log_dir) if log_dir else None,
            context_size=router_cfg.get("context_size"),
            threads=router_cfg.get("threads"),
            gpu_layers=router_cfg.get("gpu_layers"),
            pooling=router_cfg.get("pooling"),
            extra_args=list(router_cfg.get("extra_args") or []),
            # Reuses the SAME embedding.pid as the single-managed primary above —
            # safe because the two shapes are mutually exclusive per profile: a
            # router primary is a `remote` backend (writes no pid file), a single-
            # managed primary is `llama` (it does), so only one ever owns it.
            pid_file=str(resolved_state_dir / "embedding.pid"),
        )

    # The derived-text store (FTS5 trigram sidecar) is built by Harness.assemble
    # AFTER the kernel opens the collection — NOT here, before open. The store is
    # namespaced per collection: it opens the SAME
    # `<cache_dir>/derived/<namespace>/shrike.db` the kernel's DerivedEngine
    # writes (they share one file — the kernel ingests, this host surface reads).
    # The namespace canonicalizes the collection path, and that canonicalization
    # differs by whether the file EXISTS (realpath folds a symlinked prefix like
    # macOS /var/folders → /private/var/...; an absent file's lexical abspath does
    # not). Building before the kernel created the file would hash a fresh
    # collection under the abspath namespace while the kernel used the realpath
    # one, so the host /status would read an empty store. assemble builds it
    # post-open so the file exists for both sides; the native engine factory is
    # the default.

    transport_security = _build_transport_security(
        args.host,
        allowed_hosts=args.allowed_host,
        allowed_origins=args.allowed_origin,
        disable=args.no_dns_rebinding_protection,
    )
    if transport_security is None:
        logger.warning(
            "DNS-rebinding/Origin protection is OFF (%s); any Host/Origin is accepted. "
            "Endpoints are unauthenticated — rely on your network boundary "
            "(reverse proxy, VPN, firewall).",
            "--no-dns-rebinding-protection"
            if args.no_dns_rebinding_protection
            else "non-loopback bind without --allowed-host/--allowed-origin",
        )
    elif args.allowed_host or args.allowed_origin:
        logger.info(
            "Transport guard on; additionally trusting hosts=%s origins=%s",
            args.allowed_host or [],
            args.allowed_origin or [],
        )
    mcp = create_mcp(host=args.host, port=args.port, transport_security=transport_security)
    # Compile each tool schema's JSON Schema validator once instead of per call
    # (the SDK's jsonschema.validate rebuilds it every request — ~5.8ms/call).
    install_validator_cache()
    allow_private_media_fetch = args.allow_private_media_fetch or (
        os.environ.get("SHRIKE_MEDIA_ALLOW_PRIVATE_FETCH", "").strip().lower()
        in ("1", "true", "yes", "on")
    )
    if allow_private_media_fetch:
        logger.warning(
            "store_media URL fetch may reach private/loopback addresses (SSRF guard off)"
        )
    # Base URL for media-file links in fetch_media/list_media results. Behind a
    # reverse proxy the bind host isn't reachable, so `--public-url` (or env)
    # overrides it with the externally-visible origin. Otherwise derive from the
    # bind host — a wildcard bind (0.0.0.0/::) isn't connectable, so advertise
    # loopback there.
    public_url = args.public_url or os.environ.get("SHRIKE_PUBLIC_URL") or ""
    if public_url:
        media_base_url = public_url.rstrip("/")
    else:
        url_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        media_base_url = f"http://{url_host}:{args.port}"
    # Whether the server is in its default purely-local config — the shared
    # outer gate on every server-local filesystem capability (store_media's
    # `path` read, export's `output_path` write). Computed once; both gates
    # below compose it with their own root list.
    purely_local = _server_is_purely_local(
        args.host,
        allow_remote=args.allow_remote,
        no_dns_rebinding_protection=args.no_dns_rebinding_protection,
        allowed_hosts=args.allowed_host,
        allowed_origins=args.allowed_origin,
    )

    def _resolve_path_roots(
        flag_value: list[str] | None, env_name: str, flag_label: str, capability: str
    ) -> list[str]:
        """Validate + canonicalize an operator-allowed root list: per element
        (dedup, order-preserving) — the containment disjunction means the weakest
        root governs, so one bad root (filesystem root, missing dir) fails
        startup, not passes silently. Honored only on a purely-local server; a
        root set on a non-purely-local one is refused (warn), never half-enabled.
        Returns the active roots (empty → the capability stays off)."""
        raw = list(flag_value or [])
        env = os.environ.get(env_name)
        if env:
            raw += [p for p in env.split(os.pathsep) if p]
        if not raw:
            return []
        validated: list[str] = []
        for r in raw:
            try:
                resolved = validate_path_root(r)
            except ValueError as e:
                logger.error("Invalid %s %r: %s", flag_label, r, e)
                sys.exit(1)
            if resolved not in validated:
                validated.append(resolved)
        if not purely_local:
            logger.warning(
                "%s is set but the server is not purely-local (remote/proxied "
                "exposure); %s stays disabled",
                flag_label,
                capability,
            )
            return []
        logger.info("%s enabled, confined to %s", capability, validated)
        return validated

    # store_media's server-local `path` read and export's `output_path` write:
    # distinct capabilities, distinct root lists, the same gate.
    server_path_roots = _resolve_path_roots(
        args.media_path_root,
        "SHRIKE_MEDIA_PATH_ROOTS",
        "--media-path-root",
        "store_media server-local paths",
    )
    export_path_roots = _resolve_path_roots(
        args.export_path_root,
        "SHRIKE_EXPORT_PATH_ROOTS",
        "--export-path-root",
        "export server-local output paths",
    )

    # import_package's server-local `path` read: a DISTINCT capability from
    # store_media's read and export's write — import writes into the collection
    # (a merge), a higher blast radius than a media-file read — so its own root
    # list, never inheriting a media or export root. Same shared gate
    # (_resolve_path_roots): purely-local + per-root containment.
    server_import_path_roots = _resolve_path_roots(
        args.import_path_root,
        "SHRIKE_IMPORT_PATH_ROOTS",
        "--import-path-root",
        "import_package server-local paths",
    )

    # The collection/profile registry: a read-only snapshot for the
    # `list_profiles` enumeration action. Loaded from the server's config file
    # (the explicit --config, or the platform default), best-effort — a missing
    # or unreadable config yields an empty registry, never a boot failure. The
    # registry is host-side config; the server operates on --collection today,
    # and routing by selector is the capstone.
    from shrike.cli.config import DEFAULT_CONFIG_PATH, load_config
    from shrike.harness.registry import Registry

    profile_registry: Registry | None = None
    try:
        registry_config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
        profile_registry = Registry.from_config(load_config(registry_config_path))
    except Exception:  # noqa: BLE001 — enumeration is a convenience, never gates boot
        logger.debug("profile registry unavailable; list_profiles will report empty")

    # The export download store: server-named temp packages under the cache dir +
    # their one-shot download tokens, reaped on download / TTL / shutdown. Backs
    # the default export delivery (the GET /export/{token} route below); the
    # server-local output_path mode bypasses it.
    from shrike.server.export_store import ExportStore

    export_store = ExportStore(str(cache_base))

    # The cross-space image floor margin: config/env-resolved (no flag — an
    # operational knob, not a v2 capability section), so the server loads it from
    # the config file the daemon was started with. Default 1.0.
    from shrike.cli.config import load_config, resolve_cross_space_margin

    try:
        _margin_config = load_config(Path(args.config)) if args.config else load_config()
    except Exception:  # noqa: BLE001 — a missing/garbage config falls back to the default margin
        _margin_config = {}
    cross_space_floor_margin = resolve_cross_space_margin(_margin_config)

    async def _serve() -> None:
        # Install the driven runtime and park its committed N+2 threads BEFORE
        # the first kernel op (the open below): the kernel has no lazy fallback,
        # so it must be installed and the loops parked or the open would panic /
        # never be driven. shrike-core spawns no thread of its own — the harness
        # donates one io, one sync, and N compute threads, each GIL-released for
        # the server's life; the asyncio bridge submits ops, invisibly driven by
        # the io thread. driven.shutdown() closes the pools and joins them on
        # teardown.
        driven.install()
        driven.start()

        # Assembly runs ON the loop: the kernel opens with a dedicated harness
        # thread driving its executor; the wrapper rides the shared core;
        # tools/routes register before the socket binds (no request is accepted
        # until serve()).
        logger.info("Opening collection at %s", args.collection)
        harness = await Harness.assemble(
            collection_path=args.collection,
            cache_dir=str(cache_base),
            runtime=runtime,
            # derived omitted: assemble builds it post-open at the kernel's path.
            cooperative=args.cooperative_lock,
            hold_seconds=hold_seconds,
            media_read=_read_img,
            media_exists=_img_exists,
            index_save_delay=args.index_save_delay,
            index_save_threshold=args.index_save_threshold,
            secondary_runtimes=secondary_runtimes,
            cross_space_floor_margin=cross_space_floor_margin,
            shared_llama_manager=shared_llama_manager,
        )
        # Embedding starts at boot when anything configures it: a model (flag
        # or config entry) OR a bare endpoint (a remote/attach entry's endpoint
        # default model is a valid configuration with no model name).
        embedding_configured = bool(emb_params.get("model") or emb_params.get("endpoint"))
        await harness.boot(start_embedding=embedding_configured and not args.no_embedding)

        # Recognition: attach recognizers and sweep in the background. Off unless
        # configured; a dead endpoint / missing engine degrades to an 'error'
        # state row without disturbing the rest of the server. The legacy
        # --ocr-backend flag still drives OCR directly; the v2 recognizers:
        # config (describe today, remote OCR) drives the rest from the resolved
        # profile.
        if args.ocr_backend:
            harness.start_recognition(args.ocr_backend)
        for rec in recognizer_boot_plans:
            if rec.kind == "describe-remote":
                assert rec.endpoint is not None  # resolve_profile guarantees it
                harness.start_recognition_describe(
                    rec.endpoint,
                    model=rec.model,
                    api_key_env=rec.api_key_env,
                )
            elif rec.kind == "apple":
                harness.start_recognition("apple")
            else:  # pragma: no cover — resolve_profile rejects unknown kinds
                logger.warning("Unsupported recognizer kind %r; skipping", rec.kind)

        # Multi-collection routing: the manager wraps the boot harness as the
        # default collection and lazily assembles a per-collection harness for
        # any other registered profile a call selects (its own namespaced index +
        # per-collection derived store, sharing this base cache dir and this
        # embedding runtime). The default harness owns the shared runtime; routed
        # harnesses attach its backend and never stop it.
        manager = CollectionManager(
            params=HarnessParams(
                cache_dir=str(cache_base),
                runtime=runtime,
                media_read=_read_img,
                media_exists=_img_exists,
                cooperative=args.cooperative_lock,
                hold_seconds=hold_seconds,
                index_save_delay=args.index_save_delay,
                index_save_threshold=args.index_save_threshold,
                cross_space_floor_margin=cross_space_floor_margin,
            ),
            default_harness=harness,
            default_collection_path=args.collection,
            config_path=args.config or DEFAULT_CONFIG_PATH,
        )

        # These handlers cover the boot window AND the post-drain replay:
        # uvicorn's serve() installs its own SIGTERM/SIGINT handlers (drain
        # gracefully), then its capture_signals contextmanager REPLAYS the
        # received signal to these originals on exit — so a runtime SIGTERM
        # drains uvicorn first and lands here for the flush/close/exit.
        def _signal_shutdown(signum: int, frame: Any) -> None:  # noqa: ARG001
            sig_name = signal.Signals(signum).name
            logger.info("Received %s, shutting down", sig_name)
            # Sync-safe teardown: flush each assembled collection's index + close
            # the sidecars; the collections are crash-safe (WAL) and the process
            # exits now. Routed collections are flushed too; the shared runtime is
            # stopped once.
            for h in manager.active_harnesses():
                with contextlib.suppress(Exception):
                    h.kernel.save_index()
                with contextlib.suppress(Exception):
                    h.derived.close()
            with contextlib.suppress(Exception):
                harness.runtime.stop()
            with contextlib.suppress(Exception):
                export_store.close()  # reap any pending download temps
            control_listener.cleanup()  # unlink the control UDS (signal path skips the gather tail)
            server_lock.release()
            # Close the driven pools and join the committed threads last: the
            # index saves above are the durability point (runtime-independent),
            # so the kernel is quiesced enough that the queues close and the
            # joins are prompt. Joining before exit keeps the threads from being
            # torn down mid-work (the finalization-abort class).
            with contextlib.suppress(Exception):
                driven.shutdown()
            logger.info("Shutdown complete")
            sys.exit(0)

        signal.signal(signal.SIGTERM, _signal_shutdown)
        signal.signal(signal.SIGINT, _signal_shutdown)

        # register_tools binds the action registry to MCP and returns the same
        # actions as a name->Tool map for the actions-over-HTTP edge.
        action_tools = register_tools(
            mcp,
            harness.wrapper,
            index=harness.index_view,
            derived=harness.derived,
            kernel=harness.kernel,
            dedup_stats=harness.dedup_stats,
            allow_private_fetch=allow_private_media_fetch,
            server_path_roots=server_path_roots,
            server_import_path_roots=server_import_path_roots,
            media_base_url=media_base_url,
            export_path_roots=export_path_roots,
            export_store=export_store,
            server_purely_local=purely_local,
            registry=profile_registry,
            # Per-call collection routing: the manager resolves a selector to the
            # right collection's bundle, lazily assembling it.
            resolver=manager.resolve_bundle,
            # The data-plane gate (Theme C / #833): every action awaits the boot
            # collection's readiness barrier, so the data plane serves only once
            # boot/reload/re-acquire maintenance has settled. The operational
            # routes (/status, /reload, /shutdown, /embedding/*) are the control
            # plane and bypass this — they are not actions.
            readiness=harness.await_ready,
        )
        # Both uvicorn Servers are created after route registration, so the
        # /shutdown route reaches them through this late-bound holder. /shutdown
        # (a control route) drains BOTH listeners — the data plane and the control
        # plane exit together.
        server_holder: list[Any] = []

        def _request_shutdown() -> None:
            for server in server_holder:
                server.should_exit = True

        # The data routes land on the FastMCP `mcp` app (alongside /mcp); the
        # control routes come back as their own Starlette app, served on the
        # always-local control listener. The /embedding/start exec-override gate
        # keys on the control transport's trust: a filesystem-gated UDS confines
        # callers to the operator (overrides allowed), but the Windows loopback-TCP
        # fallback is reachable by any local user, so overrides are refused there.
        control_app = _register_custom_routes(
            mcp,
            harness,
            server_lock,
            meta=server_meta,
            security=transport_security,
            control_security=control_listener.security,
            request_shutdown=_request_shutdown,
            action_tools=action_tools,
            manager=manager,
            export_store=export_store,
            control_purely_local=control_listener.operator_gated,
        )

        logger.info(
            "Listening on %s:%s (data) + %s (control); log_dir=%s, log_level=%s",
            args.host,
            args.port,
            control_listener.uds or f"127.0.0.1:{control_listener.port}",
            log_dir,
            args.log_level or "info",
        )
        import uvicorn

        # Bound the graceful drain so a hung in-flight request can't wedge a
        # /shutdown forever (the daemon's stop path escalates to SIGTERM/SIGKILL).
        data_config = uvicorn.Config(
            mcp.streamable_http_app(),
            host=args.host,
            port=args.port,
            log_config=None,
            timeout_graceful_shutdown=5,
        )
        control_config = uvicorn.Config(
            control_app,
            log_config=None,
            timeout_graceful_shutdown=5,
            **control_listener.uvicorn_config_kwargs(),
        )
        data_server = uvicorn.Server(data_config)
        control_server = uvicorn.Server(control_config)
        server_holder.extend((data_server, control_server))
        # Run both listeners on the one loop. Harden the control socket (UDS →
        # owner-only) once it is bound. A /shutdown or signal sets should_exit on
        # both, so both serve()s return and the gather completes.
        try:
            await asyncio.gather(
                data_server.serve(),
                control_server.serve(**control_listener.serve_kwargs()),
                control_listener.harden(control_server),
            )
        except BaseException:
            # A listener failed to bind/serve at startup (or a signal-path exit
            # replayed through here): release the OS-visible state so the next
            # start isn't blocked by a held lock or a stale socket, then propagate.
            # Every action is idempotent, so re-running after _signal_shutdown's
            # own teardown is harmless (the graceful drain below is NOT repeated —
            # it owns the kernel close, which must run exactly once).
            control_listener.cleanup()
            with contextlib.suppress(Exception):
                server_lock.release()
            with contextlib.suppress(Exception):
                driven.shutdown()
            raise
        control_listener.cleanup()

        # serve() returned: either /shutdown set should_exit (the graceful path;
        # teardown belongs here, after the listener drained) or uvicorn exited
        # without a replayed signal. A SIGTERM replays into _signal_shutdown
        # above instead.
        logger.info("Server drained; shutting down")
        with contextlib.suppress(Exception):
            # Close every routed collection too, default last (it stops the
            # shared embedding runtime after the routed ones detach). This awaits
            # each kernel.close() (a kernel op still driven by the io thread), so
            # it MUST precede the driven shutdown below.
            await manager.close()
        with contextlib.suppress(Exception):
            export_store.close()  # reap any pending download temps
        server_lock.release()
        # The kernel is quiesced (every actor drained by manager.close above), so
        # closing the pool queues lets the committed threads return; join them
        # before this function unwinds and the interpreter finalizes.
        with contextlib.suppress(Exception):
            driven.shutdown()
        logger.info("Shutdown complete")

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
