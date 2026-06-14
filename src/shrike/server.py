from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import ipaddress
import logging
import os
import signal
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import shrike_native
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools.base import Tool
from mcp.server.transport_security import (
    TransportSecurityMiddleware,
    TransportSecuritySettings,
)
from pydantic import ValidationError

from shrike._mcp_perf import install_validator_cache
from shrike.cache_layout import derived_db_path
from shrike.collection import DEFAULT_LOCK_HOLD
from shrike.daemon import AlreadyRunningError, ServerLock
from shrike.derived import DerivedTextStore, NativeDerivedEngine
from shrike.embedding import (
    BACKEND_ALIASES,
    DEFAULT_BACKEND,
    SUPPORTED_BACKENDS,
    EmbeddingRuntime,
)
from shrike.harness import CollectionManager, Harness, HarnessParams, KernelConfigError

# The transport-free core (#275). The collectors and _maybe_rebuild moved there
# with it; re-exported here so existing import sites (tests included) are
# unchanged.
from shrike.log import configure_logging
from shrike.paths import cache_dir, state_dir
from shrike.pathsafety import (
    is_loopback,
    server_is_purely_local,
    validate_path_root,
)
from shrike.schemas import WIRE_PROTOCOL_VERSION, ActionError, ActionErrorCode
from shrike.tools import ToolInputError, register_tools

# The kernel saver's built-in flush tuning (#355): the --index-save-* help
# names the defaults the flags would override. Sourced from the kernel, not
# the retired Python facade.
DEFAULT_SAVE_DELAY = float(shrike_native.INDEX_SAVE_DELAY_DEFAULT)
DEFAULT_SAVE_THRESHOLD = int(shrike_native.INDEX_SAVE_THRESHOLD_DEFAULT)

# The actions-over-HTTP wire-version header (#505/#392). Every /actions/*
# response echoes the server's WIRE_PROTOCOL_VERSION; a request may carry the
# same header to assert it speaks the same fabric (a mismatch is refused, the
# minimum handshake a separately-shipped client needs). /status already reports
# the version in its body — this is the per-call header form.
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
    from shrike.collection import _safe_media_name

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
    and the guard rejects it (HTTP 421) — the server is reachable but answers nothing
    (#595). Folding the actual bind host into the allow-list closes that self-brick.

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
      were given — preserves the historical ``--allow-remote`` behaviour, where a
      deliberately network-bound server has no fixed Host set to validate against.

    Otherwise returns settings allow-listing the loopback Host/Origin values (when
    bound to loopback) plus any explicit additions — so a loopback server behind a
    local proxy can trust the proxy's forwarded hostname without opening up. The
    *actual* bind host is always folded in too: ``is_loopback`` accepts any 127/8
    address, so a bind Shrike happily accepts (e.g. ``--host 127.0.0.2``) that isn't
    one of the fixed loopback literals must still answer its own ``Host`` header
    rather than self-brick with HTTP 421 (#595).
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
# (store_media #164/#170, export #71, import #72) — the generic helpers live in
# shrike.pathsafety. These module-level aliases keep the historical server.py
# names that existing call sites and tests import; the per-capability *policy*
# (which root list, which gate) stays at the call sites.
_is_loopback = is_loopback
_server_is_purely_local = server_is_purely_local
_validate_media_path_root = validate_path_root


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
    off). But FastMCP would *silently re-enable* the guard on ``/mcp`` for a
    loopback host when handed ``None`` (mcp ``fastmcp/server.py`` auto-enables for
    127.0.0.1/localhost/::1) — so ``--no-dns-rebinding-protection`` would be
    honored on the custom routes yet ignored on ``/mcp`` (#605). Pass FastMCP an
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
    request_shutdown: Callable[[], None],
    action_tools: dict[str, Tool] | None = None,
    manager: CollectionManager | None = None,
    export_store: Any | None = None,
) -> None:
    """Register custom HTTP endpoints on the server.

    The custom routes bypass the MCP transport middleware, so they get the same
    Host/Origin validation applied here via ``_guard`` — otherwise a browser page
    could drive ``/shutdown`` etc. through a no-preflight POST.

    Each handler is parse → harness coroutine → JSONResponse (#332 S3d-2): the
    operational verbs live on the kernel-mode Harness and await natively; only
    what is genuinely host-specific stays here (the guard, uptime/pid/url
    assembly, the media FileResponse, process exit).

    ``action_tools`` (the ``name -> Tool`` map from :func:`register_tools`) backs
    the actions-over-HTTP edge (#505): a single ``POST /actions/{name}`` route,
    behind the same ``_guard``, runs each named action through the *same*
    ``_safe_tool``-wrapped impl the MCP tools bind — the UI edge of the one
    catalog. When None (a future host that wants only the operational routes) the
    actions route isn't registered.
    """
    wrapper = harness.wrapper
    from starlette.requests import Request
    from starlette.responses import FileResponse, JSONResponse, Response

    from shrike.collection import CollectionBusyError, _safe_media_name

    security_mw = TransportSecurityMiddleware(security)

    def _guard(
        handler: Callable[[Request], Awaitable[Response]],
    ) -> Callable[[Request], Awaitable[Response]]:
        @functools.wraps(handler)
        async def wrapped(request: Request) -> Response:
            # is_post=False: validate Host/Origin only. Content-Type is not
            # enforced here because several endpoints are intentionally bodyless
            # POSTs (/shutdown, /index/rebuild, /embedding/stop). A no-op when
            # security is None (non-loopback bind).
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
            # Every served route logs at INFO — including /status polls: knowing
            # what the server did (and how long it took) is the point. The one
            # exception is the actions edge (#505): when an action reaches its
            # impl, the _safe_tool wrapper ALREADY emits the canonical
            # one-INFO-line-per-call (tool name + params + outcome + duration),
            # so the transport line here would be a SECOND INFO line for the
            # same call. Demote it to DEBUG for /actions/* (the handler logs its
            # own INFO line for the early-return errors that never reach a tool,
            # so the "one INFO line per served call" rule holds either way).
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

    @app.custom_route("/status", methods=["GET"])
    @_guard
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

        # Per-collection rows (#68): the boot/default collection plus every
        # registered profile, with each one's held/index/col_mod. Only emitted
        # when the manager knows of more than the single boot collection, so a
        # single-collection daemon's payload is unchanged.
        if manager is not None:
            rows = manager.status_rows()
            if len(rows) > 1:
                status["collections"] = rows

        return JSONResponse(status)

    @app.custom_route("/media/{filename:path}", methods=["GET"])
    @_guard
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

    @app.custom_route("/export/{token}", methods=["GET"])
    @_guard
    async def handle_export(request: Request) -> Response:
        # Serve a pending export package by its one-shot token (#71) — the
        # download path export_package's `url` points at (no base64). Read-only;
        # same Host/Origin guard. The token is the capability (secrets-random,
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

    @app.custom_route("/index/rebuild", methods=["POST"])
    @_guard
    async def handle_index_rebuild(request: Request) -> JSONResponse:
        try:
            return JSONResponse(await harness.rebuild_index())
        except KernelConfigError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.custom_route("/index/save", methods=["POST"])
    @_guard
    async def handle_index_save(request: Request) -> JSONResponse:
        return JSONResponse(await harness.save_index())

    @app.custom_route("/embedding/start", methods=["POST"])
    @_guard
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

    @app.custom_route("/embedding/stop", methods=["POST"])
    @_guard
    async def handle_embedding_stop(request: Request) -> JSONResponse:
        return JSONResponse(await harness.stop_embedding())

    @app.custom_route("/reload", methods=["POST"])
    @_guard
    async def handle_reload(request: Request) -> JSONResponse:
        return JSONResponse(await harness.reload())

    # --- actions-over-HTTP: the UI edge of the action catalog (#505) ----------
    # POST /actions/{name}: typed JSON request -> typed JSON response over the
    # SAME named ops the MCP tools bind, served through the SAME _safe_tool path
    # (so error policy + INFO logging + arg coercion are identical), behind the
    # SAME Host/Origin guard. MCP stays the agent edge; this is the UI edge. The
    # body it returns is the structured content the MCP path emits, minus the
    # JSON-RPC envelope. The loopback daemon stays unauthenticated — auth is the
    # relay/proxy edge (#52), never here.
    def _wire_headers() -> dict[str, str]:
        return {WIRE_VERSION_HEADER: str(WIRE_PROTOCOL_VERSION)}

    def _action_error(code: ActionErrorCode, message: str, status: int) -> JSONResponse:
        body = ActionError(code=code, message=message).model_dump(mode="json", by_alias=True)
        return JSONResponse(body, status_code=status, headers=_wire_headers())

    if action_tools is not None:

        @app.custom_route("/actions/{name}", methods=["POST"])
        @_guard
        async def handle_action(request: Request) -> JSONResponse:
            name = request.path_params.get("name", "")

            # #392 handshake: an optional request wire-version header must match
            # the server's. A mismatch is refused before the op runs — the
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

    @app.custom_route("/shutdown", methods=["POST"])
    @_guard
    async def handle_shutdown(request: Request) -> JSONResponse:
        # Graceful exit via uvicorn's own machinery (#344): flag should_exit
        # and return a plain 200. Uvicorn completes in-flight responses —
        # this one included — and closes every connection with a proper FIN
        # before serve() returns; the harness teardown + lock release run on
        # the serve() tail. The previous shape (close + BackgroundTask +
        # sleep + os._exit) raced the client's read no matter the grace
        # period: a process exit can turn the un-acked response bytes into a
        # connection reset under a saturated runner, while a graceful close
        # cannot.
        request_shutdown()
        return JSONResponse({"status": "ok", "pid": os.getpid()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Shrike MCP server for Anki")
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

    # Rig native observability into this process's logging (#308/#310): the
    # Rust crates emit tracing events but never log directly — init_logging
    # bridges them onto stdlib `logging` (logger name = the Rust target) and
    # installs the span-trace subscriber behind the exception notes. Must run
    # *after* configure_logging (pyo3-log caches effective levels). A missing
    # native install just means no native logs to bridge.
    with contextlib.suppress(ImportError):
        import shrike_native

        shrike_native.init_logging()

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
    server_lock = ServerLock(state_dir_override=state_dir_override)
    server_meta = {
        "pid": None,
        "url": f"http://{args.host}:{args.port}/mcp",
        "host": args.host,
        "port": args.port,
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

    # Kernel mode (#332 S3d-2): the kernel owns the collection, the vector
    # index, and the derived ingest; the index files live in the cache dir as
    # before. The media resolver pair is path-derived (lock-free) and feeds
    # the kernel's image seam for a CLIP-style backend.
    cache_base = Path(args.cache_dir) if args.cache_dir else cache_dir()
    collection_abs = os.path.abspath(args.collection)
    media_base = (
        collection_abs[: -len(".anki2")] if collection_abs.endswith(".anki2") else collection_abs
    )
    _read_img, _img_exists = _make_image_resolver(media_base + ".media")

    # llama-server stays on loopback regardless of the MCP bind host — there is
    # never a reason to expose the embedding backend to the network.
    resolved_state_dir = state_dir_override or state_dir()
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
        # The capability config carries these (#501); the legacy flag path
        # leaves them at the runtime defaults (text-only, no projectors).
        "modalities": None,
        "mmprojs": None,
    }
    # The v2 recognizers: a list of harness-ready plans (#485). The legacy
    # (no --config) path uses the --ocr-backend flag instead and leaves this
    # empty; a v2 config fills it from the resolved profile.
    recognizer_boot_plans: tuple[Any, ...] = ()
    # The SECONDARY embedding spaces' runtime params (#233): the 2nd+ entries of
    # a multi-space v2 config. Empty for the legacy/flag path and the N=1 case,
    # so the single-space boot is byte-identical (no secondary runtime built).
    secondary_param_sets: tuple[dict[str, Any], ...] = ()
    if args.config:
        # The daemon resolves the v2 capability sections itself (#498):
        # structured entries (remote endpoints, api_key_env) have no flag
        # spelling, so the CLI hands over the config path instead of params.
        # A ProfileError here is a config error — refuse to boot, loudly
        # (the CLI pre-validates, so this is the direct-invocation backstop).
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
        from shrike.profiles import (
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
            # launch paths (#610): the CLI's resolve_embedding_profile short-
            # circuits caps.legacy to this same resolve_embedding, never running
            # the build-validating resolve_profile. Routing the daemon --config
            # path here too is what keeps the two paths consistent — the legacy
            # cascade is "warn-and-map for one release" (#523 removes it), so the
            # intended behavior is degrade, not refuse-boot. A real v2-config
            # error still hits resolve_profile below and refuses.
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
            # The resolved recognizers (#485): describe (and, post-#502, remote OCR)
            # ride the v2 config — no flag spelling, so they reach boot from here.
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
            # the 2nd+ entries become SECONDARY spaces (#233), each its own runtime.
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
            **(
                {"modalities": params["modalities"]} if params.get("modalities") is not None else {}
            ),
            **({"mmprojs": params["mmprojs"]} if params.get("mmprojs") is not None else {}),
        )

    runtime = _runtime_from_params(emb_params, pid_file=resolved_state_dir / "embedding.pid")
    # The SECONDARY embedding spaces (#233): each 2nd+ v2 entry is its own
    # runtime, attached to its own kernel embed space. Each needs a default
    # backend kind (a None backend is the no-embedder shape, which secondaries
    # never are). Only a MANAGED llama-server secondary writes a pid file, and
    # it would collide with the primary's, so secondaries get none (the in-
    # process onnx/clip and remote backends — the multi-space shapes — have no
    # subprocess to reap). Empty for the N=1 / legacy path → byte-identical.
    secondary_runtimes = [
        _runtime_from_params({**p, "backend": p.get("backend") or DEFAULT_BACKEND}, pid_file=None)
        for p in secondary_param_sets
    ]

    # The derived-text store (FTS5 trigram sidecar) — engine factory injected
    # here, like the index engine (the harness owns assembly, #278 C5). The
    # store is namespaced per collection (#547): it opens the SAME
    # `<cache_dir>/derived/<namespace>/shrike.db` the kernel's DerivedEngine
    # writes (they share one file — the kernel ingests, this host surface
    # reads), so a daemon serving several collections never shares one db.
    # Pass the SAME collection-path string the kernel receives
    # (`args.collection`, below) so both sides feed `index_namespace` identical
    # input and resolve to the same db — canonicalization (realpath/abspath)
    # then folds any spelling difference, but identical input needs no folding.
    derived = DerivedTextStore(
        path=derived_db_path(str(cache_base), args.collection),
        engine_factory=NativeDerivedEngine,
    )

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
        """Validate + canonicalize an operator-allowed root list (#71): per
        element (dedup, order-preserving) — the containment disjunction means
        the weakest root governs, so one bad root (filesystem root, missing dir)
        fails startup, not passes silently. Honored only on a purely-local
        server; a root set on a non-purely-local one is refused (warn), never
        half-enabled. Returns the active roots (empty → the capability stays
        off)."""
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

    # store_media's server-local `path` read (#170) and export's `output_path`
    # write (#71): distinct capabilities, distinct root lists, the same gate.
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

    # import_package's server-local `path` read (#72): a DISTINCT capability
    # from store_media's read and export's write — import writes into the
    # collection (a merge), a higher blast radius than a media-file read — so
    # its own root list, never inheriting a media or export root. Same shared
    # gate (#71's _resolve_path_roots): purely-local + per-root containment.
    server_import_path_roots = _resolve_path_roots(
        args.import_path_root,
        "SHRIKE_IMPORT_PATH_ROOTS",
        "--import-path-root",
        "import_package server-local paths",
    )

    # The collection/profile registry (#66): a read-only snapshot for the
    # `list_profiles` enumeration action. Loaded from the server's config file
    # (the explicit --config, or the platform default), best-effort — a missing
    # or unreadable config yields an empty registry, never a boot failure. The
    # registry is host-side config; the server operates on --collection today,
    # and routing by selector is the capstone (#68).
    from shrike.cli.config import DEFAULT_CONFIG_PATH, load_config
    from shrike.registry import Registry

    profile_registry: Registry | None = None
    try:
        registry_config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
        profile_registry = Registry.from_config(load_config(registry_config_path))
    except Exception:  # noqa: BLE001 — enumeration is a convenience, never gates boot
        logger.debug("profile registry unavailable; list_profiles will report empty")

    # The export download store (#71): server-named temp packages under the
    # cache dir + their one-shot download tokens, reaped on download / TTL /
    # shutdown. Backs the default export delivery (the GET /export/{token}
    # route below); the server-local output_path mode bypasses it.
    from shrike.export_store import ExportStore

    export_store = ExportStore(str(cache_base))

    # The cross-space image floor margin (#580): config/env-resolved (no flag —
    # an operational knob, not a v2 capability section), so the server loads it
    # from the config file the daemon was started with. Default 1.0.
    from shrike.cli.config import load_config, resolve_cross_space_margin

    try:
        _margin_config = load_config(Path(args.config)) if args.config else load_config()
    except Exception:  # noqa: BLE001 — a missing/garbage config falls back to the default margin
        _margin_config = {}
    cross_space_floor_margin = resolve_cross_space_margin(_margin_config)

    async def _serve() -> None:
        # Assembly runs ON the loop (#332 S3d-2): the kernel opens with a
        # dedicated harness thread driving its executor; the wrapper rides the
        # shared core; tools/routes register before the socket binds (no
        # request is accepted until serve()).
        logger.info("Opening collection at %s", args.collection)
        harness = await Harness.assemble(
            collection_path=args.collection,
            cache_dir=str(cache_base),
            runtime=runtime,
            derived=derived,
            cooperative=args.cooperative_lock,
            hold_seconds=hold_seconds,
            media_read=_read_img,
            media_exists=_img_exists,
            index_save_delay=args.index_save_delay,
            index_save_threshold=args.index_save_threshold,
            secondary_runtimes=secondary_runtimes,
            cross_space_floor_margin=cross_space_floor_margin,
        )
        # Embedding starts at boot when anything configures it: a model (flag
        # or config entry) OR a bare endpoint (#498 — a remote/attach entry's
        # endpoint default model is a valid configuration with no model name).
        embedding_configured = bool(emb_params.get("model") or emb_params.get("endpoint"))
        await harness.boot(start_embedding=embedding_configured and not args.no_embedding)

        # Recognition (#228/#221/#485): attach recognizers and sweep in the
        # background. Off unless configured; a dead endpoint / missing engine
        # degrades to an 'error' state row without disturbing the rest of the
        # server. The legacy --ocr-backend flag still drives OCR directly; the
        # v2 recognizers: config (describe today, remote OCR post-#502) drives
        # the rest from the resolved profile.
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

        # Multi-collection routing (#68): the manager wraps the boot harness as
        # the default collection and lazily assembles a per-collection harness
        # for any other registered profile a call selects (its own namespaced
        # index + per-collection derived store, sharing this base cache dir and
        # this embedding runtime). The default harness owns the shared runtime;
        # routed harnesses attach its backend and never stop it. Selector
        # plumbing through the tools/CLI is S2; per-collection status is S3.
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
            # exits now. Routed collections (#68) are flushed too; the shared
            # runtime is stopped once.
            for h in manager.active_harnesses():
                with contextlib.suppress(Exception):
                    h.kernel.save_index()
                with contextlib.suppress(Exception):
                    h.derived.close()
            with contextlib.suppress(Exception):
                harness.runtime.stop()
            with contextlib.suppress(Exception):
                export_store.close()  # reap any pending download temps (#71)
            server_lock.release()
            logger.info("Shutdown complete")
            sys.exit(0)

        signal.signal(signal.SIGTERM, _signal_shutdown)
        signal.signal(signal.SIGINT, _signal_shutdown)

        # register_tools binds the action registry to MCP and returns the same
        # actions as a name->Tool map for the actions-over-HTTP edge (#505).
        action_tools = register_tools(
            mcp,
            harness.wrapper,
            index=harness.index_view,
            derived=derived,
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
            # Per-call collection routing (#68 S2): the manager resolves a
            # selector to the right collection's bundle, lazily assembling it.
            resolver=manager.resolve_bundle,
        )
        # The uvicorn Server is created after route registration, so the
        # /shutdown route reaches it through this late-bound holder (#344).
        server_holder: list[Any] = []

        def _request_shutdown() -> None:
            if server_holder:
                server_holder[0].should_exit = True

        _register_custom_routes(
            mcp,
            harness,
            server_lock,
            meta=server_meta,
            security=transport_security,
            request_shutdown=_request_shutdown,
            action_tools=action_tools,
            manager=manager,
            export_store=export_store,
        )

        logger.info(
            "Listening on %s:%s (log_dir=%s, log_level=%s)",
            args.host,
            args.port,
            log_dir,
            args.log_level or "info",
        )
        import uvicorn

        config = uvicorn.Config(
            mcp.streamable_http_app(),
            host=args.host,
            port=args.port,
            log_config=None,
            # Bound the graceful drain so a hung in-flight request can't
            # wedge a /shutdown forever (the daemon's stop path escalates to
            # SIGTERM/SIGKILL regardless).
            timeout_graceful_shutdown=5,
        )
        server = uvicorn.Server(config)
        server_holder.append(server)
        await server.serve()

        # serve() returned: either /shutdown set should_exit (#344 — the
        # graceful path; teardown belongs here, after the listener drained)
        # or uvicorn exited without a replayed signal. A SIGTERM replays into
        # _signal_shutdown above instead.
        logger.info("Server drained; shutting down")
        with contextlib.suppress(Exception):
            # Close every routed collection too (#68), default last (it stops
            # the shared embedding runtime after the routed ones detach).
            await manager.close()
        with contextlib.suppress(Exception):
            export_store.close()  # reap any pending download temps (#71)
        server_lock.release()
        logger.info("Shutdown complete")

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
