from __future__ import annotations

import argparse
import asyncio
import functools
import ipaddress
import logging
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import (
    TransportSecurityMiddleware,
    TransportSecuritySettings,
)

from shrike.collection import CollectionWrapper
from shrike.daemon import AlreadyRunningError, ServerLock
from shrike.embedding import EmbeddingRuntime
from shrike.index import (
    DEFAULT_SAVE_DELAY,
    DEFAULT_SAVE_THRESHOLD,
    IndexSaver,
    VectorIndex,
)
from shrike.log import configure_logging
from shrike.paths import cache_dir, state_dir
from shrike.tools import register_tools

logger = logging.getLogger("shrike.server")


def _collect_for_rebuild(c: Any) -> tuple[list[int], int, list[str]]:
    """Gather all note ids, the collection mod stamp, and embedding texts.

    Runs on the collection worker thread (receives the live ``Collection``).
    """
    note_ids = list(c.find_notes("deck:*"))
    return note_ids, c.mod, CollectionWrapper.note_texts(c, note_ids)


def _maybe_rebuild(
    index: VectorIndex,
    model_id: str,
    col_mod: int,
    note_ids: list[int],
    texts: list[str],
) -> bool:
    """Trigger a background rebuild if the index drifted or the model changed.

    Returns True if a rebuild was started (drift detected and the collection is
    non-empty), False otherwise.
    """
    if index.check_drift(col_mod, model_id):
        if note_ids:
            index.rebuild_in_background(note_ids, texts, col_mod, model_id=model_id)
            return True
        logger.info("Collection is empty, skipping index rebuild")
    return False


# Host/Origin values accepted when the server is bound to a loopback address.
# The port is wildcarded (`:*`) so any port the user picked is allowed; matching
# is exact on the host part, which is what stops DNS-rebinding (a page on
# ``evil.com`` resolving to 127.0.0.1 still sends ``Host: evil.com``).
_LOOPBACK_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOOPBACK_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


def _is_loopback(host: str) -> bool:
    """True if *host* names the loopback interface (so binding is browser-safe)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


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
    local proxy can trust the proxy's forwarded hostname without opening up.
    """
    extra_hosts = allowed_hosts or []
    extra_origins = allowed_origins or []

    if disable:
        return None
    if not _is_loopback(host) and not extra_hosts and not extra_origins:
        return None

    base_hosts = list(_LOOPBACK_HOSTS) if _is_loopback(host) else []
    base_origins = list(_LOOPBACK_ORIGINS) if _is_loopback(host) else []
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=base_hosts + extra_hosts,
        allowed_origins=base_origins + extra_origins,
    )


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
    """
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
    wrapper: CollectionWrapper,
    server_lock: ServerLock,
    *,
    meta: dict[str, Any],
    runtime: EmbeddingRuntime,
    index: VectorIndex,
    saver: IndexSaver,
    security: TransportSecuritySettings | None,
) -> None:
    """Register custom HTTP endpoints on the server.

    The custom routes bypass the MCP transport middleware, so they get the same
    Host/Origin validation applied here via ``_guard`` — otherwise a browser page
    could drive ``/shutdown`` etc. through a no-preflight POST.
    """
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response

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
            rejection = await security_mw.validate_request(request, is_post=False)
            if rejection is not None:
                return rejection
            return await handler(request)

        return wrapped

    @app.custom_route("/status", methods=["GET"])
    @_guard
    async def handle_status(request: Request) -> JSONResponse:
        import contextlib

        status: dict[str, Any] = {
            "running": True,
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

        # health() probes llama-server over HTTP; run it off the event loop so a
        # slow/hung embedding server can't stall request handling.
        status["embedding"] = await asyncio.to_thread(runtime.health)
        status["index"] = index.status()

        return JSONResponse(status)

    @app.custom_route("/index/rebuild", methods=["POST"])
    @_guard
    async def handle_index_rebuild(request: Request) -> JSONResponse:
        from shrike.index import IndexState

        svc = runtime.service
        if svc is None or not svc.running:
            return JSONResponse({"error": "Embedding service is not running"}, status_code=400)

        if index.state == IndexState.BUILDING:
            indexed, total = index.build_progress
            return JSONResponse(
                {
                    "status": "already_building",
                    "progress": {"indexed": indexed, "total": total},
                }
            )

        model_id = await asyncio.to_thread(svc.model_fingerprint)
        all_note_ids, col_mod, texts = await wrapper.run(_collect_for_rebuild)
        if not all_note_ids:
            index.rebuild([], [], col_mod, model_id=model_id)
            return JSONResponse({"status": "complete", "size": 0})

        index.rebuild_in_background(all_note_ids, texts, col_mod, model_id=model_id)
        return JSONResponse(
            {
                "status": "started",
                "total": len(all_note_ids),
            }
        )

    @app.custom_route("/index/save", methods=["POST"])
    @_guard
    async def handle_index_save(request: Request) -> JSONResponse:
        from shrike.index import IndexState

        # Refuse mid-rebuild: a save here would persist a partial index with a
        # stale col_mod, and rebuild() saves at its own end anyway.
        if index.state == IndexState.BUILDING:
            indexed, total = index.build_progress
            return JSONResponse(
                {"status": "building", "progress": {"indexed": indexed, "total": total}}
            )
        if index.ndim is None:
            return JSONResponse({"status": "empty"})

        pending = index.pending_changes
        await asyncio.to_thread(index.save)
        return JSONResponse({"status": "saved", "size": index.size, "pending": pending})

    @app.custom_route("/embedding/start", methods=["POST"])
    @_guard
    async def handle_embedding_start(request: Request) -> JSONResponse:
        if runtime.running:
            return JSONResponse({"status": "already_running", "embedding": runtime.health()})

        import contextlib

        overrides: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            body = await request.json()
            if isinstance(body, dict):
                for key in (
                    "model",
                    "port",
                    "context_size",
                    "threads",
                    "gpu_layers",
                    "pooling",
                    "extra_args",
                    "llama_server",
                ):
                    if body.get(key) is not None:
                        overrides[key] = body[key]

        logger.info("Embedding start requested via HTTP from %s", request.client)
        try:
            # Starting llama-server blocks (model load + health wait); run it off
            # the event loop so other requests keep flowing.
            svc = await asyncio.to_thread(lambda: runtime.start(**overrides))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except (FileNotFoundError, RuntimeError) as e:
            logger.error("Failed to start embedding service: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

        model_id = await asyncio.to_thread(svc.model_fingerprint)
        all_note_ids, col_mod, texts = await wrapper.run(_collect_for_rebuild)
        _maybe_rebuild(index, model_id, col_mod, all_note_ids, texts)

        return JSONResponse(
            {
                "status": "started",
                "embedding": runtime.health(),
                "index": index.status(),
            }
        )

    @app.custom_route("/embedding/stop", methods=["POST"])
    @_guard
    async def handle_embedding_stop(request: Request) -> JSONResponse:
        if not runtime.running:
            return JSONResponse({"status": "not_running"})

        logger.info("Embedding stop requested via HTTP from %s", request.client)
        # Persist current vectors before tearing down the embedder.
        index.save()
        await asyncio.to_thread(runtime.stop)
        return JSONResponse({"status": "stopped", "index": index.status()})

    @app.custom_route("/reload", methods=["POST"])
    @_guard
    async def handle_reload(request: Request) -> JSONResponse:
        logger.info("Reload requested via HTTP from %s", request.client)
        # Close and re-open the collection, picking up on-disk changes.
        await wrapper.reopen()
        col_mod = await wrapper.run(lambda c: c.mod)

        # Re-check index drift against the re-opened collection. Without a running
        # embedder we can't rebuild (the index stays unavailable); just report.
        rebuilding = False
        svc = runtime.service
        if svc is not None and svc.running:
            model_id = await asyncio.to_thread(svc.model_fingerprint)
            all_note_ids, new_col_mod, texts = await wrapper.run(_collect_for_rebuild)
            rebuilding = _maybe_rebuild(index, model_id, new_col_mod, all_note_ids, texts)

        return JSONResponse({"status": "reloaded", "col_mod": col_mod, "rebuilding": rebuilding})

    @app.custom_route("/shutdown", methods=["POST"])
    @_guard
    async def handle_shutdown(request: Request) -> JSONResponse:
        logger.info("Shutdown requested via HTTP from %s", request.client)
        # aclose cancels the pending debounce timer and flushes if dirty.
        await saver.aclose()
        runtime.stop()
        wrapper.close()
        server_lock.release()
        logger.info("Shutdown complete")

        async def _exit_after_response() -> None:
            await asyncio.sleep(0.1)
            os._exit(0)

        asyncio.create_task(_exit_after_response())
        return JSONResponse({"status": "ok", "pid": os.getpid()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Shrike MCP server for Anki")
    parser.add_argument(
        "--collection",
        required=True,
        help="Path to the Anki collection file (collection.anki2)",
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
        "--embedding-model",
        default=None,
        help="Path to GGUF embedding model (enables embedding service)",
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
        "settings like --embedding-pooling.",
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Don't start the embedding service at boot even if a model is configured "
        "(start it later with 'shrike embedding start')",
    )
    args = parser.parse_args()

    log_dir = configure_logging(
        foreground=args.foreground,
        log_dir_override=args.log_dir,
        log_level_override=args.log_level,
    )

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

    logger.info("Opening collection at %s", args.collection)
    wrapper = CollectionWrapper(args.collection)
    notes, decks, note_types = wrapper.run_sync(
        lambda c: (c.note_count(), len(c.decks.all_names_and_ids()), len(c.models.all()))
    )
    logger.info(
        "Collection ready: %d notes, %d decks, %d note types",
        notes,
        decks,
        note_types,
    )

    # The index is always created — it can hold on-disk vectors and report
    # status even with no embedder. It reports UNAVAILABLE until a service is
    # attached, so the embedding lifecycle can be cycled at runtime.
    cache_base = Path(args.cache_dir) if args.cache_dir else cache_dir()
    index = VectorIndex(path=cache_base / "index")
    logger.info("Vector index: %d vectors, %d dims", index.size, index.ndim or 0)

    # Debounced persistence: incremental edits flush after a quiet period (or a
    # burst cap), so an idle server that's hard-killed reloads without a full
    # re-embed. The save itself runs off the event loop.
    saver = IndexSaver(
        index,
        delay=(args.index_save_delay if args.index_save_delay is not None else DEFAULT_SAVE_DELAY),
        threshold=(
            args.index_save_threshold
            if args.index_save_threshold is not None
            else DEFAULT_SAVE_THRESHOLD
        ),
    )

    # llama-server stays on loopback regardless of the MCP bind host — there is
    # never a reason to expose the embedding backend to the network.
    resolved_state_dir = state_dir_override or state_dir()
    runtime = EmbeddingRuntime(
        index=index,
        model=args.embedding_model,
        host="127.0.0.1",
        port=args.embedding_port or 8373,
        log_dir=log_dir,
        context_size=args.embedding_context_size,
        threads=args.embedding_threads,
        gpu_layers=args.embedding_gpu_layers,
        pooling=args.embedding_pooling,
        extra_args=args.embedding_arg,
        llama_server=args.llama_server,
        pid_file=resolved_state_dir / "embedding.pid",
    )

    if args.embedding_model and not args.no_embedding:
        try:
            svc = runtime.start()
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            logger.error("Failed to start embedding service: %s", e)
        else:
            model_id = svc.model_fingerprint()
            all_note_ids, col_mod, texts = wrapper.run_sync(_collect_for_rebuild)
            _maybe_rebuild(index, model_id, col_mod, all_note_ids, texts)
    elif args.no_embedding and args.embedding_model:
        logger.info("Embedding service disabled at boot (--no-embedding); model configured")

    def _signal_shutdown(signum: int, frame: Any) -> None:  # noqa: ARG001
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, shutting down", sig_name)
        index.save()
        runtime.stop()
        wrapper.close()
        server_lock.release()
        logger.info("Shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_shutdown)
    signal.signal(signal.SIGINT, _signal_shutdown)

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
    register_tools(mcp, wrapper, index=index, saver=saver)
    _register_custom_routes(
        mcp,
        wrapper,
        server_lock,
        meta=server_meta,
        runtime=runtime,
        index=index,
        saver=saver,
        security=transport_security,
    )

    logger.info(
        "Listening on %s:%s (log_dir=%s, log_level=%s)",
        args.host,
        args.port,
        log_dir,
        args.log_level or "info",
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
