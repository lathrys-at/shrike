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

from shrike._mcp_perf import install_validator_cache
from shrike.collection import DEFAULT_LOCK_HOLD, CollectionWrapper
from shrike.daemon import AlreadyRunningError, ServerLock
from shrike.derived import DerivedTextStore
from shrike.embedding import DEFAULT_BACKEND, SUPPORTED_BACKENDS, EmbeddingRuntime
from shrike.embedding_base import EmbedderBackend
from shrike.index import (
    DEFAULT_SAVE_DELAY,
    DEFAULT_SAVE_THRESHOLD,
    IndexSaver,
    NoteEmbedInput,
    VectorIndex,
)
from shrike.log import configure_logging
from shrike.paths import cache_dir, state_dir
from shrike.tools import register_tools

logger = logging.getLogger("shrike.server")


def _collect_for_rebuild(c: Any) -> tuple[list[NoteEmbedInput], int]:
    """Gather every note's embedding input (text + image names) and the collection mod stamp.

    Runs on the collection worker thread (receives the live ``Collection``).
    """
    note_ids = list(c.find_notes("deck:*"))
    return CollectionWrapper._note_embed_inputs(c, note_ids), c.mod


def _collect_derived_rows(c: Any) -> tuple[list[tuple[int, str, str, str]], int]:
    """Gather every note's ``(note_id, "field", field_name, raw_value)`` rows + the mod stamp.

    The full-build input for the derived-text store (#98). Runs on the collection worker thread.
    Independent of the embedding index — the store builds whether or not a backend is configured.
    """
    note_ids = list(c.find_notes("deck:*"))
    return list(CollectionWrapper.derived_field_rows(c, note_ids)), c.mod


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


def _maybe_rebuild(
    index: VectorIndex,
    model_id: str,
    col_mod: int,
    inputs: list[NoteEmbedInput],
    embedding: EmbedderBackend,
) -> bool:
    """Reconcile the index in the background if it drifted or the model changed.

    Drift (an external ``col.mod`` bump) reconciles incrementally — re-embedding
    only the notes whose text changed — rather than re-embedding the whole
    collection; ``reconcile`` itself falls back to a full rebuild when the model
    changed or there's no prior per-note state. Returns True if work was started
    (drift detected and the collection is non-empty), False otherwise.

    When the collection is *empty* there is nothing to embed, but we still
    materialize an empty, ready index at the model's dimension so notes added
    later in the session are indexed incrementally instead of being skipped until
    a restart (#148).
    """
    if index.check_drift(col_mod, model_id):
        if inputs:
            index.reconcile_in_background(inputs, col_mod, model_id=model_id)
            return True  # the background reconcile recalibrates the activation gate at its tail
        ndim = embedding.embedding_dim()
        if ndim is not None:
            index.materialize_empty(ndim, col_mod, model_id)
        else:
            logger.info("Collection is empty and embedding dim unknown, skipping index rebuild")
    # No background rebuild was started (the index loaded clean, or the collection is empty). Make
    # sure a clean index that predates the activation gate (#201b) gets calibrated now rather than
    # waiting for the next drift; a no-op when it's already calibrated or has no non-text modality.
    # We're already off the event loop here (callers run _maybe_rebuild via asyncio.to_thread).
    index.ensure_calibrated()
    return False


# Host/Origin values accepted when the server is bound to a loopback address.
# The port is wildcarded (`:*`) so any port the user picked is allowed; matching
# is exact on the host part, which is what stops DNS-rebinding (a page on
# ``evil.com`` resolving to 127.0.0.1 still sends ``Host: evil.com``).
_LOOPBACK_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOOPBACK_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


def _positive_int(raw: str) -> int:
    """argparse type: a >= 1 integer (e.g. --embedding-batch-size)."""
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1 (got {value})")
    return value


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


def _server_is_purely_local(
    host: str,
    *,
    allow_remote: bool,
    no_dns_rebinding_protection: bool,
    allowed_hosts: list[str] | None,
    allowed_origins: list[str] | None,
) -> bool:
    """Whether the server is in its default, purely-local configuration (#164).

    Gates the server-local ``path`` input to ``store_media``: only when the bind
    is loopback, ``--allow-remote`` is off, the DNS-rebinding guard is on, and no
    extra ``--allowed-host``/``--allowed-origin`` was added. Any of those signals
    possible remote traffic — and crucially, behind a same-host reverse proxy /
    tailnet (``--no-dns-rebinding-protection`` or an added allow-list) the loopback
    peer is the proxy, not the real (remote) client — so server-local file reads
    must stay disabled there.
    """
    return (
        _is_loopback(host)
        and not allow_remote
        and not no_dns_rebinding_protection
        and not allowed_hosts
        and not allowed_origins
    )


def _validate_media_path_root(raw: str) -> str:
    """Canonicalize and validate ``--media-path-root`` at startup (#170), or raise.

    Returns the resolved absolute real path (symlinks collapsed) used for the
    store_media containment check. Rejects the filesystem root (``dirname(p) == p``
    is true for ``/``, a Windows drive root, etc. — confining to ``/`` is no
    confinement) and a root that isn't an existing directory (so it can't 'refuse
    everything' or spring into existence later)."""
    resolved = os.path.realpath(os.path.expanduser(raw))
    if os.path.dirname(resolved) == resolved:
        raise ValueError(f"refusing the filesystem root '{resolved}' (confines nothing)")
    if not os.path.isdir(resolved):
        raise ValueError(f"'{raw}' is not an existing directory")
    return resolved


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
    derived: DerivedTextStore,
    security: TransportSecuritySettings | None,
) -> None:
    """Register custom HTTP endpoints on the server.

    The custom routes bypass the MCP transport middleware, so they get the same
    Host/Origin validation applied here via ``_guard`` — otherwise a browser page
    could drive ``/shutdown`` etc. through a no-preflight POST.
    """
    from starlette.requests import Request
    from starlette.responses import FileResponse, JSONResponse, Response

    from shrike.collection import _safe_media_name

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
        status["derived"] = derived.status()
        status["locking"] = "cooperative" if wrapper.cooperative else "permanent"
        status["collection_held"] = wrapper.is_open

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
        inputs, col_mod = await wrapper.run(_collect_for_rebuild)
        if not inputs:
            index.rebuild([], col_mod, model_id=model_id)
            return JSONResponse({"status": "complete", "size": 0})

        index.rebuild_in_background(inputs, col_mod, model_id=model_id)
        return JSONResponse(
            {
                "status": "started",
                "total": len(inputs),
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

        logger.info("Embedding start requested via HTTP from %s", request.client)
        try:
            # Starting a backend blocks (model load + health wait); run it off the
            # event loop so other requests keep flowing.
            svc = await asyncio.to_thread(lambda: runtime.start(**overrides))
        except (ValueError, ImportError) as e:
            # Unknown backend / no model (ValueError) or a missing ONNX optional
            # dependency (ImportError) are caller-actionable config errors → 400.
            return JSONResponse({"error": str(e)}, status_code=400)
        except (FileNotFoundError, RuntimeError) as e:
            logger.error("Failed to start embedding service: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

        model_id = await asyncio.to_thread(svc.model_fingerprint)
        inputs, col_mod = await wrapper.run(_collect_for_rebuild)
        await asyncio.to_thread(_maybe_rebuild, index, model_id, col_mod, inputs, svc)

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

        # The derived-text store is independent of the embedder — rebuild it on drift regardless
        # (cheap text-only build).
        if derived.check_drift(col_mod):
            rows, dmod = await wrapper.run(_collect_derived_rows)
            derived.build_in_background(rows, dmod)

        # Re-check index drift against the re-opened collection. Without a running
        # embedder we can't rebuild (the index stays unavailable); just report.
        rebuilding = False
        svc = runtime.service
        if svc is not None and svc.running:
            model_id = await asyncio.to_thread(svc.model_fingerprint)
            inputs, new_col_mod = await wrapper.run(_collect_for_rebuild)
            rebuilding = await asyncio.to_thread(
                _maybe_rebuild, index, model_id, new_col_mod, inputs, svc
            )

        return JSONResponse({"status": "reloaded", "col_mod": col_mod, "rebuilding": rebuilding})

    @app.custom_route("/shutdown", methods=["POST"])
    @_guard
    async def handle_shutdown(request: Request) -> JSONResponse:
        logger.info("Shutdown requested via HTTP from %s", request.client)
        # aclose cancels the pending debounce timer and flushes if dirty.
        await saver.aclose()
        derived.close()
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
        choices=list(SUPPORTED_BACKENDS),
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
    hold_seconds = (
        args.lock_hold_seconds if args.lock_hold_seconds is not None else DEFAULT_LOCK_HOLD
    )
    wrapper = CollectionWrapper(
        args.collection,
        cooperative=args.cooperative_lock,
        hold_seconds=hold_seconds,
    )
    if args.cooperative_lock:
        logger.info(
            "Cooperative locking on: releasing the collection after %.0fs idle", hold_seconds
        )
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
    # Let the index read image bytes + cheaply check presence (lock-free, off the worker thread)
    # for a CLIP-style backend; inert for a text-only backend. media_dir is path-derived → safe
    # before open.
    _read_img, _img_exists = _make_image_resolver(wrapper.media_dir)
    index.set_image_resolver(_read_img, _img_exists)
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
        backend=args.embedding_backend or DEFAULT_BACKEND,
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
        onnx_providers=args.embedding_onnx_provider,
        batch_size=args.embedding_batch_size,
    )

    if args.embedding_model and not args.no_embedding:
        try:
            svc = runtime.start()
        except (FileNotFoundError, RuntimeError, ValueError, ImportError) as e:
            # ImportError: the onnx backend was selected without the optional
            # 'onnx' extra installed. Like a missing model file, degrade — boot
            # without embedding rather than killing the server (the /embedding/start
            # handler returns 400 for the same case).
            logger.error("Failed to start embedding service: %s", e)
        else:
            model_id = svc.model_fingerprint()
            inputs, col_mod = wrapper.run_sync(_collect_for_rebuild)
            _maybe_rebuild(index, model_id, col_mod, inputs, svc)
    elif args.no_embedding and args.embedding_model:
        logger.info("Embedding service disabled at boot (--no-embedding); model configured")

    # The derived-text store (FTS5 trigram sidecar) is independent of the embedding index — it
    # builds whether or not a backend is configured. Cheap col_mod probe first; only read all field
    # text on real drift (first build, or an external edit), so a clean reload does no full read.
    derived = DerivedTextStore(path=cache_base / "shrike.db")
    logger.info("Derived-text store: %s", derived.status())
    d_col_mod = wrapper.run_sync(lambda c: c.mod)
    if derived.check_drift(d_col_mod):
        rows, dmod = wrapper.run_sync(_collect_derived_rows)
        derived.build_in_background(rows, dmod)
        logger.info("Derived-text store drift; building in background (%d rows)", len(rows))

    if args.cooperative_lock:
        # On every re-acquire after an idle release, re-check index drift: if the
        # collection changed on disk while we were released (Anki, sync, import),
        # rebuild. Cheap col_mod-only check; texts are read under the lock and
        # embedded off-lock only on real drift. Runs on the worker thread.
        def _acquire_hook(col: Any) -> None:
            # The derived-text store is independent of the embedder — rebuild it on drift even with
            # no embedding service (it's a cheap text-only build).
            if derived.check_drift(col.mod):
                rows, dmod = _collect_derived_rows(col)
                logger.info(
                    "Collection changed while idle; rebuilding derived store (%d rows)", len(rows)
                )
                derived.build_in_background(rows, dmod)
            if not index.available or not index.check_drift(col.mod):
                return
            svc_now = runtime.service
            if svc_now is None or not svc_now.running:
                return
            inputs, changed_mod = _collect_for_rebuild(col)
            logger.info("Collection changed while idle (col_mod=%d); rebuilding index", changed_mod)
            index.rebuild_in_background(inputs, changed_mod, model_id=svc_now.model_fingerprint())

        wrapper.set_acquire_hook(_acquire_hook)
        # Release now so a freshly-booted, never-touched idle daemon doesn't hold
        # the lock; the first request re-acquires on demand.
        wrapper.release_now()

    def _signal_shutdown(signum: int, frame: Any) -> None:  # noqa: ARG001
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, shutting down", sig_name)
        index.save()
        derived.close()
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
    # store_media's server-local `path` is OFF by default (#170): honored only
    # when the operator opts in with one or more --media-path-root, AND the server
    # is purely local, AND the path is contained in one of those roots. The two
    # gates compose — purely-local stops a remote/proxied caller from *reaching*
    # it; the roots bound *what* a permitted caller can read. Roots set on a
    # non-purely-local server are refused (warn) rather than silently half-enabled.
    raw_roots = list(args.media_path_root or [])
    env_roots = os.environ.get("SHRIKE_MEDIA_PATH_ROOTS")
    if env_roots:
        raw_roots += [p for p in env_roots.split(os.pathsep) if p]
    server_path_roots: list[str] = []
    if raw_roots:
        # Validate + canonicalize **per element** (dedup, order-preserving): the
        # containment disjunction means the weakest root governs, so a single bad
        # one (filesystem root, missing dir) must fail startup, not pass silently.
        validated: list[str] = []
        for r in raw_roots:
            try:
                resolved = _validate_media_path_root(r)
            except ValueError as e:
                logger.error("Invalid --media-path-root %r: %s", r, e)
                sys.exit(1)
            if resolved not in validated:
                validated.append(resolved)
        if _server_is_purely_local(
            args.host,
            allow_remote=args.allow_remote,
            no_dns_rebinding_protection=args.no_dns_rebinding_protection,
            allowed_hosts=args.allowed_host,
            allowed_origins=args.allowed_origin,
        ):
            server_path_roots = validated
            logger.info("store_media server-local paths enabled, confined to %s", validated)
        else:
            logger.warning(
                "--media-path-root is set but the server is not purely-local "
                "(remote/proxied exposure); store_media server-local paths stay disabled"
            )
    register_tools(
        mcp,
        wrapper,
        index=index,
        saver=saver,
        derived=derived,
        allow_private_fetch=allow_private_media_fetch,
        server_path_roots=server_path_roots,
        media_base_url=media_base_url,
    )
    _register_custom_routes(
        mcp,
        wrapper,
        server_lock,
        meta=server_meta,
        runtime=runtime,
        index=index,
        saver=saver,
        derived=derived,
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
