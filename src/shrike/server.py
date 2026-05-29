from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from shrike.collection import CollectionWrapper
from shrike.daemon import AlreadyRunningError, ServerLock
from shrike.embedding import EmbeddingRuntime
from shrike.index import VectorIndex
from shrike.log import configure_logging
from shrike.paths import cache_dir
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
) -> None:
    """Trigger a background rebuild if the index drifted or the model changed."""
    if index.check_drift(col_mod, model_id):
        if note_ids:
            index.rebuild_in_background(note_ids, texts, col_mod, model_id=model_id)
        else:
            logger.info("Collection is empty, skipping index rebuild")


def create_mcp() -> FastMCP:
    """Build a fresh FastMCP app.

    Constructed per-process inside ``main()`` rather than as an import-time
    global so the server is testable and re-usable in-process.
    """
    return FastMCP("Shrike", stateless_http=True, json_response=True)


def _register_custom_routes(
    app: FastMCP,
    wrapper: CollectionWrapper,
    server_lock: ServerLock,
    *,
    meta: dict[str, Any],
    runtime: EmbeddingRuntime,
    index: VectorIndex,
) -> None:
    """Register custom HTTP endpoints on the server."""
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @app.custom_route("/status", methods=["GET"])
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

    @app.custom_route("/embedding/start", methods=["POST"])
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
    async def handle_embedding_stop(request: Request) -> JSONResponse:
        if not runtime.running:
            return JSONResponse({"status": "not_running"})

        logger.info("Embedding stop requested via HTTP from %s", request.client)
        # Persist current vectors before tearing down the embedder.
        index.save()
        await asyncio.to_thread(runtime.stop)
        return JSONResponse({"status": "stopped", "index": index.status()})

    @app.custom_route("/shutdown", methods=["POST"])
    async def handle_shutdown(request: Request) -> JSONResponse:
        logger.info("Shutdown requested via HTTP from %s", request.client)
        index.save()
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

    runtime = EmbeddingRuntime(
        index=index,
        model=args.embedding_model,
        host=args.host,
        port=args.embedding_port or 8373,
        log_dir=log_dir,
        context_size=args.embedding_context_size,
        threads=args.embedding_threads,
        gpu_layers=args.embedding_gpu_layers,
        llama_server=args.llama_server,
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

    mcp = create_mcp()
    register_tools(mcp, wrapper, index=index)
    _register_custom_routes(
        mcp,
        wrapper,
        server_lock,
        meta=server_meta,
        runtime=runtime,
        index=index,
    )

    logger.info(
        "Listening on %s:%s (log_dir=%s, log_level=%s)",
        args.host,
        args.port,
        log_dir,
        args.log_level or "info",
    )
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
