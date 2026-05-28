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
from shrike.embedding import EmbeddingService
from shrike.index import VectorIndex
from shrike.log import configure_logging
from shrike.paths import cache_dir
from shrike.tools import register_tools

logger = logging.getLogger("shrike.server")

mcp = FastMCP(
    "Shrike",
    stateless_http=True,
    json_response=True,
)


def _register_custom_routes(
    app: FastMCP,
    wrapper: CollectionWrapper,
    server_lock: ServerLock,
    *,
    meta: dict[str, Any],
    embedding_service: EmbeddingService | None = None,
    index: VectorIndex | None = None,
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

        if embedding_service:
            status["embedding"] = embedding_service.health()

        if index:
            status["index"] = index.status()

        return JSONResponse(status)

    @app.custom_route("/index/rebuild", methods=["POST"])
    async def handle_index_rebuild(request: Request) -> JSONResponse:
        if index is None:
            return JSONResponse({"error": "No embedding service configured"}, status_code=400)

        from shrike.index import IndexState

        if index.state == IndexState.BUILDING:
            indexed, total = index.build_progress
            return JSONResponse(
                {
                    "status": "already_building",
                    "progress": {"indexed": indexed, "total": total},
                }
            )

        all_note_ids = list(wrapper.col.find_notes("deck:*"))
        if not all_note_ids:
            index.rebuild([], [], wrapper.col.mod)
            return JSONResponse({"status": "complete", "size": 0})

        texts = wrapper.note_texts_for_embedding(all_note_ids)
        index.rebuild_in_background(all_note_ids, texts, wrapper.col.mod)
        return JSONResponse(
            {
                "status": "started",
                "total": len(all_note_ids),
            }
        )

    @app.custom_route("/shutdown", methods=["POST"])
    async def handle_shutdown(request: Request) -> JSONResponse:
        logger.info("Shutdown requested via HTTP from %s", request.client)
        if index:
            index.save()
        if embedding_service:
            embedding_service.stop()
        wrapper.close()
        server_lock.release()
        logger.info("Shutdown complete")

        async def _exit_after_response() -> None:
            await asyncio.sleep(0.1)
            os._exit(0)

        asyncio.get_event_loop().create_task(_exit_after_response())
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
    summary = wrapper.get_collection_info().get("summary", {})
    logger.info(
        "Collection ready: %d notes, %d decks, %d note types",
        summary.get("notes", 0),
        summary.get("decks", 0),
        summary.get("note_types", 0),
    )

    embedding_service: EmbeddingService | None = None
    if args.embedding_model:
        embedding_service = EmbeddingService(
            model=args.embedding_model,
            host=args.host,
            port=args.embedding_port or 8373,
            log_dir=log_dir,
            context_size=args.embedding_context_size,
            threads=args.embedding_threads,
            gpu_layers=args.embedding_gpu_layers,
        )
        try:
            embedding_service.start()
        except (FileNotFoundError, RuntimeError) as e:
            logger.error("Failed to start embedding service: %s", e)
            embedding_service = None

    index: VectorIndex | None = None
    if embedding_service:
        cache_base = Path(args.cache_dir) if args.cache_dir else cache_dir()
        index_dir = cache_base / "index"
        index = VectorIndex(path=index_dir, embedding_service=embedding_service)
        logger.info("Vector index: %d vectors, %d dims", index.size, index.ndim or 0)

        if index.check_drift(wrapper.col.mod):
            all_note_ids = list(wrapper.col.find_notes("deck:*"))
            if all_note_ids:
                texts = wrapper.note_texts_for_embedding(all_note_ids)
                index.rebuild_in_background(all_note_ids, texts, wrapper.col.mod)
            else:
                logger.info("Collection is empty, skipping index rebuild")

    def _signal_shutdown(signum: int, frame: Any) -> None:  # noqa: ARG001
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, shutting down", sig_name)
        if index:
            index.save()
        if embedding_service:
            embedding_service.stop()
        wrapper.close()
        server_lock.release()
        logger.info("Shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_shutdown)
    signal.signal(signal.SIGINT, _signal_shutdown)

    register_tools(mcp, wrapper, index=index)
    _register_custom_routes(
        mcp,
        wrapper,
        server_lock,
        meta=server_meta,
        embedding_service=embedding_service,
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
