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
from shrike.log import configure_logging
from shrike.tools import register_tools

logger = logging.getLogger("shrike.server")

mcp = FastMCP(
    "Shrike",
    stateless_http=True,
    json_response=True,
)


def _register_shutdown_route(
    app: FastMCP,
    wrapper: CollectionWrapper,
    server_lock: ServerLock,
) -> None:
    """Register the POST /shutdown endpoint for cross-platform clean shutdown."""
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @app.custom_route("/shutdown", methods=["POST"])
    async def handle_shutdown(request: Request) -> JSONResponse:
        logger.info("Shutdown requested via HTTP from %s", request.client)
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
    args = parser.parse_args()

    log_dir = configure_logging(
        foreground=args.foreground,
        log_dir_override=args.log_dir,
        log_level_override=args.log_level,
    )

    # Acquire the daemon lock before touching the collection
    state_dir_override = Path(args.state_dir) if args.state_dir else None
    server_lock = ServerLock(state_dir_override=state_dir_override)
    try:
        server_lock.acquire(
            meta={
                "pid": None,  # will be overwritten after acquire sets it
                "url": f"http://{args.host}:{args.port}/mcp",
                "host": args.host,
                "port": args.port,
                "collection": args.collection,
                "log_dir": str(log_dir),
                "log_level": args.log_level or "info",
                "started": datetime.now(UTC).isoformat(),
            }
        )
    except AlreadyRunningError as e:
        logger.error("Cannot start: %s", e)
        sys.exit(1)

    logger.info("Opening collection at %s", args.collection)
    wrapper = CollectionWrapper(args.collection)
    info = wrapper.get_collection_info(include=["note_types", "decks", "stats"])
    note_count = info.get("stats", {}).get("total_notes", 0)
    deck_count = len(info.get("decks", []))
    type_count = len(info.get("note_types", []))
    logger.info(
        "Collection ready: %d notes, %d decks, %d note types",
        note_count,
        deck_count,
        type_count,
    )

    def _signal_shutdown(signum: int, frame: Any) -> None:  # noqa: ARG001
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, shutting down", sig_name)
        wrapper.close()
        server_lock.release()
        logger.info("Shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_shutdown)
    signal.signal(signal.SIGINT, _signal_shutdown)

    register_tools(mcp, wrapper)
    _register_shutdown_route(mcp, wrapper, server_lock)

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
