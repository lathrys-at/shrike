from __future__ import annotations

import argparse
import logging
import signal
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from shrike.collection import CollectionWrapper
from shrike.log import configure_logging
from shrike.tools import register_tools

logger = logging.getLogger("shrike")

mcp = FastMCP(
    "Shrike",
    stateless_http=True,
    json_response=True,
)


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
        help="Directory for log files (default: ~/.local/state/shrike/logs)",
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
    args = parser.parse_args()

    log_dir = configure_logging(
        foreground=args.foreground,
        log_dir_override=args.log_dir,
        log_level_override=args.log_level,
    )

    logger.info("Opening collection: %s", args.collection)
    wrapper = CollectionWrapper(args.collection)

    def shutdown(signum: int, frame: Any) -> None:  # noqa: ARG001
        logger.info("Shutting down...")
        wrapper.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    register_tools(mcp, wrapper)

    logger.info("Starting MCP server on %s:%s", args.host, args.port)
    logger.info("Log directory: %s", log_dir)
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
