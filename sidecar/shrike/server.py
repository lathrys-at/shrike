from __future__ import annotations

import argparse
import logging
import signal
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from shrike.collection import CollectionWrapper
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info(f"Opening collection: {args.collection}")
    wrapper = CollectionWrapper(args.collection)

    def shutdown(signum: int, frame: Any) -> None:  # noqa: ARG001
        logger.info("Shutting down...")
        wrapper.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    register_tools(mcp, wrapper)

    logger.info(f"Starting MCP server on {args.host}:{args.port}")
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
