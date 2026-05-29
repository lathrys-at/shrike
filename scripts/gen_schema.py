#!/usr/bin/env python3
"""Generate ``docs/mcp-schema.json`` from the MCP tool definitions.

The schema is derived from the FastMCP app built by ``register_tools`` — input
schemas come from the tool signatures, output schemas from the Pydantic response
models in :mod:`shrike.schemas`. This script is the single source of truth for
the documented schema; the committed ``docs/mcp-schema.json`` must match its
output, which ``tests/unit/test_schema_doc.py`` and CI enforce.

Usage::

    python scripts/gen_schema.py           # write docs/mcp-schema.json
    python scripts/gen_schema.py --check    # exit 1 if the committed file is stale
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, cast

from mcp.server.fastmcp import FastMCP

from shrike.collection import CollectionWrapper
from shrike.server import create_mcp
from shrike.tools import register_tools

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "docs" / "mcp-schema.json"


async def _list_tools(mcp: FastMCP) -> list[dict[str, Any]]:
    tools = await mcp.list_tools()
    return [
        {
            "name": t.name,
            "description": t.description,
            "inputSchema": t.inputSchema,
            "outputSchema": t.outputSchema,
        }
        for t in tools
    ]


def generate() -> str:
    """Build the MCP app and serialize its tool schemas to JSON text.

    ``register_tools`` only captures the wrapper/index in tool closures (it never
    calls them during registration), so a stand-in is sufficient here.
    """
    mcp = create_mcp()
    register_tools(mcp, cast(CollectionWrapper, object()), index=None)
    tools = asyncio.run(_list_tools(mcp))
    return json.dumps(tools, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate docs/mcp-schema.json")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed file matches; exit 1 if stale.",
    )
    args = parser.parse_args()

    generated = generate()
    if args.check:
        current = SCHEMA_PATH.read_text(encoding="utf-8") if SCHEMA_PATH.exists() else ""
        if current != generated:
            print(
                "docs/mcp-schema.json is out of date. Regenerate it with:\n"
                "    python scripts/gen_schema.py",
                file=sys.stderr,
            )
            return 1
        print("docs/mcp-schema.json is up to date.")
        return 0

    SCHEMA_PATH.write_text(generated, encoding="utf-8")
    print(f"Wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
