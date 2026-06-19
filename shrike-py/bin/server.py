"""Launcher for the Shrike MCP server (`bazel run //shrike-py/bin:server`).

Thin entry point kept out of the `shrike` package so the binary target never
collides with a package subdirectory. It calls `shrike.server.main`
(also reachable as `python -m shrike.server`).
"""

from shrike.server import main

if __name__ == "__main__":
    main()
