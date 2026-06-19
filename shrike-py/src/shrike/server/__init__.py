"""The MCP host: the FastMCP app, custom HTTP routes, the per-call validator-cache
shim, and the export download store.

``main`` is the package entry point: ``from shrike.server import main`` and the
``//shrike-py/bin`` launchers use it (``python -m shrike.server`` goes through
``__main__``, which imports ``shrike.server.server`` directly). It is a plain eager
function — always present in ``shrike.server.__dict__`` once the package is imported,
with no lazy ``__getattr__`` resolution — that imports the implementation from
``shrike.server.server`` at *call* time. The deferred import keeps the package light:
``shrike.server.server`` pulls in the whole serve stack (the native extension, FastMCP,
the harness, the ``_mcp_perf`` shim), which is loaded only when the server is run.
"""

from __future__ import annotations

__all__ = ["main"]


def main() -> None:
    """Run the MCP server. The implementation lives in ``shrike.server.server`` and is
    imported at call time (never at package import) so this module stays clear of the
    serve stack's import graph."""
    from shrike.server.server import main as _main

    _main()
