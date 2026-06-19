"""The MCP host: the FastMCP app, custom HTTP routes, the per-call validator-cache
shim, and the export download store.

``main`` is the package entry point: ``from shrike.server import main`` and the
``//shrike-py/bin`` launchers use it (``python -m shrike.server`` goes through
``__main__``, which imports ``shrike.server.server`` directly). It is a thin wrapper
that imports the implementation from ``shrike.server.server`` at *call* time, not at
package-import time. ``shrike.server.server`` pulls in the whole serve stack — the
native extension, FastMCP, the harness, a sibling ``_mcp_perf`` submodule — so
re-exporting ``main`` at import time raced that graph's partial initialization and
intermittently left the package with no ``main`` attribute (``AttributeError: module
'shrike.server' has no attribute 'main'``). Deferring the import keeps ``main`` an
always-present, patchable attribute and pulls the serve stack in only when the server
is actually run.
"""

from __future__ import annotations

__all__ = ["main"]


def main() -> None:
    """Run the MCP server. The implementation lives in ``shrike.server.server`` and is
    imported at call time (never at package import) so this module stays clear of the
    serve stack's import graph."""
    from shrike.server.server import main as _main

    _main()
