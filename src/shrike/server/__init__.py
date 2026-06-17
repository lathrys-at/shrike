"""The MCP host: the FastMCP app, custom HTTP routes, the per-call validator-cache
shim, and the export download store.

``main`` is re-exported so ``from shrike.server import main`` and the ``//bin``
launchers keep working now that ``server`` is a package; ``python -m shrike.server``
is served by ``__main__``.
"""

from shrike.server.server import main

__all__ = ["main"]
