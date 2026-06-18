"""The MCP host: the FastMCP app, custom HTTP routes, the per-call validator-cache
shim, and the export download store.

``main`` is re-exported so ``from shrike.server import main`` and the ``//shrike-py/bin``
launchers keep working now that ``server`` is a package; ``python -m shrike.server``
is served by ``__main__`` (which imports from ``shrike.server.server`` directly).

The re-export is LAZY (module ``__getattr__``, PEP 562): ``main`` resolves on first
access, not at package-import time. It cannot be eager — ``shrike.server.server``
imports a sibling submodule (``shrike.server._mcp_perf``), which runs this package's
``__init__``; an eager ``from shrike.server.server import main`` here would re-enter
``shrike.server.server`` while it is still mid-initialization (``main`` not yet bound)
and raise ``ImportError`` at import time.

The first successful resolve is CACHED into the package namespace (``globals()``), so
``main`` becomes a real attribute thereafter. This closes the intermittent
``AttributeError: module 'shrike.server' ... does not have the attribute 'main'``:
once any access (the ``//bin`` launcher's import, a prior test, the CLI's
``from shrike.server import main``) has resolved ``main``, the ``getattr`` that
``mock.patch("shrike.server.main", ...)`` runs to read the original finds the cached
real attribute and never re-imports a possibly-mid-initialization submodule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # so type checkers + IDEs still see `main` on the package
    from shrike.server.server import main as main

__all__ = ["main"]


def __getattr__(name: str) -> object:
    if name == "main":
        from shrike.server.server import main

        # Cache into the package namespace so a later access — and mock.patch's
        # restore — finds a real attribute instead of re-resolving (which could
        # re-enter a mid-initialization submodule).
        globals()["main"] = main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
