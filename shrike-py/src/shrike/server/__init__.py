"""The MCP host: the FastMCP app, custom HTTP routes, the per-call validator-cache
shim, and the export download store.

``main`` is re-exported so ``from shrike.server import main`` and the ``//shrike-py/bin``
launchers keep working now that ``server`` is a package; ``python -m shrike.server``
is served by ``__main__`` (which imports from ``shrike.server.server`` directly).

The re-export is LAZY (module ``__getattr__``, PEP 562) rather than an eager
``from shrike.server.server import main``: ``main`` resolves on first access
instead of at package-import time. The eager form binds ``main`` while
``server/__init__.py`` is still executing, which is fragile under the shared-process
test runner (xdist) — if the package object is observed mid-initialization (a
partially-imported ``shrike.server`` in ``sys.modules``), ``shrike.server.main`` is
intermittently absent, and ``mock.patch("shrike.server.main", ...)`` then fails with
``AttributeError: module 'shrike.server' ... does not have the attribute 'main'``.
Resolving on access sidesteps any import-order/partial-module window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # so type checkers + IDEs still see `main` on the package
    from shrike.server.server import main as main

__all__ = ["main"]


def __getattr__(name: str) -> object:
    if name == "main":
        from shrike.server.server import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
