"""Client-side exception hierarchy.

Kept in its own dependency-light module (no ``httpx``, no Pydantic) so the CLI
can catch :class:`ShrikeError` without importing the whole HTTP client — the CLI
loads this to render clean errors, but only pulls ``shrike.client`` (and httpx)
when a command actually talks to the server. ``shrike.client`` re-exports these,
so ``from shrike.client import ShrikeError`` keeps working.
"""

from __future__ import annotations


class ShrikeError(Exception):
    """Base class for all client-raised errors."""


class ServerError(ShrikeError):
    """The server accepted the request but a tool returned an error."""


class CollectionBusyError(ShrikeError):
    """The collection couldn't be acquired — another process holds it.

    Raised under cooperative locking when the server can't re-open the
    collection because something else (typically Anki desktop) has it open. A
    distinct, expected outcome — catch it to retry rather than treating it as a
    generic ``ServerError``.
    """


class ServerUnreachableError(ShrikeError):
    """The server could not be reached (connection refused or timed out)."""


class ServerHTTPError(ShrikeError):
    """The server returned a non-2xx HTTP status."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


class ServerStartError(ShrikeError):
    """Auto-starting the local daemon failed."""
