"""The export download store: server-named temp packages + their tokens.

The default export delivery (no server-local ``output_path``) writes the
package to a temp file under the cache dir and hands the caller a one-shot
download ``url`` — never base64 (mirroring ``fetch_media``). This module owns
that temp file's lifecycle: mint an unguessable token, map it to the on-disk
temp path, and **reap** the file on download, on a TTL sweep, and at shutdown,
so collection-bearing temp files never leak.

The token is the capability: a ``secrets.token_urlsafe`` value, not a sequential
id, so a guessing client can't enumerate other callers' exports. The
``GET /export/{token}`` route (behind the same ``_guard`` Host/Origin check as
``/media``) **claims** the token here (one-shot), streams the file, then asks
this store to reap it.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger("shrike.export")

# How long a minted export stays downloadable before the TTL sweep reaps it
# (the caller is expected to GET it promptly; this bounds the leak window for a
# client that mints then never downloads).
DEFAULT_TTL_SECONDS = 3600.0

# The temp subdir under the cache dir that holds pending download packages.
EXPORT_SUBDIR = "exports"


@dataclass
class _Pending:
    path: str
    format: str
    created: float
    claimed: bool = False


class ExportStore:
    """Tracks server-named export temp files awaiting download.

    Thread-safe (a plain lock around the small map): the action mints on the
    event loop, the route claims+reaps possibly on a worker thread. The temp
    files live under ``<cache_dir>/exports/``; a token maps to one file.
    """

    def __init__(self, cache_dir: str, *, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._dir = os.path.join(cache_dir, EXPORT_SUBDIR)
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}

    @property
    def dir(self) -> str:
        """The export temp directory (created lazily by :meth:`new_temp_path`)."""
        return self._dir

    def new_temp_path(self, *, suffix: str) -> tuple[str, str]:
        """Mint a (token, temp_path) for a pending export. The path is a
        server-generated name under the export dir (never caller-influenced);
        ``suffix`` is the package extension ('.apkg'/'.colpkg'). The token is
        registered only once the file is written (see :meth:`register`)."""
        os.makedirs(self._dir, exist_ok=True)
        token = secrets.token_urlsafe(24)
        path = os.path.join(self._dir, f"{token}{suffix}")
        return token, path

    def register(self, token: str, path: str, fmt: str) -> None:
        """Record a written export so its token resolves for download. Sweeps
        expired entries first so the map can't grow unbounded under a client
        that mints-but-never-downloads."""
        self._sweep_expired()
        with self._lock:
            self._pending[token] = _Pending(path=path, format=fmt, created=time.monotonic())

    def claim(self, token: str) -> str | None:
        """Atomically consume a token for a one-shot download: returns its on-disk
        path and marks it claimed, so any concurrent or later GET of the same
        token misses (returns None) and the route serves it exactly once. Returns
        None for an unknown, expired, already-claimed, or vanished token.

        The mark — not the reap — is what kills the race: the reap removes the
        file only *after* the body is streamed (a post-response background task),
        but the claim is synchronous, so a fast-follow second GET sees ``claimed``
        and 404s cleanly instead of racing the reap into a 500 on the vanished
        file. A claimed-but-unreaped entry (an aborted stream) stays in the map,
        so the TTL sweep and :meth:`close` still reap its file — it is not leaked."""
        self._sweep_expired()
        with self._lock:
            pending = self._pending.get(token)
            if pending is None or pending.claimed:
                return None
            if not os.path.isfile(pending.path):
                # The file vanished (manual delete / a prior reap) — drop the entry.
                self._pending.pop(token, None)
                return None
            pending.claimed = True
            return pending.path

    def reap(self, token: str) -> None:
        """Remove a token's temp file and entry (one-shot, after download)."""
        with self._lock:
            pending = self._pending.pop(token, None)
        if pending is not None:
            self._remove(pending.path)

    def _sweep_expired(self) -> None:
        now = time.monotonic()
        with self._lock:
            expired = [t for t, p in self._pending.items() if now - p.created > self._ttl]
            paths = [self._pending.pop(t).path for t in expired]
        for path in paths:
            self._remove(path)

    def close(self) -> None:
        """Reap every pending export (shutdown) — no collection-bearing temp
        file is left behind."""
        with self._lock:
            paths = [p.path for p in self._pending.values()]
            self._pending.clear()
        for path in paths:
            self._remove(path)

    @staticmethod
    def _remove(path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("could not reap export temp %s", path, exc_info=True)
