"""Local derived-text store — a sidecar SQLite (``shrike.db``) for data derived from notes.

Its first artifact is an **FTS5 trigram** index over note text, backing fast substring and fuzzy
(typo/partial) lexical search. The store is *source-seamed*: every indexed row is keyed
``(note_id, source, ref)``: ``source`` is the text's origin (``field`` today; ``ocr``/``asr``
later; never VLM image-describe, which stays embedding-only) and ``ref`` is the field name
or media filename. So a match's provenance can say *where* it hit, and new derived sources slot in
without reshaping the store.

It is a **derived cache** like the kernel's vector index: rebuildable from the
collection, ``col_mod`` drift detection, incremental on upsert/delete. It lives
in our cache dir, deliberately **not** as tables in Anki's ``collection.anki2``
— derived/rebuildable data must not ride Anki's sync or trip its schema checks
(see ``docs/dev/decisions.md``). Persistence is inherent to the SQLite file, so
there is no debounced saver (unlike the vector index): writes are transactional
and durable.

Engine split: the SQL layer lives behind a small engine — the native
``shrike-derived`` crate (rusqlite). The facade keeps the state machine, drift
policy, MATCH-expression building, and result filtering. With the default
*bundled*-SQLite build, FTS5+trigram is deterministically available, so the
availability probe below is a formality; a platform-linked build probes the host
library instead, and a host SQLite without FTS5/trigram makes the store report
``unavailable`` — every lookup then signals the caller to fall back to the linear
``find_notes`` scan, no feature regression.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shrike.harness.index import IndexState

logger = logging.getLogger("shrike.derived")

# v2: segments table + recognition meta. Must match DerivedEngine::SCHEMA_VERSION.
SCHEMA_VERSION = 2
FUZZY_MIN_SHARED = 2  # a fuzzy candidate must share at least this many query trigrams (noise floor)
DEFAULT_FUZZY_TOP_K = 20


@dataclass(frozen=True)
class LexicalMatch:
    """One lexical hit, with the provenance of *where* in the derived text it matched."""

    note_id: int
    source: str  # "field" | "ocr" | "asr" | …
    ref: str  # field name, or a media filename for a derived source
    snippet: str | None


class NativeDerivedEngine:
    """The Rust engine: the same surface over rusqlite's *bundled* SQLite.

    A thin marshaling adapter over ``shrike_native.DerivedTextEngine``. Under
    the default build the extension bundles its own SQLite, so FTS5 + trigram
    are deterministically available; a platform-linked build (a cargo-only
    ``--no-default-features`` build) relies on the host library, so :meth:`probe`
    genuinely probes either way (trivially true when bundled). Native errors
    are translated to ``sqlite3.Error`` so the facade's recovery/fallback logic
    is engine-agnostic.
    """

    def __init__(self, path: Path) -> None:
        import shrike_native

        self._native_errors = (
            shrike_native.NativeInputError,
            shrike_native.NativeUnavailableError,
            shrike_native.NativeInternalError,
        )
        try:
            self._rust = shrike_native.DerivedTextEngine(str(path), SCHEMA_VERSION)
        except self._native_errors as e:
            raise sqlite3.DatabaseError(str(e)) from e

    @staticmethod
    def probe() -> bool:
        """FTS5+trigram availability in the *extension's* SQLite.

        Trivially true under the bundled default; load-bearing when the
        extension was built against a platform SQLite.
        """
        import shrike_native

        return bool(shrike_native.derived_fts5_probe())

    def close(self) -> None:
        self._rust.close()

    def get_col_mod(self) -> int | None:
        # Explicit conversion — see NativeIndexEngine.ndim (no-any-return on CI).
        value = self._rust.get_col_mod()
        return None if value is None else int(value)

    def set_col_mod(self, value: int) -> None:
        self._rust.set_col_mod(int(value))

    def count(self) -> int:
        try:
            return int(self._rust.count())
        except self._native_errors as e:
            raise sqlite3.DatabaseError(str(e)) from e

    def ingest(self, note_id: int, source: str, refs_text: Mapping[str, str]) -> None:
        self._rust.ingest(int(note_id), source, list(refs_text.items()))

    def remove(self, note_ids: list[int], source: str | None = None) -> None:
        self._rust.remove([int(n) for n in note_ids], source)

    def build(self, rows: Iterable[tuple[int, str, str, str]], col_mod: int) -> None:
        try:
            self._rust.build(list(rows), int(col_mod))
        except self._native_errors as e:
            raise sqlite3.DatabaseError(str(e)) from e

    def search_substring(
        self, query: str, limit: int
    ) -> list[tuple[int, str, str, str | None]] | None:
        try:
            rows = self._rust.search_substring(query, int(limit))
        except self._native_errors as e:
            raise sqlite3.OperationalError(str(e)) from e
        return None if rows is None else [(int(n), s, r, sn) for n, s, r, sn in rows]

    def search_fuzzy(self, query: str, top_k: int) -> list[tuple[int, str, str, str | None]]:
        try:
            rows = self._rust.search_fuzzy(query, int(top_k))
        except self._native_errors as e:
            raise sqlite3.OperationalError(str(e)) from e
        return [(int(n), s, r, sn) for n, s, r, sn in rows]


class DerivedTextStore:
    """FTS5-trigram lexical index over note text in a sidecar ``shrike.db`` (see module doc)."""

    def __init__(
        self,
        path: str | Path,
        *,
        engine_factory: Callable[[Path], NativeDerivedEngine] | None = None,
    ) -> None:
        self._path = Path(path)
        # Injectable (the server harness passes it); defaults to the native
        # engine. A *factory*, not an instance: corrupt-file recovery recreates
        # the engine after discarding the file.
        self._engine_factory = engine_factory if engine_factory is not None else NativeDerivedEngine
        self._engine: NativeDerivedEngine | None = None
        self._available = False
        self._state = IndexState.UNAVAILABLE
        self._col_mod: int | None = None
        self._open()

    # ── lifecycle ────────────────────────────────────────────────────────────────────────────────

    def _make_engine(self) -> NativeDerivedEngine:
        """The FTS5 engine, from the injected factory (native by default)."""
        return self._engine_factory(self._path)

    def _open(self) -> None:
        if not self._probe_fts5():
            logger.warning("SQLite FTS5/trigram unavailable; lexical search disabled (find_notes)")
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # The sidecar is a throwaway derived cache, so a corrupt/unreadable file must never be fatal
        # (a hard kill mid-checkpoint, disk-full, or a non-DB file at the path). On any open/schema
        # error: drop the file (+ its WAL sidecars) and try once more from scratch; if that still
        # fails, degrade to unavailable so lookups fall back to find_notes rather than aborting the
        # daemon. (This is the corruption case the FTS5-missing branch above doesn't cover.)
        try:
            self._connect_and_init()
        except sqlite3.Error:
            logger.warning("Derived store at %s unreadable; recreating", self._path, exc_info=True)
            self._discard_file()
            try:
                self._connect_and_init()
            except sqlite3.Error:
                logger.warning(
                    "Derived store could not be initialized; lexical search disabled",
                    exc_info=True,
                )
                self._discard_file()
                return
        # Ready only once a build has stamped col_mod; until then callers fall back.
        self._state = self._idle_state()
        logger.info("Derived-text store ready at %s (col_mod=%s)", self._path, self._col_mod)

    def _idle_state(self) -> IndexState:
        """The non-building resting state: READY once a build stamped col_mod, else UNAVAILABLE."""
        return IndexState.READY if self._col_mod is not None else IndexState.UNAVAILABLE

    def _connect_and_init(self) -> None:
        """(Re)open the engine and ensure the schema. Raises ``sqlite3.Error`` on a bad file."""
        self._engine = self._make_engine()
        self._col_mod = self._engine.get_col_mod()
        self._available = True

    def _discard_file(self) -> None:
        """Close the engine and delete the sidecar (+ WAL/SHM) — recovery from corruption."""
        if self._engine is not None:
            with contextlib.suppress(Exception):
                self._engine.close()
            self._engine = None
        self._available = False
        for suffix in ("", "-wal", "-shm"):
            with contextlib.suppress(OSError):
                Path(str(self._path) + suffix).unlink()

    @staticmethod
    def _probe_fts5() -> bool:
        """Whether the selected engine's SQLite has FTS5 with the trigram tokenizer.

        The probe asks the extension's linked SQLite: under the bundled
        default that's constant True, while a platform-linked build genuinely
        probes the host library.
        """
        return NativeDerivedEngine.probe()

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    # ── properties ───────────────────────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True when the store can serve lookups (FTS5 present, schema ready, a build has run)."""
        return self._available and self._state == IndexState.READY

    def _can_serve_reads(self) -> bool:
        """Whether a *read* (substring/fuzzy) may run against the engine right now.

        Serving in both READY and BUILDING is safe because the kernel's rebuild is
        ATOMIC: it builds the new FTS5 index into a shadow off the live tables and swaps it over
        them in ONE transaction, so any reader sees either the complete OLD index or the complete
        NEW one — never a torn/partial state mid-rebuild. (The justification is the atomic swap,
        not the journal mode.) Production lexical reads run through the kernel's own derived store
        (a WAL read pool, serialized against the swap by SQLite); this facade's separate connection
        (the ``/status`` surface) likewise sees only committed state. So a BUILDING read must not
        silently field-fall-back already-present rows — serve it.
        """
        if not self._available or self._engine is None:
            return False
        return self._state in (IndexState.READY, IndexState.BUILDING)

    @property
    def state(self) -> IndexState:
        return self._state

    @property
    def col_mod(self) -> int | None:
        return self._col_mod

    @col_mod.setter
    def col_mod(self, value: int) -> None:
        """Advance the drift watermark in place, without a rebuild.

        Called after an incremental ``ingest``/``remove`` keeps the store current, or after a
        vectors-/text-unchanged metadata edit (tag/deck rename) that only bumped ``col.mod`` — so
        the next boot's :meth:`check_drift` sees no drift and skips a needless full rebuild. A no-op
        unless FTS5 is present and the store is open (callers gate on :attr:`available`).
        """
        if not self._available or self._engine is None:
            return
        self._engine.set_col_mod(value)
        self._col_mod = value

    @property
    def size(self) -> int:
        # Skip the count (and the engine lock) while a build holds it for its whole transaction —
        # `build()` keeps `_state == BUILDING` across its DELETE+N×INSERT+commit, so a concurrent
        # `status()`/`GET /status` must not block the event loop waiting on it.
        if not self._available or self._engine is None or self._state == IndexState.BUILDING:
            return 0
        try:
            return self._engine.count()
        except sqlite3.Error as e:
            # A store that can't even count is broken, not empty —
            # surface it as the error state instead of a silent ready/0.
            self._state = IndexState.ERROR
            logger.warning("Derived-text store count failed: %s", e)
            return 0

    # ── writes (PRIVATE: the low-level engine seam) ────────────────────────────
    # The KERNEL is the sole production writer of shrike.db — production never
    # writes through this facade (it drives the kernel's `rebuild_derived` + the
    # per-op ingest actor), so these are underscore-private: a non-test caller is
    # a dual-writer/SQLITE_BUSY bug, not a supported path. Tests use them to stage
    # fixture state for the read-path assertions.

    def _ingest(self, note_id: int, source: str, refs_text: Mapping[str, str]) -> None:
        """Replace a note's text rows for one ``source`` (incremental upsert).

        ``refs_text`` maps a ``ref`` (field name, or a media filename for a derived source) to its
        text. Called with ``source="field"`` today; ``"ocr"``/``"asr"`` later.
        """
        if not self._available or self._engine is None:
            return
        self._engine.ingest(note_id, source, refs_text)

    def _remove(self, note_ids: list[int], source: str | None = None) -> None:
        """Drop notes' rows (all sources, or just one)."""
        if not self._available or self._engine is None or not note_ids:
            return
        self._engine.remove(note_ids, source)

    def _build(self, rows: Iterable[tuple[int, str, str, str]], col_mod: int) -> None:
        """Full (re)build from ``(note_id, source, ref, text)`` rows; stamps ``col_mod``."""
        if not self._available or self._engine is None:
            return
        self._state = IndexState.BUILDING
        started = time.perf_counter()
        try:
            self._engine.build(rows, col_mod)
            self._col_mod = col_mod
            self._state = IndexState.READY
            logger.info(
                "Derived-text store built: %d rows (col_mod=%d, %.1fs)",
                self.size,
                col_mod,
                time.perf_counter() - started,
            )
        except Exception:
            self._state = IndexState.ERROR
            logger.exception("Derived-text store build failed")
            raise

    def claim_external_build(self) -> bool:
        """Mark the store BUILDING for the rebuild that runs in the KERNEL (the
        sole writer of ``shrike.db``; the rows never enter Python). Dedupe: a
        second drift trigger while one is in flight is a no-op. Every transition
        runs on the event loop, so no lock is needed (the host-side build thread
        that needed one is gone). Returns False when unavailable or already
        building."""
        if not self._available or self._state == IndexState.BUILDING:
            return False
        self._state = IndexState.BUILDING
        return True

    def settle_external_build(self, col_mod: int | None) -> None:
        """Record the kernel rebuild's outcome: READY + the watermark on
        success (``col_mod``), ERROR on ``None``."""
        if col_mod is None:
            self._state = IndexState.ERROR
        else:
            self._col_mod = col_mod
            self._state = IndexState.READY

    def check_drift(self, current_col_mod: int) -> bool:
        """True when the store is stale (never built, or ``col_mod`` moved) → needs a rebuild."""
        if not self._available:
            return False  # nothing to rebuild; lookups fall back regardless
        return self._col_mod is None or self._col_mod != current_col_mod

    # ── reads ────────────────────────────────────────────────────────────────────────────────────

    def search_substring(self, query: str, limit: int = 50) -> list[LexicalMatch] | None:
        """Notes whose derived text literally contains ``query`` (case-insensitive).

        Returns matches with their ``(source, ref, snippet)`` provenance, or ``None`` to tell the
        caller to use the ``find_notes`` fallback — when the store is unavailable/not-ready or the
        query is shorter than a trigram (FTS5 trigram can't match < 3 chars).
        """
        if not self._can_serve_reads() or self._engine is None:
            return None
        try:
            # The MATCH policy is the Rust engine's (single implementation);
            # None = sub-trigram query → find_notes fallback.
            rows = self._engine.search_substring(query, limit)
        except sqlite3.Error:
            logger.debug("FTS5 substring query failed for %r; falling back", query, exc_info=True)
            return None
        if rows is None:
            return None
        return [LexicalMatch(nid, source, ref, snippet) for nid, source, ref, snippet in rows]

    def search_fuzzy(
        self, query: str, top_k: int = DEFAULT_FUZZY_TOP_K
    ) -> list[tuple[int, LexicalMatch]]:
        """Notes sharing trigrams with ``query`` (typo/partial tolerant), best-first.

        Returns ``(note_id, LexicalMatch)`` ranked by FTS5 bm25 over the query's trigrams; a
        candidate must share at least ``FUZZY_MIN_SHARED`` of them (drops one-trigram noise). Empty
        when the store can't serve it (the fuzzy signal is simply absent — graceful).
        """
        if not self._can_serve_reads() or self._engine is None:
            return []
        try:
            # The trigram/overlap policy is the Rust engine's (single
            # implementation), already deduped + floored.
            rows = self._engine.search_fuzzy(query, top_k)
        except sqlite3.Error:
            logger.debug("FTS5 fuzzy query failed for %r", query, exc_info=True)
            return []
        return [
            (int(nid), LexicalMatch(int(nid), source, ref, snippet))
            for nid, source, ref, snippet in rows
        ]

    def status(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "state": self._state.value,
            "available": self.available,
            "fts5": self._available,
            "size": self.size,
            "path": str(self._path),
        }
        if self._col_mod is not None:
            info["col_mod"] = self._col_mod
        return info
