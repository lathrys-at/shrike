"""Local derived-text store — a sidecar SQLite (``shrike.db``) for data Shrike derives from notes.

Its first artifact is an **FTS5 trigram** index over note text, backing fast substring and fuzzy
(typo/partial) lexical search. The store is *source-seamed*: every indexed row is keyed
``(note_id, source, ref)`` — ``source`` is where the text came from (``field`` now; ``ocr``/``asr``
when #199 lands; never VLM image-describe, which stays embedding-only) and ``ref`` is the field name
or media filename. So a match's provenance can say *where* it hit, and new derived sources slot in
without reshaping the store.

It is a **derived cache** like the kernel's vector index: rebuildable from the
collection, ``col_mod`` drift detection, incremental on upsert/delete. It lives
in our cache dir, deliberately **not** as tables in Anki's ``collection.anki2``
— derived/rebuildable data must not ride Anki's sync or trip its schema checks
(see ``docs/decisions.md``). Persistence is inherent to the SQLite file, so
there is no debounced saver (unlike the vector index): writes are transactional
and durable.

Engine split (#281, mirroring the index's #267/#273): the SQL layer lives behind a small engine —
the native ``shrike-derived`` crate (rusqlite), unconditional since the #278 cutover. The facade
keeps the state machine, drift policy, MATCH-expression building, and result filtering. With the
default *bundled*-SQLite build, FTS5+trigram is deterministically available, so the availability
probe below is a formality; a platform-linked build (#300) probes the host library instead, and a
host SQLite without FTS5/trigram makes the store report ``unavailable`` — every lookup then signals
the caller to fall back to the linear ``find_notes`` scan, no feature regression.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shrike.harness.index import IndexState

logger = logging.getLogger("shrike.derived")

# v2 (#228): segments table + recognition meta. Must match DerivedEngine::SCHEMA_VERSION.
SCHEMA_VERSION = 2
MIN_TRIGRAM = 3  # FTS5's trigram tokenizer can't match a term shorter than 3 chars
FUZZY_MIN_SHARED = 2  # a fuzzy candidate must share at least this many query trigrams (noise floor)
SNIPPET_TOKENS = 12  # window size for FTS5 snippet()
DEFAULT_FUZZY_TOP_K = 20

# (note_id, source, ref, txt, snippet) — what an engine MATCH query returns. ``txt`` is only
# filled when asked for (the fuzzy overlap filter needs it; substring doesn't).
EngineRow = tuple[int, str, str, str | None, str | None]


@dataclass(frozen=True)
class LexicalMatch:
    """One lexical hit, with the provenance of *where* in the derived text it matched."""

    note_id: int
    source: str  # "field" | "ocr" | "asr" | …
    ref: str  # field name, or a media filename for a derived source
    snippet: str | None


def _trigrams(text: str) -> list[str]:
    s = text.lower()
    return [s[i : i + 3] for i in range(len(s) - 2)]


def _fts_quote(term: str) -> str:
    """Quote a term as an FTS5 string literal (wrap in double quotes, double internal ones).

    The only safe way to feed arbitrary user text into a MATCH expression — otherwise query
    punctuation is parsed as FTS5 syntax (injection / errors).
    """
    return '"' + term.replace('"', '""') + '"'


class NativeDerivedEngine:
    """The Rust engine (#281): the same surface over rusqlite's *bundled* SQLite.

    A thin marshaling adapter over ``shrike_native.DerivedTextEngine``. Under
    the default build the extension bundles its own SQLite, so FTS5 + trigram
    are deterministically available; a platform-linked build (#300, a cargo-only
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
        """FTS5+trigram availability in the *extension's* SQLite (#300).

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

    def match_rows(self, expr: str, limit: int, *, with_text: bool) -> list[EngineRow]:
        try:
            rows = self._rust.match_rows(expr, int(limit), with_text)
        except self._native_errors as e:
            raise sqlite3.OperationalError(str(e)) from e
        return [(int(n), s, r, t, sn) for n, s, r, t, sn in rows]

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
        # Injectable (the server harness passes it, #278 C5); defaults to the
        # native engine. A *factory*, not an instance: corrupt-file recovery
        # recreates the engine after discarding the file.
        self._engine_factory = engine_factory if engine_factory is not None else NativeDerivedEngine
        # A short-lived lock for the BUILDING claim only — never held during SQLite I/O,
        # so a /reload on the event loop can claim/skip a build without waiting on an in-flight
        # build's data transaction (the engine's internal lock would).
        self._state_lock = threading.Lock()
        self._engine: NativeDerivedEngine | None = None
        self._available = False
        self._state = IndexState.UNAVAILABLE
        self._col_mod: int | None = None
        self._build_thread: threading.Thread | None = None
        # True only while BUILDING was claimed for a build that runs *outside* this store
        # (the kernel's `rebuild_derived` against its own connection — `claim_external_build`).
        # In that window this facade's `_engine` is idle and the rows are already committed, so
        # reads can be served from already-present data; a host-side `build()` keeps it False
        # because the host engine is then mid-transaction under its own mutex (reads must stay
        # gated). Read/written only under `_state_lock`, paired with the `_state` transition.
        self._external_build = False
        self._open()

    # ── lifecycle ────────────────────────────────────────────────────────────────────────────────

    def _make_engine(self) -> NativeDerivedEngine:
        """The FTS5 engine, from the injected factory (native by default —
        unconditional since the #278 cutover)."""
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
        default that's constant True — the #281 win (the probe stops being
        load-bearing) — while a platform-linked build (#300) genuinely probes
        the host library.
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

        Looser than :attr:`available`: it also serves during the **external-build** BUILDING
        window (#650). There the kernel rebuilds against its OWN connection while this facade's
        ``_engine`` is idle and the relevant rows are already committed in ``shrike.db`` — so a
        host read is safe and consistent, and must NOT silently field-fall-back already-present
        recognition rows during the boot/reload rebuild. A *host-side* ``build()`` stays gated
        (``_external_build`` False): the host's own engine is then mid ``DELETE``+``INSERT`` under
        its engine mutex, so a concurrent read would block the event loop on that mutex.

        Writes/drift keep gating on :attr:`available` — this relaxation is read-only.
        """
        if not self._available or self._engine is None:
            return False
        if self._state == IndexState.READY:
            return True
        return self._state == IndexState.BUILDING and self._external_build

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
            # A store that can't even count is broken, not empty (#396) —
            # surface it as the error state instead of a silent ready/0.
            self._state = IndexState.ERROR
            logger.warning("Derived-text store count failed: %s", e)
            return 0

    # ── writes ───────────────────────────────────────────────────────────────────────────────────

    def ingest(self, note_id: int, source: str, refs_text: Mapping[str, str]) -> None:
        """Replace a note's text rows for one ``source`` (incremental upsert).

        ``refs_text`` maps a ``ref`` (field name, or a media filename for a derived source) to its
        text. #98 calls this with ``source="field"``; #199 will call it with ``"ocr"``/``"asr"``.
        """
        if not self._available or self._engine is None:
            return
        self._engine.ingest(note_id, source, refs_text)

    def remove(self, note_ids: list[int], source: str | None = None) -> None:
        """Drop notes' rows (all sources, or just one)."""
        if not self._available or self._engine is None or not note_ids:
            return
        self._engine.remove(note_ids, source)

    def build(self, rows: Iterable[tuple[int, str, str, str]], col_mod: int) -> None:
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
        """Claim BUILDING for a build that runs OUTSIDE this store (#445: the
        kernel's `rebuild_derived` op builds against its own engine on the
        same shrike.db; the rows never enter Python). Same dedupe rule as
        `build_in_background`: a second drift trigger while one is in flight
        is a no-op. Returns False when unavailable or already building."""
        if not self._available:
            return False
        with self._state_lock:
            if self._state == IndexState.BUILDING:
                return False
            self._state = IndexState.BUILDING
            # The build runs on the kernel's own connection; this facade's engine stays idle and
            # the rows are already committed, so reads may be served during this window (#650).
            self._external_build = True
        return True

    def settle_external_build(self, col_mod: int | None) -> None:
        """Record an external build's outcome: READY + the watermark on
        success (``col_mod``), ERROR on ``None``."""
        with self._state_lock:
            if col_mod is None:
                self._state = IndexState.ERROR
            else:
                self._col_mod = col_mod
                self._state = IndexState.READY
            self._external_build = False

    def build_in_background(self, rows: Iterable[tuple[int, str, str, str]], col_mod: int) -> None:
        """Run :meth:`build` on a daemon thread (``rows`` are materialized first — they cross)."""
        if not self._available:
            return
        # Claim BUILDING *before* spawning, so two drift triggers firing close together (boot vs an
        # immediate /reload, a cooperative re-acquire) don't both run a full rebuild — build() flips
        # BUILDING only once its worker runs, too late. The claim uses the short-lived _state_lock
        # (NOT the SQLite-access lock), so a /reload on the event loop never waits here on an
        # in-flight build's transaction.
        with self._state_lock:
            if self._state == IndexState.BUILDING:
                return
            self._state = IndexState.BUILDING
        materialized = list(rows)

        def _release_stuck_claim() -> None:
            # If the claim never reached a terminal state (build() early-returned on a closed store,
            # or the thread never started), don't strand the store in BUILDING — that would refuse
            # every future build and read.
            with self._state_lock:
                if self._state == IndexState.BUILDING:
                    self._state = self._idle_state()

        def _run() -> None:
            try:
                self.build(materialized, col_mod)  # records its own terminal READY/ERROR state
            except Exception:
                logger.debug("background derived build failed", exc_info=True)
            finally:
                _release_stuck_claim()

        try:
            self._build_thread = threading.Thread(target=_run, name="derived-build", daemon=True)
            self._build_thread.start()
        except Exception:
            _release_stuck_claim()  # thread couldn't start — release so a later trigger can retry
            logger.warning("Could not start background derived build", exc_info=True)
            return
        logger.info("Background derived-text build started (%d rows)", len(materialized))

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
            # The MATCH policy is the Rust engine's (single implementation,
            # #331); None = sub-trigram query → find_notes fallback.
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
            # implementation, #331), already deduped + floored.
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
