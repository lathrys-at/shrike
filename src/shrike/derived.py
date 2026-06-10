"""Local derived-text store — a sidecar SQLite (``shrike.db``) for data Shrike derives from notes.

Its first artifact is an **FTS5 trigram** index over note text, backing fast substring and fuzzy
(typo/partial) lexical search. The store is *source-seamed*: every indexed row is keyed
``(note_id, source, ref)`` — ``source`` is where the text came from (``field`` now; ``ocr``/``asr``
when #199 lands; never VLM image-describe, which stays embedding-only) and ``ref`` is the field name
or media filename. So a match's provenance can say *where* it hit, and new derived sources slot in
without reshaping the store.

It is a **derived cache** like ``VectorIndex``: rebuildable from the collection, ``col_mod`` drift
detection, incremental on upsert/delete. It lives in our cache dir, deliberately **not** as tables
in Anki's ``collection.anki2`` — derived/rebuildable data must not ride Anki's sync or trip its
schema checks (see ``docs/decisions.md``). Persistence is inherent to the SQLite file, so there is
no debounced saver (unlike the vector index): writes are transactional and durable.

If the runtime's SQLite lacks FTS5 or the trigram tokenizer, the store reports ``unavailable`` and
every lookup signals the caller to fall back to the linear ``find_notes`` scan — current behaviour,
no feature regression.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shrike.index import IndexState

logger = logging.getLogger("shrike.derived")

SCHEMA_VERSION = 1
MIN_TRIGRAM = 3  # FTS5's trigram tokenizer can't match a term shorter than 3 chars
FUZZY_MIN_SHARED = 2  # a fuzzy candidate must share at least this many query trigrams (noise floor)
SNIPPET_TOKENS = 12  # window size for FTS5 snippet()
DEFAULT_FUZZY_TOP_K = 20


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


class DerivedTextStore:
    """FTS5-trigram lexical index over note text in a sidecar ``shrike.db`` (see module doc)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._available = False
        self._state = IndexState.UNAVAILABLE
        self._col_mod: int | None = None
        self._build_thread: threading.Thread | None = None
        self._open()

    # ── lifecycle ────────────────────────────────────────────────────────────────────────────────

    def _open(self) -> None:
        if not self._probe_fts5():
            logger.warning("SQLite FTS5/trigram unavailable; lexical search disabled (find_notes)")
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._available = True
        with self._lock:
            self._ensure_schema()
            self._col_mod = self._get_meta_int("col_mod")
        # Ready only once a build has stamped col_mod; until then callers fall back.
        self._state = IndexState.READY if self._col_mod is not None else IndexState.UNAVAILABLE
        if self._available:
            logger.info("Derived-text store ready at %s (col_mod=%s)", self._path, self._col_mod)

    @staticmethod
    def _probe_fts5() -> bool:
        """Whether this SQLite build has FTS5 with the trigram tokenizer (on a throwaway conn)."""
        try:
            probe = sqlite3.connect(":memory:")
            try:
                probe.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='trigram')")
            finally:
                probe.close()
        except sqlite3.OperationalError:
            return False
        return True

    def _ensure_schema(self) -> None:
        assert self._conn is not None
        c = self._conn
        c.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value)")
        version = self._get_meta_int("schema_version")
        if version is not None and version != SCHEMA_VERSION:
            # No migrations yet: a schema bump drops the derived data; the next drift rebuilds it.
            logger.info("Derived store schema v%s != v%s; resetting", version, SCHEMA_VERSION)
            c.execute("DROP TABLE IF EXISTS idx")
            c.execute("DROP TABLE IF EXISTS rowmap")
            c.execute("DELETE FROM meta WHERE key='col_mod'")
        # idx holds only the searchable text (rowid auto). rowmap carries the (note_id, source, ref)
        # provenance keyed by that rowid, indexed by note_id so incremental delete-by-note is cheap
        # (FTS5 has no secondary indexes, and DELETE-by-column would scan the whole index).
        c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS idx USING fts5(txt, tokenize='trigram')")
        c.execute(
            "CREATE TABLE IF NOT EXISTS rowmap("
            "rowid INTEGER PRIMARY KEY, note_id INTEGER NOT NULL, "
            "source TEXT NOT NULL, ref TEXT NOT NULL)"
        )
        c.execute("CREATE INDEX IF NOT EXISTS rowmap_note ON rowmap(note_id, source)")
        self._set_meta("schema_version", SCHEMA_VERSION)
        c.commit()

    def close(self) -> None:
        if self._conn is not None:
            with self._lock:
                self._conn.close()
                self._conn = None

    # ── meta helpers ─────────────────────────────────────────────────────────────────────────────

    def _get_meta_int(self, key: str) -> int | None:
        assert self._conn is not None
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return int(row[0]) if row is not None and row[0] is not None else None

    def _set_meta(self, key: str, value: Any) -> None:
        assert self._conn is not None
        self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))

    # ── properties ───────────────────────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True when the store can serve lookups (FTS5 present, schema ready, a build has run)."""
        return self._available and self._state == IndexState.READY

    @property
    def state(self) -> IndexState:
        return self._state

    @property
    def col_mod(self) -> int | None:
        return self._col_mod

    @property
    def size(self) -> int:
        if not self._available or self._conn is None:
            return 0
        with self._lock:
            row = self._conn.execute("SELECT count(*) FROM rowmap").fetchone()
        return int(row[0]) if row else 0

    # ── writes ───────────────────────────────────────────────────────────────────────────────────

    def _delete_rows(self, note_ids: Iterable[int], source: str | None) -> None:
        """Delete a note's rows (optionally one source) from idx + rowmap. Caller holds the lock."""
        assert self._conn is not None
        ids = list(note_ids)
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        params: list[Any] = list(ids)
        clause = f"note_id IN ({marks})"
        if source is not None:
            clause += " AND source=?"
            params.append(source)
        rows = self._conn.execute(f"SELECT rowid FROM rowmap WHERE {clause}", params).fetchall()
        rowids = [r[0] for r in rows]
        if not rowids:
            return
        rmarks = ",".join("?" * len(rowids))
        self._conn.execute(f"DELETE FROM idx WHERE rowid IN ({rmarks})", rowids)
        self._conn.execute(f"DELETE FROM rowmap WHERE rowid IN ({rmarks})", rowids)

    def _insert_rows(self, note_id: int, source: str, refs_text: Mapping[str, str]) -> None:
        """Insert one (note_id, source) note's {ref: text} rows. Caller holds the lock."""
        assert self._conn is not None
        for ref, text in refs_text.items():
            if not (text or "").strip():
                continue
            cur = self._conn.execute("INSERT INTO idx(txt) VALUES(?)", (text,))
            self._conn.execute(
                "INSERT INTO rowmap(rowid, note_id, source, ref) VALUES(?,?,?,?)",
                (cur.lastrowid, note_id, source, ref),
            )

    def ingest(self, note_id: int, source: str, refs_text: Mapping[str, str]) -> None:
        """Replace a note's text rows for one ``source`` (incremental upsert).

        ``refs_text`` maps a ``ref`` (field name, or a media filename for a derived source) to its
        text. #98 calls this with ``source="field"``; #199 will call it with ``"ocr"``/``"asr"``.
        """
        if not self._available or self._conn is None:
            return
        with self._lock:
            self._delete_rows([note_id], source)
            self._insert_rows(note_id, source, refs_text)
            self._conn.commit()

    def remove(self, note_ids: list[int], source: str | None = None) -> None:
        """Drop notes' rows (all sources, or just one)."""
        if not self._available or self._conn is None or not note_ids:
            return
        with self._lock:
            self._delete_rows(note_ids, source)
            self._conn.commit()

    def build(self, rows: Iterable[tuple[int, str, str, str]], col_mod: int) -> None:
        """Full (re)build from ``(note_id, source, ref, text)`` rows; stamps ``col_mod``."""
        if not self._available or self._conn is None:
            return
        self._state = IndexState.BUILDING
        try:
            with self._lock:
                self._conn.execute("DELETE FROM idx")
                self._conn.execute("DELETE FROM rowmap")
                for note_id, source, ref, text in rows:
                    if not (text or "").strip():
                        continue
                    cur = self._conn.execute("INSERT INTO idx(txt) VALUES(?)", (text,))
                    self._conn.execute(
                        "INSERT INTO rowmap(rowid, note_id, source, ref) VALUES(?,?,?,?)",
                        (cur.lastrowid, note_id, source, ref),
                    )
                self._set_meta("col_mod", col_mod)
                self._conn.commit()
                self._col_mod = col_mod
            self._state = IndexState.READY
            logger.info("Derived-text store built: %d rows (col_mod=%d)", self.size, col_mod)
        except Exception:
            self._state = IndexState.ERROR
            logger.exception("Derived-text store build failed")
            raise

    def build_in_background(self, rows: Iterable[tuple[int, str, str, str]], col_mod: int) -> None:
        """Run :meth:`build` on a daemon thread (``rows`` are materialized first — they cross)."""
        if not self._available or self._state == IndexState.BUILDING:
            return
        materialized = list(rows)

        def _run() -> None:
            with contextlib.suppress(Exception):  # build() already logs + records ERROR
                self.build(materialized, col_mod)

        self._build_thread = threading.Thread(target=_run, name="derived-build", daemon=True)
        self._build_thread.start()
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
        q = query.strip()
        if not self.available or self._conn is None or len(q) < MIN_TRIGRAM:
            return None
        expr = _fts_quote(q)  # a quoted phrase → contiguous (literal substring) match
        with self._lock:
            try:
                cur = self._conn.execute(
                    "SELECT m.note_id, m.source, m.ref, snippet(idx, 0, '', '', '…', ?) "
                    "FROM idx JOIN rowmap m ON m.rowid = idx.rowid "
                    "WHERE idx MATCH ? ORDER BY rank LIMIT ?",
                    (SNIPPET_TOKENS, expr, limit),
                )
                return [LexicalMatch(int(r[0]), r[1], r[2], r[3]) for r in cur.fetchall()]
            except sqlite3.OperationalError:
                logger.debug("FTS5 substring query failed for %r; falling back", q, exc_info=True)
                return None

    def search_fuzzy(
        self, query: str, top_k: int = DEFAULT_FUZZY_TOP_K
    ) -> list[tuple[int, LexicalMatch]]:
        """Notes sharing trigrams with ``query`` (typo/partial tolerant), best-first.

        Returns ``(note_id, LexicalMatch)`` ranked by FTS5 bm25 over the query's trigrams; a
        candidate must share at least ``FUZZY_MIN_SHARED`` of them (drops one-trigram noise). Empty
        when the store can't serve it (the fuzzy signal is simply absent — graceful).
        """
        grams = _trigrams(query.strip())
        if not self.available or self._conn is None or len(grams) < FUZZY_MIN_SHARED:
            return []
        gram_set = set(grams)
        expr = " OR ".join(_fts_quote(g) for g in gram_set)
        seen: set[int] = set()
        out: list[tuple[int, LexicalMatch]] = []
        with self._lock:
            try:
                cur = self._conn.execute(
                    "SELECT m.note_id, m.source, m.ref, idx.txt, "
                    "snippet(idx, 0, '', '', '…', ?) "
                    "FROM idx JOIN rowmap m ON m.rowid = idx.rowid "
                    "WHERE idx MATCH ? ORDER BY rank LIMIT ?",
                    (SNIPPET_TOKENS, expr, top_k * 4),
                )
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                logger.debug("FTS5 fuzzy query failed for %r", query, exc_info=True)
                return []
        for note_id, source, ref, txt, snippet in rows:
            # Min-overlap floor: FTS5 OR matches ≥1 trigram; require a few shared so a single common
            # gram doesn't surface noise. (bm25 already down-ranks weak matches; this is hygiene.)
            if len(gram_set & set(_trigrams(txt))) < FUZZY_MIN_SHARED:
                continue
            nid = int(note_id)
            if nid in seen:  # dedup to one (best) row per note — rows arrive best-first
                continue
            seen.add(nid)
            out.append((nid, LexicalMatch(nid, source, ref, snippet)))
            if len(out) >= top_k:
                break
        return out

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
