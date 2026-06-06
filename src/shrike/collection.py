from __future__ import annotations

import asyncio
import atexit
import base64
import contextlib
import fnmatch
import ipaddress
import logging
import mimetypes
import os
import re
import socket
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar
from urllib.parse import urlparse

from anki.collection import Collection
from anki.consts import MODEL_CLOZE
from anki.errors import DBError, NotFoundError
from anki.notes import NoteFieldsCheckResult

from shrike.embed_text import field_is_blank, normalize_for_embedding
from shrike.schemas import COLLECTION_BUSY_CODE


class CollectionBusyError(Exception):
    """A cooperative-mode re-acquire failed: another process holds the collection.

    Raised when re-opening the collection (after an idle release, #64) hits Anki's
    ``DBError`` lock — typically because Anki desktop is open. Expected, not a bug;
    the tool layer surfaces it over MCP with ``COLLECTION_BUSY_CODE`` and the
    client maps it to its own ``CollectionBusyError``. The message is prefixed
    with the code so the client can detect it without parsing prose.
    """

    def __init__(self, detail: str = "") -> None:
        human = detail or (
            "The collection is in use by another process (is Anki open?). Close it and try again."
        )
        super().__init__(f"{COLLECTION_BUSY_CODE}: {human}")


logger = logging.getLogger("shrike.collection")

OnDuplicate = Literal["error", "skip", "allow"]

# -- media (#70) -------------------------------------------------------------
# Default cap on how large a media file fetch_media will inline as base64 (above
# this the caller reads the returned path instead). The hard ceiling on a single
# stored / downloaded file. The fetch timeout for server-side URL stores.
DEFAULT_MAX_INLINE_BYTES = 8 * 1024 * 1024
MEDIA_MAX_BYTES = 64 * 1024 * 1024
URL_FETCH_TIMEOUT = 30.0


def _guess_mime(filename: str) -> str | None:
    """Best-effort MIME from a filename's extension (None for unknown)."""
    return mimetypes.guess_type(filename)[0]


def _safe_media_name(name: str) -> str:
    """Reduce a caller-supplied name to a bare basename inside the media dir.

    Strips any directory components, so ``../../etc/passwd`` becomes ``passwd`` and
    can only ever resolve inside ``col.media.dir()`` — the path-traversal guard for
    fetch/delete. Returns "" for a name that is only separators/dots.
    """
    base = os.path.basename(name.replace("\\", "/").rstrip("/"))
    return "" if base in ("", ".", "..") else base


def _check_public_address(host: str) -> None:
    """Raise ValueError if any address ``host`` resolves to is non-public.

    SSRF guard for server-side URL fetches: rejects loopback, RFC1918, link-local,
    reserved, multicast, and unspecified ranges (so a steered fetch can't reach the
    cloud metadata endpoint or an internal service). Note the resolve-then-connect
    gap is a known TOCTOU (DNS rebinding / redirects aren't re-validated); pinning
    the connection to the checked IP is a follow-up.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"could not resolve host '{host}': {e}") from e
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
        ):
            raise ValueError(f"refusing to fetch from non-public address {addr} (host '{host}')")


def _fetch_media_url(
    url: str,
    *,
    allow_private: bool,
    max_bytes: int = MEDIA_MAX_BYTES,
    timeout: float = URL_FETCH_TIMEOUT,
) -> tuple[bytes, str | None]:
    """Download ``url`` into memory, returning (bytes, content_type).

    Off the worker thread (network I/O): callers run it via ``asyncio.to_thread``
    so a 30s fetch never blocks collection ops. Scheme is restricted to http/https,
    the host is SSRF-checked unless ``allow_private``, the body is capped at
    ``max_bytes``, and httpx honors proxy env vars (``trust_env``; SOCKS needs the
    optional ``httpx[socks]`` extra).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme: {parsed.scheme or '(none)'}")
    if not parsed.hostname:
        raise ValueError("URL has no host")
    if not allow_private:
        _check_public_address(parsed.hostname)

    import httpx

    chunks: list[bytes] = []
    total = 0
    with (
        httpx.Client(follow_redirects=True, timeout=timeout, trust_env=True) as client,
        client.stream("GET", url) as resp,
    ):
        resp.raise_for_status()
        for chunk in resp.iter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"download exceeds the {max_bytes}-byte limit")
            chunks.append(chunk)
        content_type = (resp.headers.get("content-type") or "").split(";")[0].strip() or None
    return b"".join(chunks), content_type


# Anki's note.fields_check() failures that are *not* duplicates — these are
# structural problems with the note itself and are always rejected (no policy
# knob), mapped to a machine reason + human message. DUPLICATE is handled
# separately because it's the one fields_check result a caller may legitimately
# want to allow.
_STRUCTURAL_PROBLEMS: dict[NoteFieldsCheckResult.V, tuple[str, str]] = {
    NoteFieldsCheckResult.EMPTY: ("empty", "The first field is empty."),
    NoteFieldsCheckResult.MISSING_CLOZE: (
        "missing_cloze",
        "No cloze deletions ({{c1::...}}) were found in the cloze field.",
    ),
    NoteFieldsCheckResult.NOTETYPE_NOT_CLOZE: (
        "notetype_not_cloze",
        "Cloze syntax was used but the note type is not a cloze type.",
    ),
    NoteFieldsCheckResult.FIELD_NOT_CLOZE: (
        "field_not_cloze",
        "A cloze deletion is in a field that is not the cloze field.",
    ),
}
_DUPLICATE_MESSAGE = "The first field duplicates an existing note of this type."

T = TypeVar("T")

# Default seconds to hold the collection open after the last operation before
# releasing the lock, in cooperative mode (#64). Near SQLite's conventional
# ``busy_timeout``; short because holding blocks launching Anki and re-opening is
# a cheap local SQLite open.
DEFAULT_LOCK_HOLD = 5.0

# Chars that are special inside an Anki search even within double quotes; escaped
# so the term is matched literally (verified against anki 25.9.4: ':' must be
# escaped even quoted, or a substring spanning it silently misses).
_ANKI_SEARCH_SPECIALS = ("\\", '"', "*", "_", ":")


def _escape_anki_text(text: str) -> str:
    for ch in _ANKI_SEARCH_SPECIALS:
        text = text.replace(ch, "\\" + ch)
    return text


def apply_replacement(
    value: str, search: str, replacement: str, *, regex: bool, match_case: bool
) -> str:
    """Render search→replace on a single field value, for preview only.

    Mirrors Anki's ``find_and_replace`` closely enough to preview literal edits
    *exactly* (the apply path runs Anki itself). Regex previews use Python ``re``
    and are illustrative — capture-group replacements differ (Anki uses ``$1``,
    Python ``\\1``), so the apply is authoritative for those.
    """
    if regex:
        flags = 0 if match_case else re.IGNORECASE
        return re.sub(search, replacement, value, flags=flags)
    if match_case:
        return value.replace(search, replacement)
    # Case-insensitive literal: match-escaped pattern, literal replacement (a
    # function repl avoids backref/template interpretation of the replacement).
    return re.sub(re.escape(search), lambda _m: replacement, value, flags=re.IGNORECASE)


def substring_info(content: dict[str, str] | None, text: str) -> dict[str, Any] | None:
    """Locate a case-insensitive literal substring in a note's fields.

    Returns ``{"matched_fields": [...], "snippet": ...}`` (snippet = ≈80-char
    context around the first occurrence in the first matched field), or ``None``
    when no field contains the text. The authority for exact-match evidence — the
    Anki query is only a fast pre-filter, so this drops any false positives.
    """
    needle = text.lower()
    matched: list[str] = []
    snippet: str | None = None
    for name, value in (content or {}).items():
        v = value or ""
        idx = v.lower().find(needle)
        if idx == -1:
            continue
        matched.append(name)
        if snippet is None:
            start = max(0, idx - 30)
            end = idx + len(text) + 30
            frag = v[start:end]
            if start > 0:
                frag = "…" + frag
            if end < len(v):
                frag = frag + "…"
            snippet = frag
    if not matched:
        return None
    return {"matched_fields": matched, "snippet": snippet}


class CollectionWrapper:
    """Serializes every access to the underlying ``anki.Collection``.

    The Anki collection is not safe for concurrent use, and its SQLite-backed
    state is happiest when touched from a single thread. This wrapper owns one
    dedicated worker thread and funnels all collection operations through it,
    so concurrent callers (event-loop tasks, custom HTTP routes, the startup
    path) are serialized rather than racing.

    Operations are exposed as ``async`` methods: awaiting one schedules the
    work on the worker thread and yields control, keeping the event loop
    responsive while the collection is busy. ``run_sync`` and ``close`` provide
    synchronous entry points for the startup and shutdown paths where no event
    loop is running.

    **Cooperative locking (#64, opt-in).** By default the collection is opened
    once and held for the daemon's lifetime (Anki's exclusive lock too). In
    cooperative mode the collection is still opened at boot, but released after a
    short idle window and re-opened on demand, so an *idle* daemon doesn't block
    launching Anki. ``self._open`` tracks held-vs-released; every op routes
    through ``_locked`` (re-open if released, run an acquire drift hook), and each
    async op (re)arms an idle-release timer. In the default mode ``self._open`` is
    always True and these paths are inert.
    """

    # Assigned on the worker thread in ``_open``; declared here for the type
    # checker. Never mutate or call methods on it outside the worker thread.
    # In cooperative mode it may be a *closed* handle while released; ``_locked``
    # re-opens before any access, so it is never dereferenced while closed.
    col: Collection

    def __init__(
        self,
        path: str,
        *,
        cooperative: bool = False,
        hold_seconds: float = DEFAULT_LOCK_HOLD,
        on_acquire: Callable[[Collection], None] | None = None,
    ) -> None:
        self._path = path
        self._closed = False
        self._cooperative = cooperative
        self._hold = hold_seconds
        self._on_acquire = on_acquire
        self._open_flag = False
        self._release_handle: asyncio.TimerHandle | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shrike-collection")
        # Open on the worker thread so the backend is owned by the same thread
        # that will service every subsequent operation. (Cooperative mode opens
        # at boot too; the idle-release lifecycle takes over afterwards.)
        self._executor.submit(self._open).result()
        atexit.register(self.close)

    def _open(self) -> None:
        logger.debug("Opening collection at %s", self._path)
        self.col = Collection(self._path)
        self._open_flag = True
        logger.debug("Collection opened successfully")

    def set_acquire_hook(self, hook: Callable[[Collection], None] | None) -> None:
        """Set the callback run (on the worker thread) after a fresh re-open.

        Set post-construction because the hook closes over the index/embedding
        runtime, which are built after the wrapper. Used in cooperative mode to
        re-check index drift on every re-acquire.
        """
        self._on_acquire = hook

    @property
    def cooperative(self) -> bool:
        return self._cooperative

    @property
    def is_open(self) -> bool:
        """Whether the collection is currently held open (vs idle-released)."""
        return self._open_flag

    # -- execution primitives ------------------------------------------------

    def _locked(self, fn: Callable[[Collection], T]) -> T:
        """Run ``fn(col)`` on the worker thread, re-opening first if released.

        The single place that touches ``self.col``: if cooperative mode released
        the collection, re-open it and run the acquire hook (drift re-check)
        before the op. In the default mode ``self._open_flag`` is always True, so
        this is just ``fn(self.col)``.
        """
        if not self._open_flag:
            try:
                self.col = Collection(self._path)
            except DBError as e:
                # The file opened fine at boot, so a DBError on re-acquire is
                # overwhelmingly lock contention (another process — usually Anki
                # desktop — holds it), not corruption. Surface it as busy.
                raise CollectionBusyError() from e
            self._open_flag = True
            logger.debug("Re-acquired collection at %s", self._path)
            if self._on_acquire is not None:
                with contextlib.suppress(Exception):
                    self._on_acquire(self.col)
        return fn(self.col)

    async def run(self, fn: Callable[[Collection], T]) -> T:
        """Run ``fn(col)`` on the worker thread and await the result.

        The escape hatch for collection operations that don't have a dedicated
        method here (e.g. note-type edits, reading ``col.mod``). All access is
        serialized through the same single worker thread, which re-opens the
        collection first if cooperative mode released it (``_locked``). In
        cooperative mode the idle-release timer is (re)armed after the op.
        """
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, lambda: self._locked(fn))
        finally:
            if self._cooperative and not self._closed:
                self._schedule_release(loop)

    def run_sync(self, fn: Callable[[Collection], T]) -> T:
        """Run ``fn(col)`` on the worker thread, blocking until it returns.

        For the startup and shutdown paths, which run before/after the event
        loop and so cannot ``await``. Still routes through the worker thread
        (re-opening first if released), preserving single-thread ownership. No
        idle-release is armed here — there is no loop; the next async op or an
        explicit ``release_now`` handles release.
        """
        return self._executor.submit(lambda: self._locked(fn)).result()

    # -- cooperative idle-release (#64) --------------------------------------

    def _schedule_release(self, loop: asyncio.AbstractEventLoop) -> None:
        """(Re)arm the idle-release timer; mirrors IndexSaver's debounce."""
        if self._release_handle is not None:
            self._release_handle.cancel()
        self._release_handle = loop.call_later(self._hold, self._fire_release)

    def _fire_release(self) -> None:
        self._release_handle = None
        # Close on the worker thread; fire-and-forget so the loop never blocks.
        # A re-acquire racing this is safe — _release no-ops if already re-opened
        # and _locked re-opens before any access.
        self._executor.submit(self._release)

    def _release(self) -> None:
        if not self._open_flag or self._closed:
            return
        with contextlib.suppress(Exception):
            self.col.close()
        self._open_flag = False
        logger.debug("Released collection lock after idle")

    def release_now(self) -> None:
        """Synchronously release the collection (close + mark released).

        For the boot path (release after the initial drift check so a
        never-touched idle daemon doesn't hold the lock) and tests.
        """
        self._executor.submit(self._release).result()

    async def reopen(self) -> None:
        """Close and re-open the collection on the worker thread.

        Releases Anki's SQLite lock and re-opens the file, picking up changes made
        on disk underneath the daemon (a restored backup, a file-level sync/swap).
        Runs on the single worker thread, so it's serialized against every other
        operation — nothing sees a half-swapped handle. The index is not touched
        here; the caller re-checks drift afterwards. This is the primitive
        cooperative locking (#64) will reuse for its open-on-demand lifecycle.
        """
        await self.run(self._do_reopen)

    def _do_reopen(self, _c: Collection) -> None:
        with contextlib.suppress(Exception):
            self.col.close()
        self.col = Collection(self._path)
        self._open_flag = True
        logger.info("Reopened collection at %s", self._path)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._release_handle is not None:
            self._release_handle.cancel()
            self._release_handle = None
        logger.debug("Closing collection")
        with contextlib.suppress(Exception):
            self._executor.submit(lambda _c: self.col.close(), None).result()
        self._executor.shutdown(wait=True)

    # -- collection info -----------------------------------------------------

    async def get_collection_info(
        self,
        include: list[str] | None = None,
        note_type_details: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self.run(lambda _c: self._get_collection_info(include, note_type_details))

    def _get_collection_info(
        self,
        include: list[str] | None,
        note_type_details: list[str] | None,
    ) -> dict[str, Any]:
        ALL_SECTIONS = ["summary", "note_types", "decks", "tags", "stats"]
        sections = ALL_SECTIONS if include and "all" in include else (include or ["summary"])
        detail_names = set(note_type_details or [])
        result: dict[str, Any] = {}

        # Compute the two expensive intermediates shared across sections at most
        # once: the scheduler due tree (summary + stats) and the per-deck note
        # counts (decks + stats — a full card-table scan). collection_info("all")
        # would otherwise build each twice.
        tree = self.col.sched.deck_due_tree() if {"summary", "stats"} & set(sections) else None
        counts = self._note_counts_by_deck() if {"decks", "stats"} & set(sections) else None

        if "summary" in sections:
            assert tree is not None
            result["summary"] = self._get_summary(tree)

        if "note_types" in sections:
            result["note_types"] = self._get_note_types(detail_names)

        if "decks" in sections:
            assert counts is not None
            result["decks"] = self._get_decks(counts)

        if "tags" in sections:
            result["tags"] = self.col.tags.all()

        if "stats" in sections:
            assert tree is not None and counts is not None
            result["stats"] = self._get_stats(tree, counts)

        return result

    def _get_summary(self, tree: Any) -> dict[str, Any]:
        # Top-level node counts are already rolled up to include their
        # subdecks, so the collection total is the sum over top-level decks
        # only. Recursing into children would double-count nested decks.
        total_due = sum(top.review_count + top.learn_count for top in tree.children)

        created = datetime.fromtimestamp(self.col.crt, tz=UTC).strftime("%Y-%m-%d")
        modified = datetime.fromtimestamp(self.col.mod / 1000, tz=UTC).isoformat(timespec="seconds")

        return {
            "path": self.col.path,
            "created": created,
            "modified": modified,
            "notes": self.col.note_count(),
            "cards": self.col.card_count(),
            "decks": len(self.col.decks.all_names_and_ids()),
            "note_types": len(self.col.models.all()),
            "tags": len(self.col.tags.all()),
            "due_today": total_due,
        }

    def _get_note_types(self, detail_names: set[str]) -> list[dict]:
        note_types = []
        for nt in self.col.models.all():
            entry: dict[str, Any] = {
                "name": nt["name"],
                "id": nt["id"],
                "fields": [f["name"] for f in nt["flds"]],
                "type": "cloze" if nt.get("type") == MODEL_CLOZE else "standard",
            }
            if nt["name"] in detail_names:
                entry["detail"] = {
                    "templates": [
                        {
                            "name": t["name"],
                            "front": t["qfmt"],
                            "back": t["afmt"],
                        }
                        for t in nt["tmpls"]
                    ],
                    "css": nt.get("css", ""),
                    "fields": [
                        {
                            "name": f["name"],
                            "font": f.get("font", ""),
                            "size": f.get("size", 0),
                            "description": f.get("description", ""),
                        }
                        for f in nt["flds"]
                    ],
                }
            note_types.append(entry)
        return note_types

    def _note_counts_by_deck(self) -> dict[str, int]:
        """Note count per deck (including subdecks), in a single pass.

        Mirrors ``find_notes("deck:NAME")`` without running a query per deck: a
        note counts toward a deck when any of its cards sits in that deck or a
        descendant. A card parked in a filtered deck counts toward *both* the
        filtered deck (``did``) and its original deck (``odid``), matching
        Anki's ``deck:`` search — verified against ``find_notes`` for nested,
        multi-deck, and filtered cases.
        """
        id_to_name = {d.id: d.name for d in self.col.decks.all_names_and_ids()}

        db = self.col.db
        assert db is not None  # always present on an open collection
        # Let SQLite produce the distinct (note, deck) pairs: a card in a filtered
        # deck counts toward both its current deck (did) and original (odid), and
        # UNION both dedups those and collapses a note's many cards in one deck to
        # a single row — so Python walks distinct pairs, not every card.
        nids_by_deck: dict[int, set[int]] = defaultdict(set)
        for nid, did in db.all(
            "select nid, did from cards union select nid, odid from cards where odid != 0"
        ):
            nids_by_deck[did].add(nid)

        # Roll each deck's notes up into itself and every ancestor (by name
        # prefix), so a parent's count includes all its subdecks' notes.
        rolled: dict[str, set[int]] = defaultdict(set)
        for did, nids in nids_by_deck.items():
            name = id_to_name.get(did)
            if name is None:
                continue
            parts = name.split("::")
            for i in range(1, len(parts) + 1):
                rolled["::".join(parts[:i])] |= nids

        return {name: len(rolled.get(name, set())) for name in id_to_name.values()}

    def _get_decks(self, counts: dict[str, int]) -> list[dict]:
        return [
            {
                "name": name_id.name,
                "id": name_id.id,
                "note_count": counts.get(name_id.name, 0),
            }
            for name_id in self.col.decks.all_names_and_ids()
        ]

    def _get_stats(self, tree: Any, note_counts: dict[str, int]) -> dict[str, Any]:
        # Top-level node counts already include their subdecks, so collection
        # totals sum over top-level decks only. The per-deck summary below
        # walks every node — each node's count is its own rolled-up total.
        total_due = sum(top.review_count + top.learn_count for top in tree.children)
        total_new = sum(top.new_count for top in tree.children)

        decks_summary: dict[str, dict] = {}

        def walk(node: Any, prefix: str = "") -> None:
            name = node.name
            if prefix:
                name = f"{prefix}::{node.name}"
            due = node.review_count + node.learn_count

            decks_summary[name] = {
                "notes": note_counts.get(name, 0),
                "due": due,
            }
            for child in node.children:
                walk(child, name)

        for top in tree.children:
            walk(top)

        return {
            "total_notes": self.col.note_count(),
            "total_cards": self.col.card_count(),
            "cards_due_today": total_due,
            "new_cards": total_new,
            "decks_summary": decks_summary,
        }

    # -- list / read ---------------------------------------------------------

    def _build_scope_query(
        self,
        *,
        ids: list[int] | None = None,
        deck: str | None = None,
        tags: list[str] | None = None,
        note_type: str | None = None,
    ) -> tuple[str | None, bool]:
        """Build an Anki search from structured selectors.

        Returns ``(query, no_match)``: ``query`` is ``None`` when no selector was
        given; ``no_match`` is True when a deck reference resolved to no deck (so
        the caller should treat the result as empty). Shared by list and
        find-replace so their scoping is identical.
        """
        parts: list[str] = []
        if deck is not None:
            resolved = self._resolve_deck_ref(deck)
            if resolved is None:
                return None, True
            parts.append(f'"deck:{resolved}"')
        if tags is not None:
            for tag in tags:
                if tag.startswith("-"):
                    parts.append(f"-tag:{tag[1:]}")
                else:
                    parts.append(f"tag:{tag}")
        if note_type is not None:
            parts.append(f'"note:{note_type}"')
        combined = " ".join(parts) if parts else None
        if ids is not None:
            id_query = f"nid:{','.join(str(i) for i in ids)}"
            combined = f"{id_query} {combined}" if combined else id_query
        return combined, False

    async def list_notes(
        self,
        *,
        ids: list[int] | None = None,
        deck: str | None = None,
        tags: list[str] | None = None,
        note_type: str | None = None,
        modified_since: str | None = None,
        fields_mode: str = "full",
        limit: int = 50,
    ) -> dict[str, Any]:
        return await self.run(
            lambda _c: self._list_notes(
                ids=ids,
                deck=deck,
                tags=tags,
                note_type=note_type,
                modified_since=modified_since,
                fields_mode=fields_mode,
                limit=limit,
            )
        )

    def _list_notes(
        self,
        *,
        ids: list[int] | None = None,
        deck: str | None = None,
        tags: list[str] | None = None,
        note_type: str | None = None,
        modified_since: str | None = None,
        fields_mode: str = "full",
        limit: int = 50,
    ) -> dict[str, Any]:
        if ids is not None and not any([deck, tags, note_type, modified_since]):
            return self._get_notes_by_ids(ids, fields_mode, limit)

        combined, no_match = self._build_scope_query(
            ids=ids, deck=deck, tags=tags, note_type=note_type
        )
        if no_match:
            # An explicit #id (or numeric id) that matches no deck → no hits.
            return {"notes": [], "total": 0, "limit": limit}

        if combined is None:
            if modified_since is not None:
                combined = "deck:*"
            else:
                return {"error": "At least one filter is required"}

        note_ids = list(self.col.find_notes(combined))

        mod_cutoff = None
        if modified_since is not None:
            dt = datetime.fromisoformat(modified_since)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            mod_cutoff = int(dt.timestamp())

        if mod_cutoff is not None:
            # Intersect with one indexed query against the notes table rather
            # than loading every candidate note via get_note (an N+1 over the
            # match set). ``notes.mod`` is the same value as ``Note.mod``.
            db = self.col.db
            assert db is not None  # always present on an open collection
            recent = set(db.list("select id from notes where mod >= ?", mod_cutoff))
            note_ids = [nid for nid in note_ids if nid in recent]

        total = len(note_ids)
        note_ids = note_ids[:limit]

        notes = self._notes_to_dicts(note_ids, fields_mode)
        return {"notes": notes, "total": total, "limit": limit}

    def _get_notes_by_ids(self, ids: list[int], fields_mode: str, limit: int) -> dict[str, Any]:
        # _notes_to_dicts skips ids absent from the collection, so a stale id is
        # dropped just as the per-note get_note path did (via NotFoundError).
        notes = self._notes_to_dicts(ids[:limit], fields_mode)
        return {"notes": notes, "total": len(notes), "limit": limit}

    async def query(
        self, query: str, *, fields_mode: str = "full", limit: int = 50
    ) -> dict[str, Any]:
        return await self.run(lambda _c: self._query(query, fields_mode=fields_mode, limit=limit))

    def _query(self, query: str, *, fields_mode: str = "full", limit: int = 50) -> dict[str, Any]:
        """Run a raw Anki search expression and return matching notes.

        The query string is passed straight to ``col.find_notes`` — the full
        Anki search language, no structured filters. A malformed expression
        raises ``anki.errors.SearchError``, which the tool layer turns into a
        ``ToolInputError``. Returns the same shape as ``list_notes`` (``total``
        is the full match count before ``limit``).
        """
        note_ids = list(self.col.find_notes(query))
        total = len(note_ids)
        notes = self._notes_to_dicts(note_ids[:limit], fields_mode)
        return {"notes": notes, "total": total, "limit": limit}

    async def search_substring(
        self,
        text: str,
        *,
        deck: str | None = None,
        tags: list[str] | None = None,
        exclude_ids: list[int] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return await self.run(
            lambda _c: self._search_substring(
                text, deck=deck, tags=tags, exclude_ids=exclude_ids, limit=limit
            )
        )

    def _search_substring(
        self,
        text: str,
        *,
        deck: str | None,
        tags: list[str] | None,
        exclude_ids: list[int] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Notes whose field text contains ``text`` literally (case-insensitive).

        Anki's ``"*term*"`` wildcard search is a fast pre-filter; each candidate
        is then confirmed (and annotated) by ``substring_info``, the authority.
        Returns full note dicts each carrying a ``substring`` annotation; no
        ``score`` (this is the non-semantic mechanism).
        """
        if not text.strip():
            return []
        parts = [f'"*{_escape_anki_text(text)}*"']
        if deck is not None:
            resolved = self._resolve_deck_ref(deck)
            if resolved is None:
                return []
            parts.append(f'"deck:{resolved}"')
        for tag in tags or []:
            parts.append(f'"tag:{tag}"')

        exclude = set(exclude_ids or [])
        results: list[dict[str, Any]] = []
        for nid in self.col.find_notes(" ".join(parts)):
            if nid in exclude:
                continue
            note = self._note_to_dict(nid, "full")
            info = substring_info(note.get("content"), text)
            if info is None:
                continue  # Anki matched across markup/normalization; not a literal hit
            note["substring"] = info
            results.append(note)
            if len(results) >= limit:
                break
        return results

    async def find_replace(
        self,
        search: str,
        replacement: str,
        *,
        regex: bool = False,
        match_case: bool = False,
        field: str | None = None,
        deck: str | None = None,
        tags: list[str] | None = None,
        note_type: str | None = None,
        ids: list[int] | None = None,
        dry_run: bool = False,
        sample_limit: int = 20,
    ) -> dict[str, Any]:
        return await self.run(
            lambda _c: self._find_replace(
                search,
                replacement,
                regex=regex,
                match_case=match_case,
                field=field,
                deck=deck,
                tags=tags,
                note_type=note_type,
                ids=ids,
                dry_run=dry_run,
                sample_limit=sample_limit,
            )
        )

    def _find_replace(
        self,
        search: str,
        replacement: str,
        *,
        regex: bool,
        match_case: bool,
        field: str | None,
        deck: str | None,
        tags: list[str] | None,
        note_type: str | None,
        ids: list[int] | None,
        dry_run: bool,
        sample_limit: int,
    ) -> dict[str, Any]:
        """Bulk find-and-replace over a scoped note set.

        Preview (samples + predicted count) is computed in Python; the apply runs
        Anki's ``find_and_replace`` (authoritative) and the changed set is the
        notes whose ``mod`` advanced — re-embedded by the tool layer.
        """
        empty = {"notes_changed": 0, "dry_run": dry_run, "samples": [], "changed_ids": []}
        combined, no_match = self._build_scope_query(
            ids=ids, deck=deck, tags=tags, note_type=note_type
        )
        if no_match or combined is None:
            return empty
        candidates = list(self.col.find_notes(combined))
        if not candidates:
            return empty

        # Predict changes in Python for the preview / dry-run count. Read fields
        # in one query (shared with note_texts) rather than get_note per note.
        samples: list[dict[str, Any]] = []
        predicted: set[int] = set()
        for nid, names, values in self._note_field_rows(self.col, candidates):
            for fname, value in zip(names, values, strict=False):
                if field is not None and fname != field:
                    continue
                new = apply_replacement(
                    value, search, replacement, regex=regex, match_case=match_case
                )
                if new != value:
                    predicted.add(nid)
                    if len(samples) < sample_limit:
                        samples.append({"id": nid, "field": fname, "before": value, "after": new})

        if dry_run:
            return {
                "notes_changed": len(predicted),
                "dry_run": True,
                "samples": samples,
                "changed_ids": [],
            }

        # Apply via Anki; detect the actually-changed notes by mod-bump diff.
        # Detect the actually-changed notes by diffing field content before/after
        # (note.mod is only second-resolution, so a fast edit wouldn't show a bump).
        db = self.col.db
        assert db is not None
        id_list = ",".join(str(i) for i in candidates)
        flds_sql = f"select id, flds from notes where id in ({id_list})"
        before: dict[int, str] = {int(r[0]): r[1] for r in db.all(flds_sql)}
        count = self.col.find_and_replace(
            note_ids=candidates,  # type: ignore[arg-type]
            search=search,
            replacement=replacement,
            regex=regex,
            field_name=field,
            match_case=match_case,
        ).count
        after: dict[int, str] = {int(r[0]): r[1] for r in db.all(flds_sql)}
        changed_ids = [nid for nid in candidates if after.get(nid) != before.get(nid)]
        return {
            "notes_changed": count,
            "dry_run": False,
            "samples": samples,
            "changed_ids": changed_ids,
        }

    async def note_to_dict(self, nid: int, fields_mode: str) -> dict[str, Any]:
        return await self.run(lambda _c: self._note_to_dict(nid, fields_mode))

    def _notes_to_dicts(self, nids: Sequence[int], fields_mode: str) -> list[dict[str, Any]]:
        """Serialize many notes in a fixed number of queries (no per-note N+1).

        Reads all note rows and their first-card decks in two indexed queries
        instead of ``get_note`` + ``note.cards()`` + ``decks.get()`` *per note*.
        Field values come from the raw ``flds`` column (Anki joins fields with
        U+001F, as already relied on in ``_find_empty_notes``) and tags from the
        space-delimited ``tags`` column — both verified equal to the Note API.
        Results follow input order; ids missing from the collection are skipped
        (matching ``get_note`` raising ``NotFoundError`` for the single-note path).
        """
        if not nids:
            return []
        db = self.col.db
        assert db is not None  # always present on an open collection
        id_list = ",".join(str(n) for n in nids)

        note_rows = {
            r[0]: r
            for r in db.all(f"select id, mid, tags, flds, mod from notes where id in ({id_list})")
        }
        # First card's deck per note (lowest ord), matching note.cards()[0].did.
        deck_by_nid: dict[int, int] = {}
        for nid, did in db.all(f"select nid, did from cards where nid in ({id_list}) order by ord"):
            deck_by_nid.setdefault(nid, did)
        deck_names = {d.id: d.name for d in self.col.decks.all_names_and_ids()}

        model_cache: dict[int, tuple[str, list[str]]] = {}

        def _model(mid: int) -> tuple[str, list[str]]:
            cached = model_cache.get(mid)
            if cached is None:
                nt = self.col.models.get(mid)  # type: ignore[arg-type]
                cached = (nt["name"], [f["name"] for f in nt["flds"]]) if nt else ("Unknown", [])
                model_cache[mid] = cached
            return cached

        out: list[dict[str, Any]] = []
        for nid in nids:
            row = note_rows.get(nid)
            if row is None:
                continue
            _id, mid, tags, flds, mod = row
            name, field_names = _model(mid)
            did = deck_by_nid.get(nid)
            result: dict[str, Any] = {
                "id": _id,
                "note_type": name,
                "deck": deck_names.get(did, "Default") if did is not None else "Default",
                "tags": tags.split(),
                "modified": datetime.fromtimestamp(mod, tz=UTC).isoformat(),
            }
            if fields_mode == "full":
                result["content"] = dict(zip(field_names, flds.split("\x1f"), strict=False))
            out.append(result)
        return out

    def _note_to_dict(self, nid: int, fields_mode: str) -> dict[str, Any]:
        dicts = self._notes_to_dicts([nid], fields_mode)
        if not dicts:
            # Missing note (rare — a stale id, e.g. an index neighbor pointing at
            # a deleted note): defer to Anki to raise its NotFoundError as before.
            self.col.get_note(nid)  # type: ignore[arg-type]
        return dicts[0]

    # -- upsert / delete -----------------------------------------------------

    async def upsert_notes(
        self,
        notes: list[dict[str, Any]],
        *,
        on_duplicate: OnDuplicate = "error",
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        return await self.run(
            lambda _c: self._upsert_notes(notes, on_duplicate=on_duplicate, dry_run=dry_run)
        )

    def _upsert_notes(
        self,
        notes: list[dict[str, Any]],
        *,
        on_duplicate: OnDuplicate = "error",
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        results = []
        for i, note_input in enumerate(notes):
            try:
                if "id" in note_input and note_input["id"] is not None:
                    results.append(self._update_note(note_input, index=i, dry_run=dry_run))
                else:
                    results.append(
                        self._create_note(
                            note_input, index=i, on_duplicate=on_duplicate, dry_run=dry_run
                        )
                    )
            except Exception as e:
                results.append(
                    {
                        "status": "error",
                        "index": i,
                        "error": str(e),
                    }
                )
        return results

    def _check_new_note(self, note: Any, on_duplicate: OnDuplicate) -> dict[str, Any] | None:
        """Run Anki's add-note validation on a candidate note.

        Returns ``None`` if the note is addable (including a duplicate when
        ``on_duplicate="allow"``), or a partial result dict (``status`` +
        ``reason`` [+ ``error``], no ``index``) describing why not.
        """
        state = note.fields_check()
        if state == NoteFieldsCheckResult.NORMAL:
            return None
        if state == NoteFieldsCheckResult.DUPLICATE:
            if on_duplicate == "allow":
                return None
            if on_duplicate == "skip":
                return {"status": "skipped", "reason": "duplicate"}
            return {"status": "error", "error": _DUPLICATE_MESSAGE, "reason": "duplicate"}
        reason, message = _STRUCTURAL_PROBLEMS[state]
        return {"status": "error", "error": message, "reason": reason}

    def _create_note(
        self, note_input: dict[str, Any], *, index: int, on_duplicate: OnDuplicate, dry_run: bool
    ) -> dict[str, Any]:
        note_type_name = note_input.get("note_type")
        deck_name = note_input.get("deck")
        fields = note_input.get("fields")

        if not note_type_name:
            raise ValueError("note_type is required for new notes")
        if not deck_name:
            raise ValueError("deck is required for new notes")
        if not fields:
            raise ValueError("fields is required for new notes")

        notetype = self.col.models.by_name(note_type_name)
        if notetype is None:
            return {
                "status": "error",
                "index": index,
                "error": f"Note type '{note_type_name}' not found",
                "reason": "unknown_note_type",
            }

        # Resolve the deck reference (name / numeric id / #id) to a canonical
        # name. This is read-only — a plain name for a not-yet-existing deck
        # passes through unchanged and is auto-created on the write path below,
        # so a dry run still creates nothing. Only an id/#id pointing at no deck
        # is an error.
        resolved_deck = self._resolve_deck_ref(deck_name)
        if resolved_deck is None:
            raise ValueError(f"Deck '{deck_name}' not found")
        deck_name = resolved_deck

        note = self.col.new_note(notetype)
        for field_name, value in fields.items():
            if field_name not in note:
                return {
                    "status": "error",
                    "index": index,
                    "error": (
                        f"Field '{field_name}' not found in note type '{note_type_name}'. "
                        f"Available fields: {list(note.keys())}"
                    ),
                    "reason": "unknown_field",
                }
            note[field_name] = value

        if "tags" in note_input and note_input["tags"] is not None:
            note.tags = note_input["tags"]

        # Anki's own add-note validation (duplicate / empty / cloze structure),
        # applied before any write so dry runs and real runs classify identically.
        problem = self._check_new_note(note, on_duplicate)
        if problem is not None:
            return {"index": index, **problem}

        if dry_run:
            return {"status": "ok", "index": index, "action": "create"}

        deck_id = self.col.decks.id_for_name(deck_name)
        if deck_id is None:
            deck_id = self.col.decks.id(deck_name)
        if deck_id is None:
            raise ValueError(f"Could not find or create deck '{deck_name}'")

        self.col.add_note(note, deck_id)
        logger.debug("Created note %d (type=%s, deck=%s)", note.id, note_type_name, deck_name)
        return {"status": "created", "id": note.id}

    def _update_note(
        self, note_input: dict[str, Any], *, index: int, dry_run: bool
    ) -> dict[str, Any]:
        nid = note_input["id"]
        try:
            note = self.col.get_note(nid)  # type: ignore[arg-type]
        except NotFoundError as err:
            raise ValueError(f"Note {nid} not found") from err

        if "note_type" in note_input and note_input["note_type"] is not None:
            notetype = self.col.models.get(note.mid)
            current_type = notetype["name"] if notetype else "Unknown"
            if note_input["note_type"] != current_type:
                raise ValueError(
                    f"Cannot change note type (current: '{current_type}', "
                    f"requested: '{note_input['note_type']}')"
                )

        if "fields" in note_input and note_input["fields"] is not None:
            for field_name, value in note_input["fields"].items():
                if field_name not in note:
                    nt = self.col.models.get(note.mid)
                    nt_name = nt["name"] if nt else "Unknown"
                    return {
                        "status": "error",
                        "index": index,
                        "error": (
                            f"Field '{field_name}' not found in note type "
                            f"'{nt_name}'. Available fields: {list(note.keys())}"
                        ),
                        "reason": "unknown_field",
                    }
                note[field_name] = value

        if "tags" in note_input and note_input["tags"] is not None:
            note.tags = note_input["tags"]

        if dry_run:
            return {"status": "ok", "index": index, "action": "update"}

        self.col.update_note(note)

        if "deck" in note_input and note_input["deck"] is not None:
            resolved_deck = self._resolve_deck_ref(note_input["deck"])
            if resolved_deck is None:
                raise ValueError(f"Deck '{note_input['deck']}' not found")
            target_deck_id = self.col.decks.id_for_name(resolved_deck)
            if target_deck_id is None:
                target_deck_id = self.col.decks.id(resolved_deck)
            if target_deck_id is not None:
                card_ids = note.card_ids()
                self.col.set_deck(card_ids, int(target_deck_id))

        logger.debug("Updated note %d", note.id)
        return {"status": "updated", "id": note.id}

    async def migrate_note_type(
        self,
        note_ids: list[int],
        new_note_type: str,
        field_map: dict[str, str],
        *,
        template_map: dict[str, str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return await self.run(
            lambda _c: self._migrate_note_type(
                note_ids, new_note_type, field_map, template_map=template_map, dry_run=dry_run
            )
        )

    def _migrate_note_type(
        self,
        note_ids: list[int],
        new_note_type: str,
        field_map: dict[str, str],
        *,
        template_map: dict[str, str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Change a set of notes from one note type to another (Anki's models.change).

        All ``note_ids`` must currently share a single source note type — one
        field/template map can't apply to mixed types. ``field_map`` maps source
        field *names* to target field names; source fields absent from it are
        dropped (reported in ``dropped_fields``), and target fields nothing maps
        into are reported in ``new_empty_fields``. ``template_map`` is optional;
        omitted, Anki maps templates by ordinal. Note IDs and (for mapped
        templates) card scheduling are preserved. Raises ``ValueError`` for any
        caller error — the tool layer turns it into a ``ToolInputError``.
        """
        try:
            notes = [self.col.get_note(nid) for nid in note_ids]  # type: ignore[arg-type]
        except NotFoundError as err:
            raise ValueError(f"Note not found: {err}") from err

        source_mids = {n.mid for n in notes}
        if len(source_mids) != 1:
            raise ValueError("All notes must currently share one note type to migrate together.")
        source = self.col.models.get(source_mids.pop())
        assert source is not None

        target = self.col.models.by_name(new_note_type)
        if target is None:
            raise ValueError(f"Note type '{new_note_type}' not found")
        if target["id"] == source["id"]:
            raise ValueError(f"Notes already use note type '{new_note_type}'.")

        src_fields = {f["name"]: f["ord"] for f in source["flds"]}
        tgt_fields = {f["name"]: f["ord"] for f in target["flds"]}
        for old, new in field_map.items():
            if old not in src_fields:
                raise ValueError(f"Source field '{old}' not in note type '{source['name']}'")
            if new not in tgt_fields:
                raise ValueError(f"Target field '{new}' not in note type '{new_note_type}'")
        targets = list(field_map.values())
        ambiguous = sorted({t for t in targets if targets.count(t) > 1})
        if ambiguous:
            raise ValueError(f"Multiple source fields map to the same target field(s): {ambiguous}")

        # fmap covers every source field ord: mapped -> target ord, else None (drop).
        fmap: dict[int, int | None] = {
            ord_: (tgt_fields[field_map[name]] if name in field_map else None)
            for name, ord_ in src_fields.items()
        }
        dropped_fields = [name for name in src_fields if name not in field_map]
        mapped_targets = set(field_map.values())
        new_empty_fields = [name for name in tgt_fields if name not in mapped_targets]

        cmap: dict[int, int | None] | None = None
        if template_map:
            src_tmpls = {t["name"]: t["ord"] for t in source["tmpls"]}
            tgt_tmpls = {t["name"]: t["ord"] for t in target["tmpls"]}
            for old, new in template_map.items():
                if old not in src_tmpls:
                    raise ValueError(f"Source template '{old}' not in note type '{source['name']}'")
                if new not in tgt_tmpls:
                    raise ValueError(f"Target template '{new}' not in note type '{new_note_type}'")
            cmap = {
                ord_: (tgt_tmpls[template_map[name]] if name in template_map else None)
                for name, ord_ in src_tmpls.items()
            }

        result = {
            "changed": [n.id for n in notes],
            "from_note_type": source["name"],
            "to_note_type": target["name"],
            "dropped_fields": dropped_fields,
            "new_empty_fields": new_empty_fields,
            "dry_run": dry_run,
        }
        if dry_run:
            return result

        self.col.models.change(
            source,
            [n.id for n in notes],  # type: ignore[misc]
            target,
            fmap,
            cmap,
        )
        logger.debug(
            "Migrated %d note(s) %s -> %s (dropped=%s)",
            len(notes),
            source["name"],
            target["name"],
            dropped_fields,
        )
        return result

    async def delete_notes(self, ids: list[int]) -> dict[str, Any]:
        return await self.run(lambda _c: self._delete_notes(ids))

    def _delete_notes(self, ids: list[int]) -> dict[str, Any]:
        existing = set(self.col.find_notes(f"nid:{','.join(str(i) for i in ids)}"))
        not_found = [i for i in ids if i not in existing]
        deleted = list(existing)

        if deleted:
            self.col.remove_notes(deleted)

        return {"deleted": deleted, "not_found": not_found}

    # -- tags ----------------------------------------------------------------

    async def update_note_tags(
        self,
        note_ids: list[int],
        *,
        set_tags: list[str] | None,
        add: list[str],
        remove: list[str],
    ) -> dict[str, Any]:
        return await self.run(
            lambda _c: self._update_note_tags(note_ids, set_tags=set_tags, add=add, remove=remove)
        )

    def _update_note_tags(
        self,
        note_ids: list[int],
        *,
        set_tags: list[str] | None,
        add: list[str],
        remove: list[str],
    ) -> dict[str, Any]:
        """Edit tags on a set of notes.

        ``set_tags`` is a full replace (an empty list clears all tags); it is
        mutually exclusive with ``add``/``remove``, which apply additively and
        subtractively without disturbing other tags. Validation of that rule
        lives in the tool layer — here ``set_tags is not None`` selects replace
        mode. Returns the notes the operation applied to and any IDs not found.
        """
        existing = set(self.col.find_notes(f"nid:{','.join(str(i) for i in note_ids)}"))
        not_found = [i for i in note_ids if i not in existing]
        targets = [i for i in note_ids if i in existing]

        if targets:
            if set_tags is not None:
                for nid in targets:
                    note = self.col.get_note(nid)  # type: ignore[arg-type]
                    note.tags = list(set_tags)
                    self.col.update_note(note)
            else:
                # Remove before add so a tag named in both ends up present.
                if remove:
                    self.col.tags.bulk_remove(targets, " ".join(remove))  # type: ignore[arg-type]
                if add:
                    self.col.tags.bulk_add(targets, " ".join(add))  # type: ignore[arg-type]

        return {"notes_modified": len(targets), "not_found": not_found}

    async def rename_tag(self, old: str, new: str, note_ids: list[int]) -> dict[str, Any]:
        return await self.run(lambda _c: self._rename_tag(old, new, note_ids))

    def _rename_tag(self, old: str, new: str, note_ids: list[int]) -> dict[str, Any]:
        """Rename a tag collection-wide (empty ``note_ids``) or on a note set.

        The note-scoped path renames the tag *exactly* (match notes carrying
        ``old``, then swap it for ``new``) rather than a substring find/replace,
        so renaming ``jp`` never touches ``jp-verbs``.
        """
        if not note_ids:
            count = self.col.tags.rename(old, new).count
            return {"notes_modified": count}

        scope = ",".join(str(i) for i in note_ids)
        matching = list(self.col.find_notes(f"(nid:{scope}) tag:{old}"))
        if matching:
            self.col.tags.bulk_remove(matching, old)
            self.col.tags.bulk_add(matching, new)
        return {"notes_modified": len(matching)}

    # -- collection maintenance ----------------------------------------------

    async def prune(
        self,
        *,
        unused_tags: bool,
        empty_notes: bool,
        empty_cards: bool,
        unused_media: bool,
        dry_run: bool,
    ) -> tuple[dict[str, Any], list[int]]:
        return await self.run(
            lambda _c: self._prune(
                unused_tags=unused_tags,
                empty_notes=empty_notes,
                empty_cards=empty_cards,
                unused_media=unused_media,
                dry_run=dry_run,
            )
        )

    def _find_unused_media(self) -> list[str]:
        """Media files on disk that no note references (Anki's media check)."""
        return list(self.col.media.check().unused)

    def _find_empty_notes(self) -> list[int]:
        """Note ids whose every field is blank (no text and no media).

        Uses ``embed_text.field_is_blank`` per field, so a note made only of an
        image or ``[sound:…]`` is *not* empty. Scans the whole ``notes`` table —
        prune is a maintenance op, so the full pass is acceptable.
        """
        db = self.col.db
        assert db is not None  # always present on an open collection
        empty: list[int] = []
        for nid, flds in db.all("select id, flds from notes"):
            if all(field_is_blank(f) for f in flds.split("\x1f")):
                empty.append(int(nid))
        return empty

    def _prune(
        self,
        *,
        unused_tags: bool,
        empty_notes: bool,
        empty_cards: bool,
        unused_media: bool,
        dry_run: bool,
    ) -> tuple[dict[str, Any], list[int]]:
        """Run the requested cleanups; return (result, note ids removed).

        On apply the cleanups run in order — empty notes, then empty cards, then
        unused tags, then unused media — so tag-registry names and media files
        orphaned by the deletions get cleared in the same call. The returned
        note-id list (empty notes + empty cards that lost their last card) is what
        the tool layer removes from the index; trashing media touches no vectors.

        Dry-run mutates nothing and computes each cleanup against the *current*
        collection; the empty-cards preview subtracts the empty-note ids (those
        notes go first on apply) so a note isn't listed under both. Because the
        previews are independent, a real apply may clear a few more tags than the
        dry-run reported (deletions free additional tags).
        """
        result: dict[str, Any] = {"dry_run": dry_run}
        removed_note_ids: list[int] = []

        empty_note_ids: list[int] = []
        if empty_notes:
            empty_note_ids = self._find_empty_notes()
            if not dry_run and empty_note_ids:
                self.col.remove_notes(empty_note_ids)  # type: ignore[arg-type]
            result["empty_notes"] = {"removed": empty_note_ids}
            removed_note_ids += empty_note_ids

        if empty_cards:
            report = self.col.get_empty_cards()
            card_ids = [cid for n in report.notes for cid in n.card_ids]
            notes_deleted = [n.note_id for n in report.notes if n.will_delete_note]
            if dry_run:
                # Empty notes go first on apply, so don't double-list them here.
                already = set(empty_note_ids)
                notes_deleted = [nid for nid in notes_deleted if nid not in already]
            elif card_ids:
                self.col.remove_cards_and_orphaned_notes(card_ids)  # type: ignore[arg-type]
            result["empty_cards"] = {
                "cards_removed": len(card_ids),
                "notes_deleted": notes_deleted,
            }
            removed_note_ids += notes_deleted

        if unused_tags:
            names = self._unused_tag_names()
            if not dry_run and names:
                self.col.tags.clear_unused_tags()
            result["unused_tags"] = {"removed": len(names), "tags": names}

        if unused_media:
            # Last, so on apply it catches media orphaned by the note/card deletions
            # above (Anki's check re-reads the post-deletion reference set). Like
            # unused_tags, the dry-run preview reflects the *current* references, so
            # an apply may trash a few more than the preview showed. Trashing media
            # changes no note text or note set, so the index is untouched.
            media_files = self._find_unused_media()
            if not dry_run and media_files:
                self.col.media.trash_files(media_files)
            result["unused_media"] = {"removed": len(media_files), "files": media_files}

        return result, removed_note_ids

    def _unused_tag_names(self) -> list[str]:
        """Registered tags present on no note — in one scan, not a find_notes per
        tag. A tag is *used* if a note carries it or any descendant (``a`` is used
        when a note has ``a::b``), mirroring Anki's hierarchical, case-insensitive
        ``tag:`` search: collect every note tag's ancestor chain, case-folded, and
        flag the registered tags absent from it.
        """
        db = self.col.db
        assert db is not None  # always present on an open collection
        used: set[str] = set()
        for (tagstr,) in db.all("select distinct tags from notes"):
            for tag in tagstr.split():
                parts = tag.lower().split("::")
                for i in range(1, len(parts) + 1):
                    used.add("::".join(parts[:i]))
        return [t for t in self.col.tags.all() if t.lower() not in used]

    # -- media (#70) ---------------------------------------------------------

    async def store_media(
        self, items: list[dict[str, Any]], *, allow_private_fetch: bool
    ) -> list[dict[str, Any]]:
        """Store a batch of media files; one bad item never sinks the batch.

        URL items are fetched **off the worker thread** (``asyncio.to_thread``) so a
        slow download doesn't block collection ops; base64 items are decoded here.
        Both then write to the media folder on the worker thread. Per-item failures
        (bad base64, unfetchable/SSRF-blocked URL, oversize) become error results.
        """
        prepared: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            name = item.get("filename")
            try:
                if item.get("data") is not None:
                    raw = base64.b64decode(item["data"], validate=True)
                    prepared.append(
                        {"index": index, "raw": raw, "name": name, "content_type": None}
                    )
                else:
                    url = item["url"]
                    raw, content_type = await asyncio.to_thread(
                        _fetch_media_url, url, allow_private=allow_private_fetch
                    )
                    prepared.append(
                        {
                            "index": index,
                            "raw": raw,
                            "name": name or _safe_media_name(urlparse(url).path),
                            "content_type": content_type,
                        }
                    )
            except Exception as e:
                prepared.append({"index": index, "error": str(e), "name": name})
        return await self.run(lambda _c: self._write_media_batch(prepared))

    def _write_media_batch(self, prepared: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for p in prepared:
            if "error" in p:
                results.append(
                    {
                        "status": "error",
                        "index": p["index"],
                        "filename": p["name"],
                        "error": p["error"],
                    }
                )
                continue
            try:
                results.append(self._write_one_media(p))
            except Exception as e:
                results.append(
                    {
                        "status": "error",
                        "index": p["index"],
                        "filename": p.get("name"),
                        "error": str(e),
                    }
                )
        return results

    def _write_one_media(self, p: dict[str, Any]) -> dict[str, Any]:
        raw: bytes = p["raw"]
        name: str | None = p["name"]
        content_type: str | None = p["content_type"]
        if len(raw) > MEDIA_MAX_BYTES:
            raise ValueError(f"file exceeds the {MEDIA_MAX_BYTES}-byte limit")
        # Derive an extension from the HTTP type when the (URL-derived) name lacks
        # one, so Anki's renderer knows what it is. A base64 item always has a
        # filename with an extension (schema-enforced).
        if (not name or "." not in os.path.basename(name)) and content_type:
            name = self.col.media.add_extension_based_on_mime(name or "media", content_type)
        safe = _safe_media_name(name or "")
        if not safe:
            raise ValueError("could not determine a filename")
        existed = self.col.media.have(safe)
        stored = self.col.media.write_data(safe, raw)
        return {
            "status": "stored",
            "index": p["index"],
            "filename": stored,
            "mime": _guess_mime(stored),
            "size_bytes": len(raw),
            # Identical content already present → Anki kept the name, wrote nothing
            # new. A different-content collision instead yields stored != safe.
            "deduped": existed and stored == safe,
        }

    async def fetch_media(
        self, filenames: list[str], *, max_inline_bytes: int
    ) -> list[dict[str, Any]]:
        return await self.run(lambda _c: self._fetch_media(filenames, max_inline_bytes))

    def _fetch_media(self, filenames: list[str], max_inline_bytes: int) -> list[dict[str, Any]]:
        """Read each media file back: inline (base64) when small, else path-only."""
        media_dir = self.col.media.dir()
        results: list[dict[str, Any]] = []
        for fn in filenames:
            safe = _safe_media_name(fn)
            path = os.path.join(media_dir, safe) if safe else ""
            if not safe or not os.path.isfile(path):
                results.append({"status": "missing", "filename": fn})
                continue
            size = os.path.getsize(path)
            mime = _guess_mime(safe)
            if size > max_inline_bytes:
                results.append(
                    {
                        "status": "too_large",
                        "filename": safe,
                        "path": path,
                        "mime": mime,
                        "size_bytes": size,
                    }
                )
                continue
            with open(path, "rb") as fh:
                data = base64.b64encode(fh.read()).decode("ascii")
            results.append(
                {
                    "status": "inline",
                    "filename": safe,
                    "path": path,
                    "mime": mime,
                    "size_bytes": size,
                    "data": data,
                }
            )
        return results

    async def list_media(self, *, pattern: str | None, limit: int | None) -> dict[str, Any]:
        return await self.run(lambda _c: self._list_media(pattern, limit))

    def _list_media(self, pattern: str | None, limit: int | None) -> dict[str, Any]:
        media_dir = self.col.media.dir()
        if os.path.isdir(media_dir):
            names = sorted(
                n for n in os.listdir(media_dir) if os.path.isfile(os.path.join(media_dir, n))
            )
        else:
            names = []
        if pattern:
            names = [n for n in names if fnmatch.fnmatch(n, pattern)]
        count = len(names)
        if limit is not None:
            names = names[:limit]
        files = [
            {
                "filename": n,
                "mime": _guess_mime(n),
                "size_bytes": os.path.getsize(os.path.join(media_dir, n)),
            }
            for n in names
        ]
        return {"media_dir": media_dir, "count": count, "files": files}

    async def delete_media(self, filenames: list[str]) -> dict[str, Any]:
        return await self.run(lambda _c: self._delete_media(filenames))

    def _delete_media(self, filenames: list[str]) -> dict[str, Any]:
        """Move existing media files to Anki's trash (recoverable, sync-aware)."""
        deleted: list[str] = []
        not_found: list[str] = []
        to_trash: list[str] = []
        for fn in filenames:
            safe = _safe_media_name(fn)
            if safe and self.col.media.have(safe):
                to_trash.append(safe)
                deleted.append(fn)  # echo the caller's reference, like delete_decks
            else:
                not_found.append(fn)
        if to_trash:
            self.col.media.trash_files(to_trash)
        return {"deleted": deleted, "not_found": not_found}

    async def media_check(self) -> dict[str, Any]:
        return await self.run(lambda _c: self._media_check())

    def _media_check(self) -> dict[str, Any]:
        media_dir = self.col.media.dir()
        report = self.col.media.check()
        return {
            "media_dir": media_dir,
            "unused": list(report.unused),
            "missing": list(report.missing),
            "missing_media_notes": [int(nid) for nid in report.missing_media_notes],
            "have_trash": report.have_trash,
        }

    # -- decks ---------------------------------------------------------------

    async def resolve_deck_ref(self, ref: str) -> str | None:
        return await self.run(lambda _c: self._resolve_deck_ref(ref))

    def _resolve_deck_ref(self, ref: str) -> str | None:
        """Map a deck reference to a deck name.

        Accepts a deck name, a numeric deck ID, or a ``#``-prefixed ID:
        - ``#<id>`` is always an ID — returns the deck's name, or ``None`` if no
          deck has that ID.
        - a bare integer is tried as an ID first, falling back to a literal name
          if no deck has that ID (a deck genuinely named "123" still resolves).
        - anything else is a name, returned unchanged.
        """
        if ref.startswith("#") and ref[1:].isdigit():
            return self.col.decks.name_if_exists(int(ref[1:]))  # type: ignore[arg-type]
        if ref.isdigit():
            name = self.col.decks.name_if_exists(int(ref))  # type: ignore[arg-type]
            return name if name is not None else ref
        return ref

    async def upsert_decks(self, decks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return await self.run(lambda _c: self._upsert_decks(decks))

    def _upsert_decks(self, decks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Create or rename decks in bulk (same shape as ``_upsert_notes``).

        Each item carries the desired ``name``; an optional ``id`` selects an
        existing deck to rename to that name. Renaming onto a name already used by
        a *different* deck is rejected (Anki would silently disambiguate to
        ``name+``); to consolidate, move the notes instead. Per-item try/except so
        one bad item doesn't sink the batch.
        """
        results: list[dict[str, Any]] = []
        for index, deck in enumerate(decks):
            try:
                results.append(self._upsert_one_deck(deck))
            except Exception as e:
                results.append({"status": "error", "index": index, "error": str(e)})
        return results

    def _upsert_one_deck(self, deck: dict[str, Any]) -> dict[str, Any]:
        name = deck.get("name")
        if not name:
            raise ValueError("name is required")
        deck_id = deck.get("id")
        if deck_id is not None:
            if self.col.decks.get(deck_id, default=False) is None:  # type: ignore[arg-type]
                raise ValueError(f"Deck {deck_id} not found")
            clash = self.col.decks.id_for_name(name)
            if clash is not None and int(clash) != int(deck_id):
                raise ValueError(f"A deck named '{name}' already exists")
            self.col.decks.rename(deck_id, name)  # type: ignore[arg-type]
            logger.debug("Renamed deck %d -> %s", deck_id, name)
            return {"status": "updated", "id": int(deck_id), "name": name}
        existing = self.col.decks.id_for_name(name)
        if existing is not None:
            return {"status": "updated", "id": int(existing), "name": name}
        new_id = self.col.decks.add_normal_deck_with_name(name).id
        logger.debug("Created deck %s (id=%d)", name, new_id)
        return {"status": "created", "id": int(new_id), "name": name}

    async def delete_decks(self, names: list[str]) -> dict[str, Any]:
        return await self.run(lambda _c: self._delete_decks(names))

    def _delete_decks(self, names: list[str]) -> dict[str, Any]:
        """Delete decks by name — only if empty (no cards in the deck or subdecks).

        Emptying a deck is composed separately (move/merge its notes out, then
        delete), so this never deletes a card or note and never touches the index
        beyond the caller's col_mod bump.
        """
        deleted: list[str] = []
        not_found: list[str] = []
        not_empty: list[str] = []
        to_remove: list[int] = []
        # Report back the reference the caller passed (name or #id), not the
        # resolved name, so the result lists echo their input.
        for ref in names:
            resolved = self._resolve_deck_ref(ref)
            deck_id = self.col.decks.id_for_name(resolved) if resolved is not None else None
            if deck_id is None:
                not_found.append(ref)
            elif self.col.decks.card_count(deck_id, include_subdecks=True) > 0:
                not_empty.append(ref)
            else:
                to_remove.append(deck_id)
                deleted.append(ref)
        if to_remove:
            self.col.decks.remove(to_remove)  # type: ignore[arg-type]
        return {"deleted": deleted, "not_found": not_found, "not_empty": not_empty}

    # -- embedding text ------------------------------------------------------

    async def note_texts_for_embedding(self, note_ids: Sequence[int]) -> list[str]:
        """Return concatenated field text for each note, suitable for embedding.

        Notes that don't exist are returned as empty strings (same index position).
        """
        return await self.run(lambda c: self.note_texts(c, note_ids))

    @staticmethod
    def note_texts(col: Collection, note_ids: Sequence[int]) -> list[str]:
        """Normalized field text for each note id. Must run on the worker thread.

        Each field value is run through ``normalize_for_embedding`` (cloze fill,
        HTML/media stripping, entity/whitespace cleanup) so we embed the rendered
        text — deterministically, the same regardless of when or how the note is
        embedded. Fields that normalize to nothing (pure markup/media) are
        dropped.

        Built on ``_note_field_rows`` (one query, no per-note ``get_note``), which
        runs once per note on every index rebuild. Missing ids yield "" at the
        same position.
        """
        rendered = {
            nid: "\n".join(
                f"{k}: {cleaned}"
                for k, v in zip(names, values, strict=False)
                if (cleaned := normalize_for_embedding(v))
            )
            for nid, names, values in CollectionWrapper._note_field_rows(col, note_ids)
        }
        return [rendered.get(nid, "") for nid in note_ids]

    @staticmethod
    def _note_field_rows(
        col: Collection, note_ids: Sequence[int]
    ) -> Iterator[tuple[int, list[str], list[str]]]:
        """Yield ``(note_id, field_names, field_values)`` per existing note, in
        input order, from a single query over the raw notes table.

        The shared low-level field reader behind ``note_texts`` and the
        find-replace preview — both need a note's fields without a per-note
        ``get_note`` round trip. Field names come from the note type (field order,
        == ``Note.keys()``); values from the ``flds`` column split on U+001F
        (== ``Note.values()``, as relied on in ``_find_empty_notes``). Ids absent
        from the collection are skipped (so callers that need a fixed-length,
        position-aligned result map by id rather than zipping).
        """
        ids = list(note_ids)
        if not ids:
            return
        db = col.db
        assert db is not None  # always present on an open collection
        id_list = ",".join(str(n) for n in ids)
        rows = {
            r[0]: (r[1], r[2])
            for r in db.all(f"select id, mid, flds from notes where id in ({id_list})")
        }
        field_names: dict[int, list[str]] = {}
        for nid in ids:
            row = rows.get(nid)
            if row is None:
                continue
            mid, flds = row
            names = field_names.get(mid)
            if names is None:
                nt = col.models.get(mid)  # type: ignore[arg-type]
                names = [f["name"] for f in nt["flds"]] if nt else []
                field_names[mid] = names
            yield nid, names, flds.split("\x1f")

    # -- note types ----------------------------------------------------------

    async def delete_note_types(self, ids: list[int]) -> dict[str, Any]:
        return await self.run(lambda _c: self._delete_note_types(ids))

    def _delete_note_types(self, ids: list[int]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for nt_id in ids:
            nt = self.col.models.get(nt_id)  # type: ignore[arg-type]
            if nt is None:
                results.append({"id": nt_id, "status": "not_found"})
                continue

            use_count = self.col.models.use_count(nt)
            if use_count > 0:
                results.append(
                    {
                        "id": nt_id,
                        "name": nt["name"],
                        "status": "error",
                        "error": f"Cannot delete: {use_count} note(s) use this type",
                    }
                )
                continue

            self.col.models.remove(nt_id)  # type: ignore[arg-type]
            logger.debug("Deleted note type %s (%d)", nt["name"], nt_id)
            results.append({"id": nt_id, "name": nt["name"], "status": "deleted"})

        return {"results": results}
