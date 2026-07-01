"""CollectionWrapper — the async facade over the NATIVE collection core.

Every collection operation runs in Rust (``shrike_native.CollectionCore``, anki
consumed exclusively through its protobuf service layer). This module is the
Python *harness half*: one dedicated worker thread serializing every access,
asyncio ergonomics for the HTTP host, the cooperative idle-release lifecycle,
the busy surface, and the response-shape glue the tool layer consumes. There is
no ``anki.Collection`` here — the pip ``anki`` package is a test-only oracle.

Threading model: the native core is internally synchronized (anki's Backend
mutex), but Shrike still routes every op through a single worker thread so
operations are *ordered*, the cooperative release/reopen lifecycle has one
owner, and the event loop never blocks. ``run``/``run_sync`` expose that thread;
``fn`` receives the native ``CollectionCore``.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import logging
import mimetypes
import os
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar

import shrike_native
from shrike_native import CollectionCore

from shrike.harness.engines.embedding.base import NoteEmbedInput
from shrike.observability.metrics import metrics
from shrike.schemas import COLLECTION_BUSY_CODE

logger = logging.getLogger("shrike.collection")

OnDuplicate = Literal["error", "skip", "allow"]

T = TypeVar("T")

# The ``collection`` metric label for the boot collection. A reserved sentinel,
# NOT "default": a routed registry profile may legitimately be named "default",
# and sharing the label would collapse two distinct collections (different files)
# into one series. Routed harnesses pass their profile name; only the boot
# collection takes this default — the cardinality contract keeps one series per
# collection.
BOOT_COLLECTION_KEY = "__boot__"

# Default seconds to hold the collection open after the last operation before
# releasing the lock, in cooperative mode. Near SQLite's conventional
# ``busy_timeout``; short because holding blocks launching Anki and re-opening is
# a cheap local SQLite open.
DEFAULT_LOCK_HOLD = 5.0


class CollectionBusyError(Exception):
    """The collection is held by another process (lock contention).

    Raised when a cooperative-mode re-acquire can't open the collection —
    typically because Anki desktop is open. Expected, not a bug; the server
    surfaces it over MCP with ``COLLECTION_BUSY_CODE`` and the client maps it
    back to a typed error, so callers catch-and-retry instead of parsing text.
    """

    def __init__(self) -> None:
        human = "the collection is in use by another process (is Anki open?); try again shortly"
        super().__init__(f"{COLLECTION_BUSY_CODE}: {human}")


def _safe_media_name(name: str) -> str:
    """Reduce a caller-supplied name to a bare basename inside the media dir.

    Strips any directory components, so ``../../etc/passwd`` becomes ``passwd`` and
    can only ever resolve inside the media dir — the path-traversal guard for
    fetch/delete. Returns "" for a name that is only separators/dots.
    """
    base = os.path.basename(name.replace("\\", "/").rstrip("/"))
    return "" if base in ("", ".", "..") else base


def _guess_mime(filename: str) -> str | None:
    """Best-effort MIME from a filename's extension (None for unknown)."""
    return mimetypes.guess_type(filename)[0]


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


# ── collection-thread collectors (module-level: the kernel's Scheduler port
# passes the native core straight through) ───────────────────────────────────


def collect_embed_inputs(core: CollectionCore) -> tuple[list[NoteEmbedInput], int]:
    """Every note's embedding input (text + image names) + the col_mod stamp."""
    note_ids = core.find_notes("deck:*")
    inputs = [
        NoteEmbedInput(note_id=nid, text=text, image_names=images)
        for nid, text, images in core.note_embed_inputs(note_ids)
    ]
    return inputs, core.col_mod()


class CollectionWrapper:
    """Serializes every access to the native collection core.

    One dedicated worker thread funnels all collection operations, so
    concurrent callers (event-loop tasks, custom HTTP routes, the startup
    path) are *ordered* rather than racing, and the cooperative idle-release
    lifecycle has a single owner. Operations are exposed as ``async``
    methods; ``run_sync`` and ``close`` serve the startup/shutdown paths.
    """

    def __init__(
        self,
        path: str,
        *,
        cooperative: bool = False,
        hold_seconds: float = DEFAULT_LOCK_HOLD,
        on_acquire: Callable[[CollectionCore], None] | None = None,
    ) -> None:
        # Absolutize once at construction (cwd is the daemon's startup dir): every
        # downstream use — opening the collection, the lock-free media_dir — is then
        # cwd-independent by construction, not merely by the daemon never chdir'ing.
        self._path = os.path.abspath(path)
        self._closed = False
        self._cooperative = cooperative
        self._hold = hold_seconds
        self._on_acquire = on_acquire
        self._open_flag = False
        self._release_handle: asyncio.TimerHandle | None = None
        self._metrics_key = BOOT_COLLECTION_KEY
        # Standalone mode (tests, library use): the wrapper owns the worker
        # thread. The server uses `over_kernel` instead, where the kernel's
        # injected executor is the one serialization domain.
        self._kernel: Any | None = None
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="shrike-collection"
        )
        # Open on the worker thread so the lifecycle is owned by the same thread
        # that orders every subsequent operation. (Cooperative mode opens at
        # boot too; the idle-release lifecycle takes over afterwards.)
        self._executor.submit(self._open).result()
        atexit.register(self.close)

    @classmethod
    def over_kernel(
        cls,
        kernel: Any,
        path: str,
        *,
        cooperative: bool = False,
        hold_seconds: float = DEFAULT_LOCK_HOLD,
        collection_key: str = BOOT_COLLECTION_KEY,
    ) -> CollectionWrapper:
        """The server's mode: wrap the kernel's OWN collection.

        ``core`` is the kernel's Arc-shared ``core_handle()`` and every op runs
        as one ``kernel.run_job`` on the kernel's injected executor — the same
        serialization domain the kernel's Rust ops use, so harness direct ops
        and kernel ops are ordered together. No thread is owned here;
        ``run_sync``/``release_now`` are unsupported (loop-free phases don't
        exist in this mode — assembly happens on the loop).
        """
        self = cls.__new__(cls)
        self._path = os.path.abspath(path)
        self._closed = False
        self._cooperative = cooperative
        self._hold = hold_seconds
        self._on_acquire = None
        self._open_flag = True
        self._release_handle = None
        self._kernel = kernel
        self._executor = None
        # The ``collection`` label for this wrapper's lock_held gauge — the boot
        # sentinel for the boot collection, the registry profile name for a routed one.
        self._metrics_key = collection_key
        self.core = kernel.core_handle()
        # The kernel opens its collection at boot — the lock is held from here,
        # so the gauge starts at 1 (not 0 until the first status/index op).
        metrics.lock_held.labels(self._metrics_key).set(1)
        atexit.register(self.close)
        return self

    def _open(self) -> None:
        logger.debug("Opening collection at %s", self._path)
        self.core = CollectionCore(self._path)
        self._open_flag = True
        # The lock is held from the initial open, not only after a re-acquire —
        # set the gauge here so the boot hold is covered, cleared on release/close.
        metrics.lock_held.labels(self._metrics_key).set(1)
        logger.debug("Collection opened successfully")

    def set_acquire_hook(self, hook: Callable[[CollectionCore], None] | None) -> None:
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

    @property
    def media_dir(self) -> str:
        """The media folder path, derived from the collection path — **lock-free**.

        Always ``<stem>.media`` next to the collection (the same derivation the
        native open uses), so the static media HTTP route can resolve files
        without acquiring the collection (no CollectionBusyError under
        cooperative locking). Absolute because ``self._path`` is absolutized in
        ``__init__``."""
        base = self._path[: -len(".anki2")] if self._path.endswith(".anki2") else self._path
        return base + ".media"

    # -- execution primitives ------------------------------------------------

    def _locked(self, fn: Callable[[CollectionCore], T]) -> T:
        """Run ``fn(core)`` on the worker thread, re-acquiring first if released.

        If cooperative mode released the collection, re-open it and run the
        acquire hook (drift re-check) before the op. In the default mode
        ``self._open_flag`` is always True, so this is just ``fn(self.core)``.
        A re-acquire that can't open (another process holds the file) surfaces
        as :class:`CollectionBusyError` — typed, immediate, retryable.
        """
        if not self._open_flag:
            started = time.perf_counter()
            try:
                self.core.reopen()
            except shrike_native.NativeBusyError as e:
                metrics.lock_attempts.labels("busy").inc()
                metrics.lock_wait.labels("busy").observe(time.perf_counter() - started)
                raise CollectionBusyError() from e
            metrics.lock_attempts.labels("acquired").inc()
            metrics.lock_wait.labels("acquired").observe(time.perf_counter() - started)
            self._open_flag = True
            metrics.lock_held.labels(self._metrics_key).set(1)
            logger.debug("Re-acquired collection at %s", self._path)
            if self._on_acquire is not None:
                with contextlib.suppress(Exception):
                    self._on_acquire(self.core)
        return fn(self.core)

    async def run(self, fn: Callable[[CollectionCore], T]) -> T:
        """Run ``fn(core)`` on the worker thread and await the result.

        The escape hatch for collection operations that don't have a dedicated
        method here. All access is serialized through the same single worker
        thread, which re-acquires the collection first if cooperative mode
        released it (``_locked``). In cooperative mode the idle-release timer
        is (re)armed after the op.
        """
        loop = asyncio.get_running_loop()
        try:
            if self._kernel is not None:
                # Kernel mode: ONE serialized job on the kernel's executor —
                # the same domain its own Rust ops run under (re-acquire +
                # busy mapping + acquire hook included, inside the job).
                return await self._kernel.run_job(lambda: self._locked(fn))  # type: ignore[no-any-return]
            assert self._executor is not None
            return await loop.run_in_executor(self._executor, lambda: self._locked(fn))
        finally:
            if self._cooperative and not self._closed:
                self._schedule_release(loop)

    def run_sync(self, fn: Callable[[CollectionCore], T]) -> T:
        """Run ``fn(core)`` on the worker thread, blocking until it returns.

        For the startup and shutdown paths, which run before/after the event
        loop and so cannot ``await``. Still routes through the worker thread
        (re-acquiring first if released), preserving the single-owner ordering.
        No idle-release is armed here — there is no loop; the next async op or
        an explicit ``release_now`` handles release. Standalone mode only: in
        kernel mode assembly runs on the loop, so loop-free phases don't exist.
        """
        if self._executor is None:
            raise RuntimeError("run_sync is unsupported in kernel mode; await run() instead")
        return self._executor.submit(lambda: self._locked(fn)).result()

    # -- cooperative idle-release --------------------------------------------

    def _schedule_release(self, loop: asyncio.AbstractEventLoop) -> None:
        """(Re)arm the idle-release timer; mirrors the index saver's debounce."""
        if self._release_handle is not None:
            self._release_handle.cancel()
        self._release_handle = loop.call_later(self._hold, self._fire_release)

    def _fire_release(self) -> None:
        self._release_handle = None
        # Release on the worker/executor; fire-and-forget so the loop never
        # blocks. A re-acquire racing this is safe — _release no-ops if already
        # re-opened and _locked re-opens before any access.
        if self._kernel is not None:
            # Loop callback context: a running loop exists, so the job
            # schedules; the returned future is deliberately dropped.
            self._kernel.run_job(self._release)
        else:
            assert self._executor is not None
            self._executor.submit(self._release)

    def _release(self) -> None:
        if not self._open_flag or self._closed:
            return
        with contextlib.suppress(Exception):
            self.core.release()
        self._open_flag = False
        metrics.lock_held.labels(self._metrics_key).set(0)
        logger.debug("Released collection lock after idle")

    def release_now(self) -> None:
        """Synchronously release the collection (close + mark released).

        For the boot path (release after the initial drift check so a
        never-touched idle daemon doesn't hold the lock) and tests.
        Standalone mode only; kernel mode releases via ``kernel.release()``.
        """
        if self._executor is None:
            raise RuntimeError("release_now is unsupported in kernel mode")
        self._executor.submit(self._release).result()

    async def reopen(self) -> None:
        """Close and re-open the collection on the worker thread.

        Releases Anki's SQLite lock and re-opens the file, picking up changes made
        on disk underneath the daemon (a restored backup, a file-level sync/swap).
        Serialized against every other operation — nothing sees a half-swapped
        handle. The index is not touched here; the caller re-checks drift after.
        """
        await self.run(self._do_reopen)

    def _do_reopen(self, core: CollectionCore) -> None:
        # The native reopen maps contention to the busy tier; /reload on a
        # collection another process holds should surface exactly that.
        try:
            core.reopen()
        except shrike_native.NativeBusyError as e:
            raise CollectionBusyError() from e
        self._open_flag = True
        metrics.lock_held.labels(self._metrics_key).set(1)
        logger.info("Reopened collection at %s", self._path)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._release_handle is not None:
            self._release_handle.cancel()
            self._release_handle = None
        if self._executor is None:
            # Kernel mode: the kernel owns the collection — its close()
            # (driven by the server's shutdown path) closes the core.
            self._open_flag = False
            metrics.lock_held.labels(self._metrics_key).set(0)
            return
        logger.debug("Closing collection")
        with contextlib.suppress(Exception):
            self._executor.submit(self.core.close).result()
        self._open_flag = False
        metrics.lock_held.labels(self._metrics_key).set(0)
        self._executor.shutdown(wait=True)

    # -- info / read -----------------------------------------------------------

    async def col_mod(self) -> int:
        """The collection-modified watermark (drift detection's anchor)."""
        return await self.run(lambda c: c.col_mod())

    async def get_collection_info(
        self,
        include: list[str] | None = None,
        note_type_details: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self.run(
            lambda c: json.loads(c.collection_info(include or [], note_type_details or []))
        )

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
        cutoff: int | None = None
        if modified_since is not None:
            try:
                dt = datetime.fromisoformat(modified_since)
            except ValueError as e:
                # Caller-supplied bad input: raise a clean, non-leaky ValueError
                # the tool layer turns into a ToolInputError, not
                # fromisoformat's "Invalid isoformat string" leak via the
                # catch-all "Unhandled error" + traceback.
                raise ValueError(
                    f"`modified_since` is not a valid ISO 8601 datetime: {modified_since!r}"
                ) from e
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            cutoff = int(dt.timestamp())

        def _list(c: CollectionCore) -> dict[str, Any]:
            try:
                listed: dict[str, Any] = json.loads(
                    c.list_notes(
                        ids=ids,
                        deck=deck,
                        tags=tags,
                        note_type=note_type,
                        modified_since=cutoff,
                        with_fields=fields_mode == "full",
                        limit=limit,
                    )
                )
                return listed
            except shrike_native.NativeInputError as e:
                if "filter" in str(e):
                    # The tool layer's contract for the no-filter case.
                    return {"error": "At least one filter is required"}
                raise

        return await self.run(_list)

    async def query(
        self, query: str, *, fields_mode: str = "full", limit: int = 50
    ) -> dict[str, Any]:
        """Run a raw Anki search expression and return matching notes.

        The full Anki search language, no structured filters. A malformed
        expression raises the native input error (a ``ValueError``), which the
        tool layer turns into a ``ToolInputError``. Same shape as ``list_notes``
        (``total`` is the full match count before ``limit``).
        """
        return await self.run(
            lambda c: json.loads(c.query(query, with_fields=fields_mode == "full", limit=limit))
        )

    async def note_to_dict(self, nid: int, fields_mode: str) -> dict[str, Any]:
        def _one(c: CollectionCore) -> dict[str, Any]:
            listed = json.loads(c.list_notes(ids=[nid], with_fields=fields_mode == "full"))
            notes = listed["notes"]
            if not notes:
                raise ValueError(f"Note {nid} not found")
            return notes[0]  # type: ignore[no-any-return]

        return await self.run(_one)

    async def notes_by_id(self, nids: list[int], fields_mode: str) -> dict[int, dict[str, Any]]:
        """Batch ``note_to_dict``: ONE collection job for the whole id set.
        Missing ids are simply absent from the map."""
        if not nids:
            return {}

        def _many(core: CollectionCore) -> dict[int, dict[str, Any]]:
            listed = json.loads(
                core.list_notes(ids=nids, with_fields=fields_mode == "full", limit=len(nids))
            )
            return {n["id"]: n for n in listed["notes"]}

        return await self.run(_many)

    async def search_substring(
        self,
        text: str,
        *,
        deck: str | None = None,
        tags: list[str] | None = None,
        exclude_ids: list[int] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Notes whose field text contains ``text`` literally (case-insensitive).

        Anki's ``"*term*"`` wildcard search is a fast pre-filter; each candidate
        is then confirmed (and annotated) by ``substring_info``, the authority.
        Returns full note dicts each carrying a ``substring`` annotation.
        """

        def _search(c: CollectionCore) -> list[dict[str, Any]]:
            if not text.strip():
                return []
            parts = [f'"*{_escape_anki_text(text)}*"']
            if deck is not None:
                resolved = c.resolve_deck_ref(deck)
                if resolved is None:
                    return []
                parts.append(f'"deck:{resolved}"')
            for tag in tags or []:
                parts.append(f'"tag:{tag}"')

            exclude = set(exclude_ids or [])
            candidates = [nid for nid in c.find_notes(" ".join(parts)) if nid not in exclude]
            results: list[dict[str, Any]] = []
            for note in json.loads(c.list_notes(ids=candidates, limit=len(candidates) or 1))[
                "notes"
            ]:
                info = substring_info(note.get("content"), text)
                if info is None:
                    continue  # Anki matched across markup/normalization; not a literal hit
                note["substring"] = info
                results.append(note)
                if len(results) >= limit:
                    break
            return results

        return await self.run(_search)

    async def resolve_deck_ref(self, ref: str) -> str | None:
        return await self.run(lambda c: c.resolve_deck_ref(ref))

    # -- write ------------------------------------------------------------------

    async def upsert_notes(
        self,
        notes: list[dict[str, Any]],
        *,
        on_duplicate: OnDuplicate = "error",
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        return await self.run(
            lambda c: json.loads(c.upsert_notes(json.dumps(notes), on_duplicate, dry_run))
        )

    async def delete_notes(self, ids: list[int]) -> dict[str, Any]:
        def _delete(c: CollectionCore) -> dict[str, Any]:
            existing = set(c.find_notes(f"nid:{','.join(str(i) for i in ids)}")) if ids else set()
            not_found = [i for i in ids if i not in existing]
            deleted = [i for i in ids if i in existing]
            if deleted:
                c.delete_notes(deleted)
            return {"deleted": deleted, "not_found": not_found}

        return await self.run(_delete)

    async def update_note_tags(
        self,
        note_ids: list[int],
        *,
        set_tags: list[str] | None,
        add: list[str],
        remove: list[str],
    ) -> dict[str, Any]:
        return await self.run(
            lambda c: json.loads(
                c.update_note_tags(note_ids, set_tags=set_tags, add=add, remove=remove)
            )
        )

    async def rename_tag(self, old: str, new: str, note_ids: list[int]) -> dict[str, Any]:
        return await self.run(lambda c: json.loads(c.rename_tag(old, new, note_ids)))

    async def upsert_decks(self, decks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return await self.run(lambda c: json.loads(c.upsert_decks(json.dumps(decks))))

    async def delete_decks(self, names: list[str]) -> dict[str, Any]:
        return await self.run(lambda c: json.loads(c.delete_decks(names)))

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
        def _replace(c: CollectionCore) -> dict[str, Any]:
            # Scope resolution mirrors list_notes' structured filters.
            parts: list[str] = []
            if deck is not None:
                resolved = c.resolve_deck_ref(deck)
                if resolved is None:
                    return {
                        "notes_changed": 0,
                        "dry_run": dry_run,
                        "samples": [],
                        "changed_ids": [],
                    }
                parts.append(f'"deck:{resolved}"')
            for tag in tags or []:
                parts.append(f"-tag:{tag[1:]}" if tag.startswith("-") else f"tag:{tag}")
            if note_type is not None:
                parts.append(f'"note:{note_type}"')
            combined = " ".join(parts) if parts else None
            if ids is not None:
                id_query = f"nid:{','.join(str(i) for i in ids)}"
                combined = f"{id_query} {combined}" if combined else id_query
            if combined is None:
                raise ValueError("A scope is required: provide deck, tags, note_type, or ids.")
            candidates = c.find_notes(combined)
            if not candidates:
                return {"notes_changed": 0, "dry_run": dry_run, "samples": [], "changed_ids": []}

            # Preview in Python (exact for literal, illustrative for regex).
            samples: list[dict[str, Any]] = []
            predicted: set[int] = set()
            for nid, names, values in c.note_field_map(candidates):
                for fname, value in zip(names, values, strict=False):
                    if field is not None and fname != field:
                        continue
                    new = apply_replacement(
                        value, search, replacement, regex=regex, match_case=match_case
                    )
                    if new != value:
                        predicted.add(nid)
                        if len(samples) < sample_limit:
                            samples.append(
                                {"id": nid, "field": fname, "before": value, "after": new}
                            )

            if dry_run:
                return {
                    "notes_changed": len(predicted),
                    "dry_run": True,
                    "samples": samples,
                    "changed_ids": [],
                }

            notes_changed, changed_ids = c.find_replace_notes(
                candidates, search, replacement, regex, match_case, field
            )
            return {
                "notes_changed": notes_changed,
                "dry_run": False,
                "samples": samples,
                "changed_ids": changed_ids,
            }

        return await self.run(_replace)

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
            lambda c: json.loads(
                c.migrate_note_type(
                    note_ids,
                    new_note_type,
                    json.dumps(field_map),
                    json.dumps(template_map) if template_map else "",
                    dry_run,
                )
            )
        )

    # -- note types ---------------------------------------------------------------

    async def upsert_note_types(self, note_types: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return await self.run(lambda c: json.loads(c.upsert_note_types(json.dumps(note_types))))

    async def update_note_type_fields(
        self, note_type_name: str, operations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return await self.run(
            lambda c: json.loads(c.update_note_type_fields(note_type_name, json.dumps(operations)))
        )

    async def update_note_type_templates(
        self, note_type_name: str, operations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return await self.run(
            lambda c: json.loads(
                c.update_note_type_templates(note_type_name, json.dumps(operations))
            )
        )

    async def find_replace_note_types(
        self,
        note_type_name: str,
        *,
        search: str,
        replacement: str,
        regex: bool = False,
        match_case: bool = True,
        front: bool = True,
        back: bool = True,
        css: bool = True,
    ) -> dict[str, Any]:
        return await self.run(
            lambda c: json.loads(
                c.find_replace_note_types(
                    note_type_name, search, replacement, regex, match_case, front, back, css
                )
            )
        )

    async def update_note_type_field_metadata(
        self, note_type_name: str, updates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return await self.run(
            lambda c: json.loads(
                c.update_note_type_field_metadata(note_type_name, json.dumps(updates))
            )
        )

    async def delete_note_types(self, ids: list[int]) -> dict[str, Any]:
        return await self.run(lambda c: json.loads(c.delete_note_types(ids)))

    # -- maintenance ----------------------------------------------------------------

    async def prune(
        self,
        *,
        unused_tags: bool,
        empty_notes: bool,
        empty_cards: bool,
        unused_media: bool,
        dry_run: bool,
    ) -> tuple[dict[str, Any], list[int]]:
        def _prune(c: CollectionCore) -> tuple[dict[str, Any], list[int]]:
            # The binding hands removed_note_ids out of band (kernel-internal,
            # never part of the response wire).
            result_json, removed = c.prune(
                unused_tags, empty_notes, empty_cards, unused_media, dry_run
            )
            return json.loads(result_json), removed

        return await self.run(_prune)

    # -- media ---------------------------------------------------------------------

    async def store_media(
        self,
        items: list[dict[str, Any]],
        *,
        allow_private_fetch: bool,
        server_path_roots: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Store a batch of media files; one bad item never sinks the batch.

        The STANDALONE wrapper path (tests; the server's store_media action
        rides the kernel's op, which owns the concurrent prepare on its
        blocking pool). Each item is prepared off the worker thread and
        concurrently (``asyncio.gather`` over ``to_thread`` of the NATIVE
        fetch/decode); the prepared bytes are then written on the worker
        thread; server-local ``path`` items go through the native store whole
        (its containment gates are authoritative).
        """
        roots = server_path_roots or []

        async def _prepare(index: int, item: dict[str, Any]) -> dict[str, Any]:
            name = item.get("filename")
            try:
                if item.get("path") is not None:
                    # Native handles the roots gate + containment + zero-copy.
                    return {"index": index, "path_item": item, "name": name}
                if item.get("data") is not None:
                    raw = await asyncio.to_thread(shrike_native.decode_media_b64, item["data"])
                    return {"index": index, "raw": raw, "name": name, "content_type": None}
                raw, content_type = await asyncio.to_thread(
                    shrike_native.fetch_media_url, item["url"], allow_private_fetch
                )
                from urllib.parse import urlparse

                return {
                    "index": index,
                    "raw": raw,
                    "name": name or _safe_media_name(urlparse(item["url"]).path),
                    "content_type": content_type,
                }
            except Exception as e:
                return {"index": index, "error": str(e), "name": name}

        prepared = await asyncio.gather(*(_prepare(i, item) for i, item in enumerate(items)))

        def _write(c: CollectionCore) -> list[dict[str, Any]]:
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
                    if "path_item" in p:
                        out = json.loads(
                            c.store_media_items(
                                json.dumps([p["path_item"]]),
                                allow_private_fetch,
                                roots,
                            )
                        )[0]
                    else:
                        out = json.loads(
                            c.store_media_bytes(
                                bytes(p["raw"]),
                                filename=p["name"] or None,
                                content_type=p["content_type"],
                            )
                        )
                    out["index"] = p["index"]
                    results.append(out)
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

        return await self.run(_write)

    async def fetch_media(self, filenames: list[str]) -> list[dict[str, Any]]:
        return await self.run(lambda c: json.loads(c.fetch_media(filenames)))

    async def list_media(self, *, pattern: str | None, limit: int | None) -> dict[str, Any]:
        return await self.run(lambda c: json.loads(c.list_media(pattern, limit)))

    async def delete_media(self, filenames: list[str]) -> dict[str, Any]:
        return await self.run(lambda c: json.loads(c.delete_media(filenames)))

    async def media_check(self) -> dict[str, Any]:
        return await self.run(lambda c: json.loads(c.media_check()))

    # -- embedding text -------------------------------------------------------------

    async def note_texts_for_embedding(self, note_ids: Sequence[int]) -> list[str]:
        """Normalized embedding text per note id ("" at missing positions)."""
        ids = list(note_ids)
        return await self.run(lambda c: c.note_texts(ids))

    async def note_embed_inputs(self, note_ids: Sequence[int]) -> list[NoteEmbedInput]:
        """Each note's embedding input — normalized text + image filenames."""
        ids = list(note_ids)
        return await self.run(
            lambda c: [
                NoteEmbedInput(note_id=nid, text=text, image_names=images)
                for nid, text, images in c.note_embed_inputs(ids)
            ]
        )

    async def note_field_map(self, note_ids: Sequence[int]) -> dict[int, dict[str, str]]:
        """Per note: its **raw** field values as ``{field_name: value}``."""
        ids = list(note_ids)
        return await self.run(
            lambda c: {
                nid: dict(zip(names, values, strict=False))
                for nid, names, values in c.note_field_map(ids)
            }
        )

    async def derived_rows(self, note_ids: Sequence[int]) -> list[tuple[int, str, str, str]]:
        """``(note_id, "field", field_name, raw_value)`` rows for the derived store."""
        ids = list(note_ids)
        return await self.run(lambda c: c.derived_field_rows(ids))
