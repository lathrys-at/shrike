from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar

from anki.collection import Collection
from anki.consts import MODEL_CLOZE
from anki.errors import NotFoundError
from anki.notes import NoteFieldsCheckResult

from shrike.embed_text import normalize_for_embedding

logger = logging.getLogger("shrike.collection")

OnDuplicate = Literal["error", "skip", "allow"]

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

# Chars that are special inside an Anki search even within double quotes; escaped
# so the term is matched literally (verified against anki 25.9.4: ':' must be
# escaped even quoted, or a substring spanning it silently misses).
_ANKI_SEARCH_SPECIALS = ("\\", '"', "*", "_", ":")


def _escape_anki_text(text: str) -> str:
    for ch in _ANKI_SEARCH_SPECIALS:
        text = text.replace(ch, "\\" + ch)
    return text


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
    """

    # Assigned on the worker thread in ``_open``; declared here for the type
    # checker. Never mutate or call methods on it outside the worker thread.
    col: Collection

    def __init__(self, path: str) -> None:
        self._path = path
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shrike-collection")
        # Open on the worker thread so the backend is owned by the same thread
        # that will service every subsequent operation.
        self._executor.submit(self._open).result()
        atexit.register(self.close)

    def _open(self) -> None:
        logger.debug("Opening collection at %s", self._path)
        self.col = Collection(self._path)
        logger.debug("Collection opened successfully")

    # -- execution primitives ------------------------------------------------

    async def run(self, fn: Callable[[Collection], T]) -> T:
        """Run ``fn(col)`` on the worker thread and await the result.

        The escape hatch for collection operations that don't have a dedicated
        method here (e.g. note-type edits, reading ``col.mod``). All access is
        serialized through the same single worker thread.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, self.col)

    def run_sync(self, fn: Callable[[Collection], T]) -> T:
        """Run ``fn(col)`` on the worker thread, blocking until it returns.

        For the startup and shutdown paths, which run before/after the event
        loop and so cannot ``await``. Still routes through the worker thread,
        preserving single-thread ownership of the collection.
        """
        return self._executor.submit(fn, self.col).result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
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

        if "summary" in sections:
            result["summary"] = self._get_summary()

        if "note_types" in sections:
            result["note_types"] = self._get_note_types(detail_names)

        if "decks" in sections:
            result["decks"] = self._get_decks()

        if "tags" in sections:
            result["tags"] = self.col.tags.all()

        if "stats" in sections:
            result["stats"] = self._get_stats()

        return result

    def _get_summary(self) -> dict[str, Any]:
        tree = self.col.sched.deck_due_tree()
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
        nids_by_deck: dict[int, set[int]] = defaultdict(set)
        for nid, did, odid in db.all("select nid, did, odid from cards"):
            nids_by_deck[did].add(nid)
            if odid:
                nids_by_deck[odid].add(nid)

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

    def _get_decks(self) -> list[dict]:
        counts = self._note_counts_by_deck()
        return [
            {
                "name": name_id.name,
                "id": name_id.id,
                "note_count": counts.get(name_id.name, 0),
            }
            for name_id in self.col.decks.all_names_and_ids()
        ]

    def _get_stats(self) -> dict[str, Any]:
        tree = self.col.sched.deck_due_tree()

        # Top-level node counts already include their subdecks, so collection
        # totals sum over top-level decks only. The per-deck summary below
        # walks every node — each node's count is its own rolled-up total.
        total_due = sum(top.review_count + top.learn_count for top in tree.children)
        total_new = sum(top.new_count for top in tree.children)

        note_counts = self._note_counts_by_deck()
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

        query_parts: list[str] = []
        if deck is not None:
            resolved_deck = self._resolve_deck_ref(deck)
            if resolved_deck is None:
                # An explicit #id (or numeric id) that matches no deck → no hits.
                return {"notes": [], "total": 0, "limit": limit}
            query_parts.append(f'"deck:{resolved_deck}"')
        if tags is not None:
            for tag in tags:
                if tag.startswith("-"):
                    query_parts.append(f"-tag:{tag[1:]}")
                else:
                    query_parts.append(f"tag:{tag}")
        if note_type is not None:
            query_parts.append(f'"note:{note_type}"')

        combined = " ".join(query_parts) if query_parts else None

        if ids is not None:
            id_query = f"nid:{','.join(str(i) for i in ids)}"
            combined = f"{id_query} {combined}" if combined else id_query

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

        notes = [self._note_to_dict(nid, fields_mode) for nid in note_ids]
        return {"notes": notes, "total": total, "limit": limit}

    def _get_notes_by_ids(self, ids: list[int], fields_mode: str, limit: int) -> dict[str, Any]:
        notes = []
        for nid in ids[:limit]:
            try:
                notes.append(self._note_to_dict(nid, fields_mode))
            except NotFoundError:
                continue
        return {"notes": notes, "total": len(notes), "limit": limit}

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

    async def note_to_dict(self, nid: int, fields_mode: str) -> dict[str, Any]:
        return await self.run(lambda _c: self._note_to_dict(nid, fields_mode))

    def _note_to_dict(self, nid: int, fields_mode: str) -> dict[str, Any]:
        note = self.col.get_note(nid)  # type: ignore[arg-type]
        notetype = self.col.models.get(note.mid)

        # Anki permits per-card decks; a note's cards can live in different
        # decks. We report the first card's deck — adequate for Shrike's
        # one-deck-per-note model, but it does not represent split-deck notes.
        cards = note.cards()
        deck_id = cards[0].did if cards else None
        deck_obj = self.col.decks.get(deck_id) if deck_id else None
        deck_name = deck_obj["name"] if deck_obj else "Default"

        result: dict[str, Any] = {
            "id": note.id,
            "note_type": notetype["name"] if notetype else "Unknown",
            "deck": deck_name,
            "tags": note.tags,
            "modified": datetime.fromtimestamp(note.mod, tz=UTC).isoformat(),
        }

        if fields_mode == "full":
            result["content"] = dict(zip(note.keys(), note.values(), strict=False))

        return result

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
        """
        texts: list[str] = []
        for nid in note_ids:
            try:
                note = col.get_note(nid)  # type: ignore[arg-type]
            except NotFoundError:
                texts.append("")
                continue
            parts = []
            for k, v in zip(note.keys(), note.values(), strict=False):
                cleaned = normalize_for_embedding(v)
                if cleaned:
                    parts.append(f"{k}: {cleaned}")
            texts.append("\n".join(parts))
        return texts

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
