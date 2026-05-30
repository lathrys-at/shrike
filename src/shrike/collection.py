from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, TypeVar

from anki.collection import Collection
from anki.consts import MODEL_CLOZE
from anki.errors import NotFoundError

logger = logging.getLogger("shrike.collection")

T = TypeVar("T")


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
        query: str | None = None,
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
                query=query,
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
        query: str | None = None,
        fields_mode: str = "full",
        limit: int = 50,
    ) -> dict[str, Any]:
        if ids is not None and not any([deck, tags, note_type, modified_since, query]):
            return self._get_notes_by_ids(ids, fields_mode, limit)

        query_parts: list[str] = []
        if deck is not None:
            query_parts.append(f'"deck:{deck}"')
        if tags is not None:
            for tag in tags:
                if tag.startswith("-"):
                    query_parts.append(f"-tag:{tag[1:]}")
                else:
                    query_parts.append(f"tag:{tag}")
        if note_type is not None:
            query_parts.append(f'"note:{note_type}"')
        if query is not None:
            query_parts.append(query)

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

    async def upsert_notes(self, notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return await self.run(lambda _c: self._upsert_notes(notes))

    def _upsert_notes(self, notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for i, note_input in enumerate(notes):
            try:
                if "id" in note_input and note_input["id"] is not None:
                    results.append(self._update_note(note_input))
                else:
                    results.append(self._create_note(note_input))
            except Exception as e:
                results.append(
                    {
                        "status": "error",
                        "index": i,
                        "error": str(e),
                    }
                )
        return results

    def _create_note(self, note_input: dict[str, Any]) -> dict[str, Any]:
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
            raise ValueError(f"Note type '{note_type_name}' not found")

        deck_id = self.col.decks.id_for_name(deck_name)
        if deck_id is None:
            deck_id = self.col.decks.id(deck_name)

        if deck_id is None:
            raise ValueError(f"Could not find or create deck '{deck_name}'")

        note = self.col.new_note(notetype)
        for field_name, value in fields.items():
            if field_name not in note:
                raise ValueError(
                    f"Field '{field_name}' not found in note type '{note_type_name}'. "
                    f"Available fields: {list(note.keys())}"
                )
            note[field_name] = value

        if "tags" in note_input and note_input["tags"] is not None:
            note.tags = note_input["tags"]

        self.col.add_note(note, deck_id)
        logger.debug("Created note %d (type=%s, deck=%s)", note.id, note_type_name, deck_name)
        return {"status": "created", "id": note.id}

    def _update_note(self, note_input: dict[str, Any]) -> dict[str, Any]:
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
                    raise ValueError(
                        f"Field '{field_name}' not found in note type "
                        f"'{nt_name}'. Available fields: {list(note.keys())}"
                    )
                note[field_name] = value

        if "tags" in note_input and note_input["tags"] is not None:
            note.tags = note_input["tags"]

        self.col.update_note(note)

        if "deck" in note_input and note_input["deck"] is not None:
            target_deck_id = self.col.decks.id_for_name(note_input["deck"])
            if target_deck_id is None:
                target_deck_id = self.col.decks.id(note_input["deck"])
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

    # -- embedding text ------------------------------------------------------

    async def note_texts_for_embedding(self, note_ids: Sequence[int]) -> list[str]:
        """Return concatenated field text for each note, suitable for embedding.

        Notes that don't exist are returned as empty strings (same index position).
        """
        return await self.run(lambda c: self.note_texts(c, note_ids))

    @staticmethod
    def note_texts(col: Collection, note_ids: Sequence[int]) -> list[str]:
        """Concatenated field text for each note id. Must run on the worker thread."""
        texts: list[str] = []
        for nid in note_ids:
            try:
                note = col.get_note(nid)  # type: ignore[arg-type]
            except NotFoundError:
                texts.append("")
                continue
            parts = [f"{k}: {v}" for k, v in zip(note.keys(), note.values(), strict=False) if v]
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
