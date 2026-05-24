from __future__ import annotations

import atexit
import logging
from datetime import datetime, timezone
from typing import Any

import anki
from anki.collection import Collection
from anki.consts import MODEL_CLOZE
from anki.errors import NotFoundError

logger = logging.getLogger("shrike")


class CollectionWrapper:
    def __init__(self, path: str):
        self.col = Collection(path)
        atexit.register(self.close)

    def close(self):
        if self.col:
            try:
                self.col.close()
            except Exception:
                pass
            self.col = None

    def get_collection_info(
        self,
        include: list[str] | None = None,
        note_type_details: list[str] | None = None,
    ) -> dict[str, Any]:
        sections = include or ["note_types", "decks", "tags", "stats"]
        detail_names = set(note_type_details or [])
        result: dict[str, Any] = {}

        if "note_types" in sections:
            result["note_types"] = self._get_note_types(detail_names)

        if "decks" in sections:
            result["decks"] = self._get_decks()

        if "tags" in sections:
            result["tags"] = self.col.tags.all()

        if "stats" in sections:
            result["stats"] = self._get_stats()

        return result

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
                entry["templates"] = [
                    {
                        "name": t["name"],
                        "front": t["qfmt"],
                        "back": t["afmt"],
                    }
                    for t in nt["tmpls"]
                ]
                entry["css"] = nt.get("css", "")
            note_types.append(entry)
        return note_types

    def _get_decks(self) -> list[dict]:
        decks = []
        for name_id in self.col.decks.all_names_and_ids():
            note_ids = self.col.find_notes(f'"deck:{name_id.name}"')
            decks.append({
                "name": name_id.name,
                "id": name_id.id,
                "note_count": len(note_ids),
            })
        return decks

    def _get_stats(self) -> dict[str, Any]:
        tree = self.col.sched.deck_due_tree()

        total_due = 0
        total_new = 0
        decks_summary: dict[str, dict] = {}

        def walk(node, prefix=""):
            nonlocal total_due, total_new
            name = node.name
            if prefix:
                name = f"{prefix}::{node.name}"
            due = node.review_count + node.learn_count
            total_due += due
            total_new += node.new_count

            note_ids = self.col.find_notes(f'"deck:{name}"')
            decks_summary[name] = {
                "notes": len(note_ids),
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

    def list_notes(
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
                dt = dt.replace(tzinfo=timezone.utc)
            mod_cutoff = int(dt.timestamp())

        if mod_cutoff is not None:
            note_ids = [
                nid for nid in note_ids
                if self.col.get_note(nid).mod >= mod_cutoff
            ]

        total = len(note_ids)
        note_ids = note_ids[:limit]

        notes = [self._note_to_dict(nid, fields_mode) for nid in note_ids]
        return {"notes": notes, "total": total, "limit": limit}

    def _get_notes_by_ids(
        self, ids: list[int], fields_mode: str, limit: int
    ) -> dict[str, Any]:
        notes = []
        for nid in ids[:limit]:
            try:
                notes.append(self._note_to_dict(nid, fields_mode))
            except NotFoundError:
                continue
        return {"notes": notes, "total": len(notes), "limit": limit}

    def _note_to_dict(self, nid: int, fields_mode: str) -> dict[str, Any]:
        note = self.col.get_note(nid)
        notetype = self.col.models.get(note.mid)

        cards = note.cards()
        deck_id = cards[0].did if cards else None
        deck_name = self.col.decks.get(deck_id)["name"] if deck_id else "Default"

        result: dict[str, Any] = {
            "id": note.id,
            "note_type": notetype["name"],
            "deck": deck_name,
            "tags": note.tags,
            "modified": datetime.fromtimestamp(
                note.mod, tz=timezone.utc
            ).isoformat(),
        }

        if fields_mode == "full":
            result["content"] = dict(zip(note.keys(), note.values()))

        return result

    def upsert_notes(self, notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for i, note_input in enumerate(notes):
            try:
                if "id" in note_input and note_input["id"] is not None:
                    results.append(self._update_note(note_input))
                else:
                    results.append(self._create_note(note_input))
            except Exception as e:
                results.append({
                    "status": "error",
                    "index": i,
                    "error": str(e),
                })
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

        note = self.col.new_note(notetype)
        for field_name, value in fields.items():
            if field_name not in note.keys():
                raise ValueError(
                    f"Field '{field_name}' not found in note type '{note_type_name}'. "
                    f"Available fields: {list(note.keys())}"
                )
            note[field_name] = value

        if "tags" in note_input and note_input["tags"] is not None:
            note.tags = note_input["tags"]

        self.col.add_note(note, deck_id)
        return {"status": "created", "id": note.id}

    def _update_note(self, note_input: dict[str, Any]) -> dict[str, Any]:
        nid = note_input["id"]
        try:
            note = self.col.get_note(nid)
        except NotFoundError:
            raise ValueError(f"Note {nid} not found")

        if "note_type" in note_input and note_input["note_type"] is not None:
            current_type = self.col.models.get(note.mid)["name"]
            if note_input["note_type"] != current_type:
                raise ValueError(
                    f"Cannot change note type (current: '{current_type}', "
                    f"requested: '{note_input['note_type']}')"
                )

        if "fields" in note_input and note_input["fields"] is not None:
            for field_name, value in note_input["fields"].items():
                if field_name not in note.keys():
                    notetype = self.col.models.get(note.mid)
                    raise ValueError(
                        f"Field '{field_name}' not found in note type "
                        f"'{notetype['name']}'. Available fields: {list(note.keys())}"
                    )
                note[field_name] = value

        if "tags" in note_input and note_input["tags"] is not None:
            note.tags = note_input["tags"]

        self.col.update_note(note)

        if "deck" in note_input and note_input["deck"] is not None:
            deck_id = self.col.decks.id_for_name(note_input["deck"])
            if deck_id is None:
                deck_id = self.col.decks.id(note_input["deck"])
            card_ids = note.card_ids()
            self.col.set_deck(card_ids, deck_id)

        return {"status": "updated", "id": note.id}

    def delete_notes(self, ids: list[int]) -> dict[str, Any]:
        existing = set(self.col.find_notes(
            f"nid:{','.join(str(i) for i in ids)}"
        ))
        not_found = [i for i in ids if i not in existing]
        deleted = list(existing)

        if deleted:
            self.col.remove_notes(deleted)

        return {"deleted": deleted, "not_found": not_found}
