"""Collection-layer deck lifecycle: upsert create/rename, empty-only delete."""

from __future__ import annotations

import pytest
import shrike_native


async def _deck_names(wrapper) -> set[str]:
    import json

    return {
        d["name"]
        for d in wrapper.run_sync(lambda c: json.loads(c.collection_info(["decks"], []))["decks"])
    }


async def _make_note(wrapper, deck: str) -> int:
    results = await wrapper.upsert_notes(
        [
            {
                "deck": deck,
                "note_type": "Basic",
                "fields": {"Front": "Q", "Back": "A"},
            }
        ]
    )
    return results[0]["id"]


class TestUpsertDecks:
    async def test_create_new(self, wrapper):
        results = await wrapper.upsert_decks([{"name": "Japanese::Vocab"}])
        assert results[0]["status"] == "created"
        assert results[0]["name"] == "Japanese::Vocab"
        assert "Japanese::Vocab" in await _deck_names(wrapper)

    async def test_create_existing_is_updated_noop(self, wrapper):
        first = await wrapper.upsert_decks([{"name": "Dup"}])
        again = await wrapper.upsert_decks([{"name": "Dup"}])
        assert first[0]["status"] == "created"
        assert again[0]["status"] == "updated"
        assert again[0]["id"] == first[0]["id"]

    async def test_rename_by_id(self, wrapper):
        created = await wrapper.upsert_decks([{"name": "Old"}])
        did = created[0]["id"]
        renamed = await wrapper.upsert_decks([{"id": did, "name": "New"}])
        assert renamed[0]["status"] == "updated"
        assert renamed[0]["id"] == did
        names = await _deck_names(wrapper)
        assert "New" in names and "Old" not in names

    async def test_reparent(self, wrapper):
        created = await wrapper.upsert_decks([{"name": "Loose"}])
        did = created[0]["id"]
        await wrapper.upsert_decks([{"id": did, "name": "Parent::Loose"}])
        assert "Parent::Loose" in await _deck_names(wrapper)

    async def test_rename_onto_existing_name_errors(self, wrapper):
        # Decks don't merge; renaming onto a name another deck uses is rejected
        # (rather than Anki's silent "B+" disambiguation). Both decks survive.
        a = await wrapper.upsert_decks([{"name": "A"}])
        await wrapper.upsert_decks([{"name": "B"}])
        results = await wrapper.upsert_decks([{"id": a[0]["id"], "name": "B"}])
        assert results[0]["status"] == "error"
        assert "already exists" in results[0]["error"]
        names = await _deck_names(wrapper)
        assert {"A", "B"} <= names

    async def test_rename_to_own_name_is_noop_ok(self, wrapper):
        created = await wrapper.upsert_decks([{"name": "Same"}])
        results = await wrapper.upsert_decks([{"id": created[0]["id"], "name": "Same"}])
        assert results[0]["status"] == "updated"

    async def test_unknown_id_is_error_item(self, wrapper):
        results = await wrapper.upsert_decks([{"id": 9999999999999, "name": "X"}])
        assert results[0]["status"] == "error"
        assert "index" in results[0]

    async def test_missing_name_is_error_item(self, wrapper):
        # Typed input: a name-LESS item rejects at the binding parse (never
        # legal — the tool layer's Pydantic DeckInput requires name); an EMPTY
        # name stays the per-item error.
        with pytest.raises(shrike_native.NativeInputError, match="decks must be a JSON list"):
            await wrapper.upsert_decks([{}])
        results = await wrapper.upsert_decks([{"name": ""}])
        assert results[0]["status"] == "error"
        assert "name is required" in results[0]["error"]

    async def test_batch_isolates_failures(self, wrapper):
        results = await wrapper.upsert_decks(
            [{"name": "Good"}, {"id": 9999999999999, "name": "Bad"}]
        )
        assert results[0]["status"] == "created"
        assert results[1]["status"] == "error"


class TestDeleteDecks:
    async def test_delete_empty(self, wrapper):
        await wrapper.upsert_decks([{"name": "Empty"}])
        result = await wrapper.delete_decks(["Empty"])
        assert result["deleted"] == ["Empty"]
        assert "Empty" not in await _deck_names(wrapper)

    async def test_refuse_when_deck_has_cards(self, wrapper):
        await _make_note(wrapper, "Full")
        result = await wrapper.delete_decks(["Full"])
        assert result["not_empty"] == ["Full"]
        assert result["deleted"] == []
        assert "Full" in await _deck_names(wrapper)

    async def test_refuse_when_subdeck_has_cards(self, wrapper):
        await _make_note(wrapper, "Parent::Child")
        # Parent itself holds no cards, but a subdeck does → refuse.
        result = await wrapper.delete_decks(["Parent"])
        assert result["not_empty"] == ["Parent"]
        assert "Parent" in await _deck_names(wrapper)

    async def test_not_found(self, wrapper):
        result = await wrapper.delete_decks(["Nope"])
        assert result["not_found"] == ["Nope"]

    async def test_mixed(self, wrapper):
        await wrapper.upsert_decks([{"name": "GoneSoon"}])
        await _make_note(wrapper, "Keep")
        result = await wrapper.delete_decks(["GoneSoon", "Keep", "Missing"])
        assert result["deleted"] == ["GoneSoon"]
        assert result["not_empty"] == ["Keep"]
        assert result["not_found"] == ["Missing"]
