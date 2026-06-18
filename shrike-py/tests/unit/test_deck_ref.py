"""Deck references by ID: name / numeric id / #id resolution and wiring."""

from __future__ import annotations


async def _deck_id(wrapper, name: str) -> int:
    results = await wrapper.upsert_decks([{"name": name}])
    return results[0]["id"]


async def _note_in(wrapper, deck: str) -> int:
    results = await wrapper.upsert_notes(
        [{"deck": deck, "note_type": "Basic", "fields": {"Front": "q", "Back": "a"}}]
    )
    return results[0]["id"]


class TestResolveDeckRef:
    async def test_name_passthrough(self, wrapper):
        assert await wrapper.resolve_deck_ref("Anything") == "Anything"

    async def test_numeric_existing_id(self, wrapper):
        did = await _deck_id(wrapper, "Math")
        assert await wrapper.resolve_deck_ref(str(did)) == "Math"

    async def test_hash_id(self, wrapper):
        did = await _deck_id(wrapper, "Science")
        assert await wrapper.resolve_deck_ref(f"#{did}") == "Science"

    async def test_hash_id_missing_is_none(self, wrapper):
        assert await wrapper.resolve_deck_ref("#9999999999999") is None

    async def test_bare_numeric_missing_falls_back_to_name(self, wrapper):
        # No deck has this id, so it's treated as a literal name.
        assert await wrapper.resolve_deck_ref("123456") == "123456"

    async def test_hash_non_numeric_is_name(self, wrapper):
        assert await wrapper.resolve_deck_ref("#notanid") == "#notanid"


class TestListByDeckId:
    async def test_list_by_numeric_and_hash_id(self, wrapper):
        did = await _deck_id(wrapper, "ListByID")
        await _note_in(wrapper, "ListByID")
        assert (await wrapper.list_notes(deck=str(did)))["total"] == 1
        assert (await wrapper.list_notes(deck=f"#{did}"))["total"] == 1

    async def test_list_by_missing_hash_id_is_empty(self, wrapper):
        result = await wrapper.list_notes(deck="#9999999999999")
        assert result["total"] == 0
        assert result["notes"] == []


class TestCreateUpdateByDeckId:
    async def test_create_into_deck_by_hash_id(self, wrapper):
        did = await _deck_id(wrapper, "Target")
        results = await wrapper.upsert_notes(
            [{"deck": f"#{did}", "note_type": "Basic", "fields": {"Front": "q", "Back": "a"}}]
        )
        assert results[0]["status"] == "created"
        note = await wrapper.note_to_dict(results[0]["id"], "meta")
        assert note["deck"] == "Target"

    async def test_create_with_unknown_hash_id_errors(self, wrapper):
        results = await wrapper.upsert_notes(
            [
                {
                    "deck": "#9999999999999",
                    "note_type": "Basic",
                    "fields": {"Front": "q", "Back": "a"},
                }
            ]
        )
        assert results[0]["status"] == "error"
        assert "not found" in results[0]["error"].lower()

    async def test_update_moves_by_hash_id(self, wrapper):
        nid = await _note_in(wrapper, "Src")
        dst = await _deck_id(wrapper, "Dst")
        await wrapper.upsert_notes([{"id": nid, "deck": f"#{dst}"}])
        assert (await wrapper.note_to_dict(nid, "meta"))["deck"] == "Dst"


class TestDeleteDecksByID:
    async def test_delete_by_hash_id_echoes_ref(self, wrapper):
        did = await _deck_id(wrapper, "Empty")
        result = await wrapper.delete_decks([f"#{did}"])
        assert result["deleted"] == [f"#{did}"]

    async def test_delete_missing_hash_id_not_found(self, wrapper):
        result = await wrapper.delete_decks(["#9999999999999"])
        assert result["not_found"] == ["#9999999999999"]

    async def test_delete_non_empty_by_id_refused(self, wrapper):
        did = await _deck_id(wrapper, "Full")
        await _note_in(wrapper, "Full")
        result = await wrapper.delete_decks([str(did)])
        assert result["not_empty"] == [str(did)]
