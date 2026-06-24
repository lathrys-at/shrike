"""Collection-layer find-and-replace."""

from __future__ import annotations


async def _note(wrapper, deck: str, front: str, back: str = "x", tags=None) -> int:
    results = await wrapper.upsert_notes(
        [
            {
                "deck": deck,
                "note_type": "Basic",
                "fields": {"Front": front, "Back": back},
                "tags": tags or [],
            }
        ]
    )
    return results[0]["id"]


async def _content(wrapper, nid: int) -> dict:
    return (await wrapper.list_notes(ids=[nid], fields_mode="full"))["notes"][0]["content"]


class TestFindReplace:
    async def test_literal_apply(self, wrapper):
        nid = await _note(wrapper, "Bio", "teh cell", "teh power")
        await _note(wrapper, "Bio", "unaffected", "y")
        res = await wrapper.find_replace("teh", "the", deck="Bio", dry_run=False)
        assert res["notes_changed"] == 1
        assert res["changed_ids"] == [nid]
        content = await _content(wrapper, nid)
        assert content["Front"] == "the cell"
        assert content["Back"] == "the power"

    async def test_dry_run_mutates_nothing(self, wrapper):
        nid = await _note(wrapper, "Bio", "teh cell")
        res = await wrapper.find_replace("teh", "the", deck="Bio", dry_run=True)
        assert res["notes_changed"] == 1
        assert res["changed_ids"] == []
        assert res["samples"][0]["before"] == "teh cell"
        assert res["samples"][0]["after"] == "the cell"
        assert (await _content(wrapper, nid))["Front"] == "teh cell"  # untouched

    async def test_match_case(self, wrapper):
        await _note(wrapper, "Bio", "Teh and teh")
        insensitive = await wrapper.find_replace("teh", "X", deck="Bio", dry_run=True)
        assert insensitive["samples"][0]["after"] == "X and X"
        sensitive = await wrapper.find_replace(
            "teh", "X", deck="Bio", match_case=True, dry_run=True
        )
        assert sensitive["samples"][0]["after"] == "Teh and X"

    async def test_field_restriction(self, wrapper):
        nid = await _note(wrapper, "Bio", "teh front", "teh back")
        await wrapper.find_replace("teh", "the", deck="Bio", field="Front", dry_run=False)
        content = await _content(wrapper, nid)
        assert content["Front"] == "the front"
        assert content["Back"] == "teh back"  # other field untouched

    async def test_regex(self, wrapper):
        nid = await _note(wrapper, "Bio", "colour and flavour")
        res = await wrapper.find_replace(
            "(colou?r|flavou?r)", "X", regex=True, deck="Bio", dry_run=False
        )
        assert res["notes_changed"] == 1
        assert (await _content(wrapper, nid))["Front"] == "X and X"

    async def test_regex_capture_uses_anki_dollar_refs(self, wrapper):
        nid = await _note(wrapper, "Bio", "2024-01-02")
        await wrapper.find_replace(
            r"(\d{4})-(\d{2})-(\d{2})", r"$3/$2/$1", regex=True, deck="Bio", dry_run=False
        )
        assert (await _content(wrapper, nid))["Front"] == "02/01/2024"

    async def test_scope_deck_by_hash_id(self, wrapper):
        nid = await _note(wrapper, "ById", "teh deck")
        await _note(wrapper, "Other", "teh other")
        deck_id = next(
            d["id"]
            for d in (await wrapper.get_collection_info(["decks"], []))["decks"]
            if d["name"] == "ById"
        )
        res = await wrapper.find_replace("teh", "the", deck=f"#{deck_id}", dry_run=False)
        assert res["changed_ids"] == [nid]

    async def test_scope_by_tags(self, wrapper):
        a = await _note(wrapper, "Bio", "teh a", tags=["fix"])
        await _note(wrapper, "Bio", "teh b", tags=["other"])
        res = await wrapper.find_replace("teh", "the", tags=["fix"], dry_run=False)
        assert res["changed_ids"] == [a]

    async def test_scope_by_ids(self, wrapper):
        a = await _note(wrapper, "Bio", "teh one")
        await _note(wrapper, "Bio", "teh two")
        res = await wrapper.find_replace("teh", "the", ids=[a], dry_run=False)
        assert res["changed_ids"] == [a]

    async def test_no_match(self, wrapper):
        await _note(wrapper, "Bio", "nothing here")
        res = await wrapper.find_replace("zzz", "x", deck="Bio", dry_run=False)
        assert res["notes_changed"] == 0
        assert res["changed_ids"] == []
        assert res["samples"] == []
