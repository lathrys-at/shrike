"""Exact substring search (#86): substring_info helper + wrapper.search_substring."""

from __future__ import annotations

from shrike.collection import substring_info


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


class TestSubstringInfo:
    def test_basic_match(self):
        info = substring_info({"Front": "Electron transport chain", "Back": "x"}, "transport")
        assert info is not None
        assert info["matched_fields"] == ["Front"]
        assert "transport" in info["snippet"]

    def test_case_insensitive(self):
        assert substring_info({"Front": "Electron TRANSPORT"}, "transport") is not None
        assert substring_info({"Front": "electron transport"}, "TRANSPORT") is not None

    def test_no_match_is_none(self):
        assert substring_info({"Front": "alpha", "Back": "beta"}, "gamma") is None

    def test_multiple_fields(self):
        info = substring_info({"Front": "cell wall", "Back": "the cell"}, "cell")
        assert info["matched_fields"] == ["Front", "Back"]

    def test_snippet_has_ellipsis_for_long_value(self):
        val = "x" * 100 + "needle" + "y" * 100
        info = substring_info({"Front": val}, "needle")
        assert "needle" in info["snippet"]
        assert info["snippet"].startswith("…") and info["snippet"].endswith("…")
        assert len(info["snippet"]) < len(val)

    def test_empty_content(self):
        assert substring_info(None, "x") is None
        assert substring_info({}, "x") is None


class TestSearchSubstring:
    async def test_finds_and_annotates(self, wrapper):
        await _note(wrapper, "Sci", "Electron transport chain", "ATP")
        await _note(wrapper, "Sci", "Ribosome", "protein synthesis")
        hits = await wrapper.search_substring("transport")
        assert len(hits) == 1
        assert hits[0]["substring"]["matched_fields"] == ["Front"]
        assert "score" not in hits[0]

    async def test_case_insensitive(self, wrapper):
        await _note(wrapper, "Sci", "Mitochondria")
        assert len(await wrapper.search_substring("mitochondria")) == 1
        assert len(await wrapper.search_substring("MITO")) == 1

    async def test_special_chars_escaped(self, wrapper):
        # ':' must be escaped in the Anki query or a substring spanning it misses.
        await _note(wrapper, "Sci", "ATP synthase: makes ATP", "x")
        assert len(await wrapper.search_substring("synthase: makes")) == 1
        # literal wildcard chars are matched literally, not as wildcards
        await _note(wrapper, "Sci", "a*b_c", "x")
        assert len(await wrapper.search_substring("a*b_c")) == 1
        assert len(await wrapper.search_substring("axxb")) == 0

    async def test_no_match_empty(self, wrapper):
        await _note(wrapper, "Sci", "alpha")
        assert await wrapper.search_substring("zzz") == []

    async def test_deck_filter(self, wrapper):
        # Distinct first fields (upsert rejects duplicates), both containing "shared".
        await _note(wrapper, "Keep", "shared term in keep")
        await _note(wrapper, "Drop", "shared term in drop")
        hits = await wrapper.search_substring("shared", deck="Keep")
        assert len(hits) == 1
        assert hits[0]["deck"] == "Keep"

    async def test_deck_filter_by_hash_id(self, wrapper):
        await _note(wrapper, "ById", "deck ref term")
        deck_id = next(
            d["id"]
            for d in (await wrapper.get_collection_info(["decks"], []))["decks"]
            if d["name"] == "ById"
        )
        hits = await wrapper.search_substring("deck ref", deck=f"#{deck_id}")
        assert len(hits) == 1

    async def test_tag_filter(self, wrapper):
        await _note(wrapper, "Sci", "tagged term one", tags=["keep"])
        await _note(wrapper, "Sci", "tagged term two", tags=["other"])
        hits = await wrapper.search_substring("tagged", tags=["keep"])
        assert len(hits) == 1
        assert "keep" in hits[0]["tags"]

    async def test_exclude_ids(self, wrapper):
        a = await _note(wrapper, "Sci", "common word a")
        await _note(wrapper, "Sci", "common word b")
        hits = await wrapper.search_substring("common", exclude_ids=[a])
        assert all(h["id"] != a for h in hits)

    async def test_limit(self, wrapper):
        for i in range(5):
            await _note(wrapper, "Sci", f"repeat term {i}")
        hits = await wrapper.search_substring("repeat", limit=2)
        assert len(hits) == 2
