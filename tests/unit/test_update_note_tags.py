"""Collection-layer tag operations (#73): set/add/remove, rename, clear-unused."""

from __future__ import annotations

import pytest


async def _tags(wrapper, nid: int) -> list[str]:
    result = await wrapper.list_notes(ids=[nid], fields_mode="meta")
    return sorted(result["notes"][0]["tags"])


async def _make_note(wrapper, front: str, tags: list[str]) -> int:
    results = await wrapper.upsert_notes(
        [
            {
                "deck": "Test",
                "note_type": "Basic",
                "fields": {"Front": front, "Back": "x"},
                "tags": tags,
            }
        ]
    )
    return results[0]["id"]


class TestUpdateNoteTags:
    async def test_set_replaces(self, wrapper, basic_note):
        result = await wrapper.update_note_tags(
            [basic_note], set_tags=["new", "fresh"], add=[], remove=[]
        )
        assert result["notes_modified"] == 1
        assert result["not_found"] == []
        assert await _tags(wrapper, basic_note) == ["fresh", "new"]

    async def test_set_empty_clears(self, wrapper, basic_note):
        await wrapper.update_note_tags([basic_note], set_tags=[], add=[], remove=[])
        assert await _tags(wrapper, basic_note) == []

    async def test_add_leaves_others_intact(self, wrapper, basic_note):
        # basic_note starts with ["math", "easy"]
        await wrapper.update_note_tags([basic_note], set_tags=None, add=["verb"], remove=[])
        assert await _tags(wrapper, basic_note) == ["easy", "math", "verb"]

    async def test_remove_leaves_others_intact(self, wrapper, basic_note):
        await wrapper.update_note_tags([basic_note], set_tags=None, add=[], remove=["easy"])
        assert await _tags(wrapper, basic_note) == ["math"]

    async def test_add_and_remove_combine_swap(self, wrapper):
        nid = await _make_note(wrapper, "swap", ["jp-verbs", "keep"])
        await wrapper.update_note_tags(
            [nid], set_tags=None, add=["jp", "verbs"], remove=["jp-verbs"]
        )
        assert await _tags(wrapper, nid) == ["jp", "keep", "verbs"]

    async def test_not_found_reported(self, wrapper, basic_note):
        result = await wrapper.update_note_tags(
            [basic_note, 9999999999999], set_tags=None, add=["x"], remove=[]
        )
        assert result["notes_modified"] == 1
        assert 9999999999999 in result["not_found"]

    async def test_add_applies_across_multiple_notes(self, wrapper):
        a = await _make_note(wrapper, "a", ["one"])
        b = await _make_note(wrapper, "b", ["two"])
        result = await wrapper.update_note_tags([a, b], set_tags=None, add=["shared"], remove=[])
        assert result["notes_modified"] == 2
        assert "shared" in await _tags(wrapper, a)
        assert "shared" in await _tags(wrapper, b)


class TestRenameTag:
    async def test_collection_wide_rename(self, wrapper):
        a = await _make_note(wrapper, "a", ["history::ww2"])
        b = await _make_note(wrapper, "b", ["history::ww2", "other"])
        result = await wrapper.rename_tag("history::ww2", "history::wwii", [])
        assert result["notes_modified"] == 2
        assert "history::wwii" in await _tags(wrapper, a)
        assert "history::wwii" in await _tags(wrapper, b)
        assert "history::ww2" not in await _tags(wrapper, b)

    async def test_scoped_rename_only_touches_named_notes(self, wrapper):
        a = await _make_note(wrapper, "a", ["draft"])
        b = await _make_note(wrapper, "b", ["draft"])
        result = await wrapper.rename_tag("draft", "final", [a])
        assert result["notes_modified"] == 1
        assert await _tags(wrapper, a) == ["final"]
        assert await _tags(wrapper, b) == ["draft"]

    async def test_scoped_rename_is_exact_no_substring_bleed(self, wrapper):
        nid = await _make_note(wrapper, "n", ["jp", "jp-verbs"])
        await wrapper.rename_tag("jp", "japanese", [nid])
        # Only the exact "jp" tag is renamed; "jp-verbs" is untouched.
        assert await _tags(wrapper, nid) == ["japanese", "jp-verbs"]

    async def test_scoped_rename_no_match_is_noop(self, wrapper, basic_note):
        result = await wrapper.rename_tag("absent", "whatever", [basic_note])
        assert result["notes_modified"] == 0


class TestClearUnusedTags:
    async def test_clears_orphaned_tag_names(self, wrapper):
        nid = await _make_note(wrapper, "n", ["temp-tag"])
        # Remove the tag from the note; the name lingers in the registry.
        await wrapper.update_note_tags([nid], set_tags=[], add=[], remove=[])
        assert "temp-tag" in wrapper.run_sync(lambda c: c.tags.all())

        result = await wrapper.clear_unused_tags()
        assert result["tags_removed"] >= 1
        assert "temp-tag" not in wrapper.run_sync(lambda c: c.tags.all())


@pytest.mark.parametrize("scope", [[], [123]])
async def test_rename_identical_is_handled_at_collection_layer(wrapper, scope):
    # The collection layer does not reject old == new (the tool does); a
    # collection-wide rename to the same name is simply a no-op count.
    result = await wrapper.rename_tag("same", "same", scope)
    assert result["notes_modified"] == 0
