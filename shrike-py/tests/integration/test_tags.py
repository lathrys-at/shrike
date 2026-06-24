"""Integration tests for note-tag and tag-rename MCP tools over transport."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestTagOps:
    """update_note_tags and rename_tag over transport."""

    def _make(self, mcp, tags):
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                        "tags": tags,
                    }
                ]
            },
        )
        return created["results"][0]["id"]

    def _tags(self, mcp, nid):
        return sorted(mcp("list_notes", {"ids": [nid]})["notes"][0]["tags"])

    def test_set_replaces(self, mcp):
        nid = self._make(mcp, ["old"])
        result = mcp("update_note_tags", {"note_ids": [nid], "set": ["a", "b"]})
        assert result["notes_modified"] == 1
        assert self._tags(mcp, nid) == ["a", "b"]

    def test_add_and_remove_combine(self, mcp):
        nid = self._make(mcp, ["jp-verbs", "keep"])
        mcp("update_note_tags", {"note_ids": [nid], "add": ["jp", "verbs"], "remove": ["jp-verbs"]})
        assert self._tags(mcp, nid) == ["jp", "keep", "verbs"]

    def test_not_found_reported(self, mcp):
        result = mcp("update_note_tags", {"note_ids": [9999999999999], "add": ["x"]})
        assert result["notes_modified"] == 0
        assert 9999999999999 in result["not_found"]

    def test_set_with_add_rejected(self, mcp):
        nid = self._make(mcp, ["x"])
        with pytest.raises(RuntimeError, match="not both"):
            mcp("update_note_tags", {"note_ids": [nid], "set": ["a"], "add": ["b"]})

    def test_no_mode_rejected(self, mcp):
        nid = self._make(mcp, ["x"])
        with pytest.raises(RuntimeError, match="Specify"):
            mcp("update_note_tags", {"note_ids": [nid]})

    def test_rename_collection_wide(self, mcp):
        nid = self._make(mcp, ["history::ww2"])
        result = mcp("rename_tag", {"old": "history::ww2", "new": "history::wwii"})
        assert result["notes_modified"] == 1
        assert "history::wwii" in self._tags(mcp, nid)

    def test_rename_scoped_exact(self, mcp):
        nid = self._make(mcp, ["jp", "jp-verbs"])
        result = mcp("rename_tag", {"old": "jp", "new": "japanese", "note_ids": [nid]})
        assert result["notes_modified"] == 1
        assert self._tags(mcp, nid) == ["japanese", "jp-verbs"]

    def test_rename_identical_rejected(self, mcp):
        with pytest.raises(RuntimeError, match="identical"):
            mcp("rename_tag", {"old": "a", "new": "a"})
