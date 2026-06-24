"""Integration tests for the collection_prune MCP tool over transport."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestCollectionPrune:
    """collection_prune over transport: unused tags + empty notes."""

    def test_preview_then_apply(self, mcp):
        # A note carrying a tag, then blank its fields (an update may empty a note)
        # and strip its tag — so it's both an empty note and the source of an
        # orphan tag.
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Prune",
                        "note_type": "Basic",
                        "fields": {"Front": "temp", "Back": "x"},
                        "tags": ["pruneme"],
                    }
                ]
            },
        )
        nid = created["results"][0]["id"]
        mcp("upsert_notes", {"notes": [{"id": nid, "fields": {"Front": "", "Back": ""}}]})
        mcp("update_note_tags", {"note_ids": [nid], "set": []})  # orphan "pruneme"

        # Preview (dry_run: true) reports both and mutates nothing.
        preview = mcp("collection_prune", {"dry_run": True})
        assert preview["dry_run"] is True
        assert nid in preview["empty_notes"]["removed"]
        assert "pruneme" in preview["unused_tags"]["tags"]
        assert mcp("list_notes", {"ids": [nid]})["total"] == 1  # still there

        # Apply (the default) — the empty note is gone and the orphan tag cleared.
        applied = mcp("collection_prune", {})
        assert applied["dry_run"] is False
        assert nid in applied["empty_notes"]["removed"]
        assert mcp("list_notes", {"ids": [nid]})["total"] == 0
        assert "pruneme" not in mcp("collection_info", {"include": ["tags"]})["tags"]

    def test_selected_cleanup_only(self, mcp):
        # dry_run: true so a stray unused-tag clear can't perturb the shared
        # collection's tag registry (which the per-test reset can't restore).
        result = mcp("collection_prune", {"unused_tags": True, "dry_run": True})
        assert result["unused_tags"] is not None
        assert result["empty_notes"] is None
        assert result["empty_cards"] is None
