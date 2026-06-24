"""Integration tests for the note lifecycle, upsert duplicate policy, list filters, and bulk ops."""

from __future__ import annotations

import pytest

from tests.integration.conftest import search_until

pytestmark = pytest.mark.integration


class TestNoteLifecycle:
    """Full create -> list -> update -> delete cycle over HTTP."""

    def test_create_and_retrieve(self, mcp):
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Lifecycle",
                        "note_type": "Basic",
                        "fields": {"Front": "Capital of France?", "Back": "Paris"},
                        "tags": ["geography"],
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "created"
        note_id = result["results"][0]["id"]

        listed = mcp("list_notes", {"ids": [note_id]})
        assert listed["total"] == 1
        note = listed["notes"][0]
        assert note["content"]["Front"] == "Capital of France?"
        assert note["deck"] == "Lifecycle"
        assert "geography" in note["tags"]

    def test_list_by_deck(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "DeckFilter",
                        "note_type": "Basic",
                        "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
                    }
                    for i in range(3)
                ]
            },
        )
        result = mcp("list_notes", {"deck": "DeckFilter"})
        assert result["total"] == 3

    def test_list_by_tags(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Tags",
                        "note_type": "Basic",
                        "fields": {"Front": "Q1", "Back": "A1"},
                        "tags": ["findme"],
                    },
                    {
                        "deck": "Tags",
                        "note_type": "Basic",
                        "fields": {"Front": "Q2", "Back": "A2"},
                        "tags": ["other"],
                    },
                ]
            },
        )
        result = mcp("list_notes", {"tags": ["findme"]})
        assert result["total"] == 1
        assert result["notes"][0]["content"]["Front"] == "Q1"

    def test_list_by_note_type(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        result = mcp("list_notes", {"note_type": "Basic"})
        assert result["total"] >= 1
        assert all(n["note_type"] == "Basic" for n in result["notes"])

    def test_list_meta_only(self, mcp):
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Meta",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        note_id = result["results"][0]["id"]
        listed = mcp("list_notes", {"ids": [note_id], "fields": "meta"})
        note = listed["notes"][0]
        assert note["content"] is None
        assert note["note_type"] == "Basic"
        assert note["deck"] == "Meta"

    def test_list_with_limit(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Limit",
                        "note_type": "Basic",
                        "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
                    }
                    for i in range(5)
                ]
            },
        )
        result = mcp("list_notes", {"deck": "Limit", "limit": 2})
        assert len(result["notes"]) == 2
        assert result["total"] == 5

    def test_update_fields(self, mcp):
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Update",
                        "note_type": "Basic",
                        "fields": {"Front": "Original Q", "Back": "Original A"},
                    }
                ]
            },
        )
        note_id = created["results"][0]["id"]

        updated = mcp(
            "upsert_notes",
            {"notes": [{"id": note_id, "fields": {"Back": "Updated A"}}]},
        )
        assert updated["results"][0]["status"] == "updated"

        note = mcp("list_notes", {"ids": [note_id]})["notes"][0]
        assert note["content"]["Back"] == "Updated A"
        assert note["content"]["Front"] == "Original Q"

    def test_update_tags(self, mcp):
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                        "tags": ["old"],
                    }
                ]
            },
        )
        note_id = created["results"][0]["id"]

        mcp("upsert_notes", {"notes": [{"id": note_id, "tags": ["new", "updated"]}]})
        note = mcp("list_notes", {"ids": [note_id]})["notes"][0]
        assert set(note["tags"]) == {"new", "updated"}

    def test_move_deck(self, mcp):
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Source",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        note_id = created["results"][0]["id"]

        mcp("upsert_notes", {"notes": [{"id": note_id, "deck": "Destination"}]})
        note = mcp("list_notes", {"ids": [note_id]})["notes"][0]
        assert note["deck"] == "Destination"

    def test_delete(self, mcp):
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Delete",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        note_id = created["results"][0]["id"]

        result = mcp("delete_notes", {"ids": [note_id]})
        assert note_id in result["deleted"]

        listed = mcp("list_notes", {"ids": [note_id]})
        assert listed["total"] == 0

    def test_note_has_modified_timestamp(self, mcp):
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        note_id = created["results"][0]["id"]
        note = mcp("list_notes", {"ids": [note_id]})["notes"][0]
        assert "modified" in note


class TestUpsertDuplicatePolicy:
    """on_duplicate policy and dry_run over HTTP."""

    DUP = {"deck": "Dup", "note_type": "Basic", "fields": {"Front": "Dup Q", "Back": "A"}}

    def _count(self, mcp) -> int:
        return mcp("collection_info", {"include": ["stats"]})["stats"]["total_notes"]

    def test_error_default_blocks_duplicate(self, mcp):
        created = mcp("upsert_notes", {"notes": [self.DUP]})  # helper injects allow
        assert created["results"][0]["status"] == "created"

        blocked = mcp("upsert_notes", {"notes": [self.DUP], "on_duplicate": "error"})
        assert blocked["results"][0]["status"] == "error"
        assert blocked["results"][0]["reason"] == "duplicate"

    def test_skip_and_allow(self, mcp):
        mcp("upsert_notes", {"notes": [self.DUP]})
        before = self._count(mcp)

        skipped = mcp("upsert_notes", {"notes": [self.DUP], "on_duplicate": "skip"})
        assert skipped["results"][0] == {"status": "skipped", "index": 0, "reason": "duplicate"}
        assert self._count(mcp) == before  # nothing added

        allowed = mcp("upsert_notes", {"notes": [self.DUP], "on_duplicate": "allow"})
        assert allowed["results"][0]["status"] == "created"
        assert self._count(mcp) == before + 1

    def test_dry_run_validates_without_writing(self, mcp):
        before = self._count(mcp)
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {"deck": "Dry", "note_type": "Basic", "fields": {"Front": "DryQ", "Back": "x"}},
                    {"deck": "Dry", "note_type": "Basic", "fields": {"Front": "", "Back": "y"}},
                ],
                "dry_run": True,
            },
        )
        assert result["dry_run"] is True
        assert result["results"][0] == {"status": "ok", "index": 0, "action": "create"}
        assert result["results"][1]["status"] == "error"
        assert result["results"][1]["reason"] == "empty"
        assert self._count(mcp) == before  # wrote nothing


class TestListNotesAdvanced:
    """Tests for modified_since, query, and limit clamping."""

    def test_modified_since(self, mcp):
        # End-to-end plumbing of the `modified_since` filter both ways — a past
        # cutoff includes the note, a future cutoff excludes it. The precise
        # boundary-between-two-notes semantics is covered deterministically in
        # the unit tests (no real-time sleep needed here).
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "TimeDeck",
                        "note_type": "Basic",
                        "fields": {"Front": "Note", "Back": "Body"},
                    }
                ]
            },
        )

        past = mcp("list_notes", {"deck": "TimeDeck", "modified_since": "2000-01-01T00:00:00Z"})
        assert past["total"] == 1
        assert past["notes"][0]["content"]["Front"] == "Note"

        future = mcp("list_notes", {"deck": "TimeDeck", "modified_since": "2099-01-01T00:00:00Z"})
        assert future["total"] == 0

    def test_search_substring_without_index(self, mcp):
        # This server has no embedding index, so search_notes' semantic ranking is
        # skipped — but exact substring matching still works and is annotated.
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Search",
                        "note_type": "Basic",
                        "fields": {"Front": "mitochondria", "Back": "powerhouse"},
                    },
                    {
                        "deck": "Search",
                        "note_type": "Basic",
                        "fields": {"Front": "ribosome", "Back": "protein"},
                    },
                ]
            },
        )
        # The derived (FTS5) write rides the async ingest drain — poll until the
        # substring match lands, then read the full response (now settled) for the message.
        search_until(mcp, ["mitochondria"], lambda ms: len(ms) == 1)
        result = mcp("search_notes", {"queries": ["mitochondria"]})
        matches = result["results"][0]["matches"]
        assert len(matches) == 1
        assert matches[0]["score"] is None
        assert matches[0]["substring"]["ref"] == "Front"
        assert "exact text matches" in (result.get("message") or "")

    def test_limit_over_max_rejected(self, mcp):
        # limit is schema-constrained (0-200); above-max is rejected.
        with pytest.raises(RuntimeError, match="less than or equal to 200"):
            mcp("list_notes", {"deck": "Clamp", "limit": 999})

    def test_limit_below_min_rejected(self, mcp):
        # The floor is 0 (0 means "return all"); a negative is rejected.
        with pytest.raises(RuntimeError, match="greater than or equal to 0"):
            mcp("list_notes", {"deck": "ClampMin", "limit": -5})

    def test_limit_zero_accepted(self, mcp):
        # limit=0 means "return all" — a valid value, not a bound violation.
        result = mcp("list_notes", {"deck": "ClampZero", "limit": 0})
        assert "notes" in result


class TestBulkOperations:
    """Batch creates, partial failures, mixed deletes."""

    def test_create_multiple(self, mcp):
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Bulk",
                        "note_type": "Basic",
                        "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
                    }
                    for i in range(5)
                ]
            },
        )
        assert len(result["results"]) == 5
        assert all(r["status"] == "created" for r in result["results"])

        listed = mcp("list_notes", {"deck": "Bulk"})
        assert listed["total"] == 5

    def test_partial_failure(self, mcp):
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Bulk",
                        "note_type": "Basic",
                        "fields": {"Front": "Good", "Back": "Note"},
                    },
                    {
                        "deck": "Bulk",
                        "note_type": "DoesNotExist",
                        "fields": {"Front": "Bad", "Back": "Note"},
                    },
                ]
            },
        )
        assert result["results"][0]["status"] == "created"
        assert result["results"][1]["status"] == "error"

    def test_delete_nonexistent(self, mcp):
        result = mcp("delete_notes", {"ids": [9999999999999]})
        assert result["deleted"] == []
        assert 9999999999999 in result["not_found"]

    def test_delete_mixed(self, mcp):
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        real_id = created["results"][0]["id"]
        result = mcp("delete_notes", {"ids": [real_id, 9999999999999]})
        assert real_id in result["deleted"]
        assert 9999999999999 in result["not_found"]

    def test_delete_multiple(self, mcp):
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Basic",
                        "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
                    }
                    for i in range(3)
                ]
            },
        )
        ids = [r["id"] for r in created["results"]]
        result = mcp("delete_notes", {"ids": ids})
        assert set(result["deleted"]) == set(ids)
