"""Integration tests that exercise every MCP tool over HTTP transport.

Each test class gets its own isolated server with a fresh collection.
"""

from __future__ import annotations

import time

import httpx
import pytest

pytestmark = pytest.mark.integration


class TestCollectionInfo:
    """collection_info on an empty collection."""

    def test_returns_summary_by_default(self, mcp):
        result = mcp("collection_info")
        assert "summary" in result
        summary = result["summary"]
        assert "path" in summary
        assert "notes" in summary
        assert "decks" in summary

    def test_include_filters(self, mcp):
        result = mcp("collection_info", {"include": ["stats"]})
        assert result["stats"] is not None
        assert result["note_types"] is None
        assert result["decks"] is None

    def test_include_multiple(self, mcp):
        result = mcp("collection_info", {"include": ["decks", "tags"]})
        assert result["decks"] is not None
        assert result["tags"] is not None
        assert result["note_types"] is None

    def test_note_type_details(self, mcp):
        result = mcp(
            "collection_info",
            {"include": ["note_types"], "note_type_details": ["Basic"]},
        )
        basic = next(nt for nt in result["note_types"] if nt["name"] == "Basic")
        assert basic["detail"]["templates"]
        assert "css" in basic["detail"]

    def test_default_note_types(self, mcp):
        result = mcp("collection_info", {"include": ["note_types"]})
        names = {nt["name"] for nt in result["note_types"]}
        assert "Basic" in names
        assert "Cloze" in names

    def test_default_deck(self, mcp):
        result = mcp("collection_info", {"include": ["decks"]})
        assert any(d["name"] == "Default" for d in result["decks"])

    def test_stats_empty_collection(self, mcp):
        result = mcp("collection_info", {"include": ["stats"]})
        assert result["stats"]["total_notes"] == 0
        assert result["stats"]["total_cards"] == 0

    def test_stats_after_adding_notes(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Stats",
                        "note_type": "Basic",
                        "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
                    }
                    for i in range(3)
                ]
            },
        )
        result = mcp("collection_info", {"include": ["stats", "decks"]})
        assert result["stats"]["total_notes"] == 3
        stats_deck = next(d for d in result["decks"] if d["name"] == "Stats")
        assert stats_deck["note_count"] == 3

    def test_tags_appear(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                        "tags": ["alpha", "beta"],
                    }
                ]
            },
        )
        result = mcp("collection_info", {"include": ["tags"]})
        assert "alpha" in result["tags"]
        assert "beta" in result["tags"]


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
        import time
        from datetime import UTC, datetime

        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "TimeDeck",
                        "note_type": "Basic",
                        "fields": {"Front": "Old", "Back": "Note"},
                    }
                ]
            },
        )
        time.sleep(1)
        cutoff = datetime.now(UTC).isoformat()
        time.sleep(1)
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "TimeDeck",
                        "note_type": "Basic",
                        "fields": {"Front": "New", "Back": "Note"},
                    }
                ]
            },
        )

        result = mcp("list_notes", {"deck": "TimeDeck", "modified_since": cutoff})
        assert result["total"] == 1
        assert result["notes"][0]["content"]["Front"] == "New"

    def test_query_raw_search(self, mcp):
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
        result = mcp("list_notes", {"query": "mitochondria"})
        assert result["total"] == 1
        assert "mitochondria" in result["notes"][0]["content"]["Front"]

    def test_limit_over_max_rejected(self, mcp):
        # limit is schema-constrained (1-200); out-of-range is rejected.
        with pytest.raises(RuntimeError, match="less than or equal to 200"):
            mcp("list_notes", {"deck": "Clamp", "limit": 999})

    def test_limit_below_min_rejected(self, mcp):
        with pytest.raises(RuntimeError, match="greater than or equal to 1"):
            mcp("list_notes", {"deck": "ClampMin", "limit": -5})


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


class TestNoteTypeLifecycle:
    """Create, inspect, use, and update custom note types."""

    def test_create_and_inspect(self, mcp):
        result = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Custom",
                        "fields": ["Term", "Definition", "Example"],
                        "templates": [
                            {
                                "name": "Forward",
                                "front": "<div>{{Term}}</div>",
                                "back": "{{FrontSide}}<hr>{{Definition}}<br>{{Example}}",
                            }
                        ],
                        "css": ".card { font-family: sans-serif; }",
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "created"

        info = mcp(
            "collection_info",
            {"include": ["note_types"], "note_type_details": ["Custom"]},
        )
        nt = next(nt for nt in info["note_types"] if nt["name"] == "Custom")
        assert nt["fields"] == ["Term", "Definition", "Example"]
        assert len(nt["detail"]["templates"]) == 1
        assert "font-family" in nt["detail"]["css"]

    def test_create_note_with_custom_type(self, mcp):
        mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Vocab",
                        "fields": ["Word", "Meaning"],
                        "templates": [
                            {"name": "Card 1", "front": "{{Word}}", "back": "{{Meaning}}"}
                        ],
                        "css": ".card { font-size: 16px; }",
                    }
                ]
            },
        )
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Vocab",
                        "fields": {"Word": "shrike", "Meaning": "A predatory songbird"},
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "created"

    def test_update_css(self, mcp):
        created = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Styled",
                        "fields": ["Q", "A"],
                        "templates": [{"name": "Card 1", "front": "{{Q}}", "back": "{{A}}"}],
                        "css": ".card { color: black; }",
                    }
                ]
            },
        )
        nt_id = created["results"][0]["id"]

        mcp(
            "upsert_note_types",
            {"note_types": [{"id": nt_id, "css": ".card { color: red; }"}]},
        )
        info = mcp(
            "collection_info",
            {"include": ["note_types"], "note_type_details": ["Styled"]},
        )
        nt = next(nt for nt in info["note_types"] if nt["name"] == "Styled")
        assert "color: red" in nt["detail"]["css"]

    def test_update_name(self, mcp):
        created = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "OldName",
                        "fields": ["F"],
                        "templates": [{"name": "Card 1", "front": "{{F}}", "back": "{{F}}"}],
                        "css": "",
                    }
                ]
            },
        )
        nt_id = created["results"][0]["id"]

        result = mcp("upsert_note_types", {"note_types": [{"id": nt_id, "name": "NewName"}]})
        assert result["results"][0]["status"] == "updated"

        info = mcp("collection_info", {"include": ["note_types"]})
        names = {nt["name"] for nt in info["note_types"]}
        assert "NewName" in names
        assert "OldName" not in names

    def test_field_update_preserves_note_data(self, mcp):
        # Regression (#76): updating a note type's fields used to blank every
        # note's content and delete its cards. Rename a field and confirm the
        # existing note keeps its data under the new field name.
        created = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Preserve",
                        "fields": ["Front", "Back"],
                        "templates": [{"name": "C", "front": "{{Front}}", "back": "{{Back}}"}],
                        "css": "",
                    }
                ]
            },
        )
        nt_id = created["results"][0]["id"]
        note = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Preserve",
                        "note_type": "Preserve",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        note_id = note["results"][0]["id"]

        mcp("upsert_note_types", {"note_types": [{"id": nt_id, "fields": ["Frente", "Back"]}]})

        listed = mcp("list_notes", {"ids": [note_id]})["notes"][0]
        assert listed["content"] == {"Frente": "Q", "Back": "A"}

    def test_duplicate_name_rejected(self, mcp):
        result = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Basic",
                        "fields": ["A"],
                        "templates": [{"name": "C", "front": "{{A}}", "back": "{{A}}"}],
                        "css": "",
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "error"


class TestSearchNotesStub:
    def test_returns_stub_message(self, mcp):
        result = mcp("search_notes", {"queries": ["anything"]})
        assert result["results"] == []
        assert result["message"]

    def test_requires_queries_or_ids(self, mcp):
        with pytest.raises(RuntimeError, match="queries or ids"):
            mcp("search_notes", {})


class TestValidation:
    """Input validation errors reported over transport."""

    def test_list_notes_requires_filter(self, mcp):
        with pytest.raises(RuntimeError, match="filter"):
            mcp("list_notes", {})

    def test_upsert_notes_missing_deck(self, mcp):
        result = mcp(
            "upsert_notes",
            {"notes": [{"note_type": "Basic", "fields": {"Front": "Q", "Back": "A"}}]},
        )
        assert result["results"][0]["status"] == "error"

    def test_upsert_notes_invalid_note_type(self, mcp):
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Test",
                        "note_type": "Nonexistent",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "error"
        assert "not found" in result["results"][0]["error"].lower()

    def test_upsert_notes_nonexistent_field(self, mcp):
        result = mcp(
            "upsert_notes",
            {"notes": [{"deck": "Test", "note_type": "Basic", "fields": {"Nonexistent": "Q"}}]},
        )
        assert result["results"][0]["status"] == "error"

    def test_update_nonexistent_note(self, mcp):
        result = mcp("upsert_notes", {"notes": [{"id": 9999999999999, "fields": {"Front": "Q"}}]})
        assert result["results"][0]["status"] == "error"

    def test_list_nonexistent_id(self, mcp):
        result = mcp("list_notes", {"ids": [9999999999999]})
        assert result["total"] == 0
        assert result["notes"] == []


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


class TestDeckOps:
    """upsert_decks + delete_decks over transport."""

    def _deck_names(self, mcp):
        return {d["name"] for d in mcp("collection_info", {"include": ["decks"]})["decks"]}

    def test_create(self, mcp):
        result = mcp("upsert_decks", {"decks": [{"name": "Japanese::Vocab"}]})
        assert result["results"][0]["status"] == "created"
        assert "Japanese::Vocab" in self._deck_names(mcp)

    def test_create_existing_is_updated(self, mcp):
        mcp("upsert_decks", {"decks": [{"name": "Dup"}]})
        again = mcp("upsert_decks", {"decks": [{"name": "Dup"}]})
        assert again["results"][0]["status"] == "updated"

    def test_rename(self, mcp):
        created = mcp("upsert_decks", {"decks": [{"name": "Before"}]})
        did = created["results"][0]["id"]
        renamed = mcp("upsert_decks", {"decks": [{"id": did, "name": "After"}]})
        assert renamed["results"][0]["status"] == "updated"
        names = self._deck_names(mcp)
        assert "After" in names and "Before" not in names

    def test_rename_onto_existing_errors(self, mcp):
        a = mcp("upsert_decks", {"decks": [{"name": "Aa"}]})
        mcp("upsert_decks", {"decks": [{"name": "Bb"}]})
        result = mcp("upsert_decks", {"decks": [{"id": a["results"][0]["id"], "name": "Bb"}]})
        assert result["results"][0]["status"] == "error"

    def test_delete_empty(self, mcp):
        mcp("upsert_decks", {"decks": [{"name": "Temp"}]})
        result = mcp("delete_decks", {"decks": ["Temp"]})
        assert result["deleted"] == ["Temp"]
        assert "Temp" not in self._deck_names(mcp)

    def test_delete_non_empty_refused(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {"deck": "Full", "note_type": "Basic", "fields": {"Front": "Q", "Back": "A"}}
                ]
            },
        )
        result = mcp("delete_decks", {"decks": ["Full"]})
        assert result["not_empty"] == ["Full"]
        assert "Full" in self._deck_names(mcp)

    def test_delete_not_found(self, mcp):
        result = mcp("delete_decks", {"decks": ["Ghost"]})
        assert result["not_found"] == ["Ghost"]

    def test_deck_by_id_across_tools(self, mcp):
        # create deck, then reference it by #id (create note) and numeric id (list)
        did = mcp("upsert_decks", {"decks": [{"name": "ByID"}]})["results"][0]["id"]
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {"deck": f"#{did}", "note_type": "Basic", "fields": {"Front": "q", "Back": "a"}}
                ]
            },
        )
        assert mcp("list_notes", {"deck": str(did)})["total"] == 1
        # non-empty deck deleted by #id is refused (echoing the ref)
        assert mcp("delete_decks", {"decks": [f"#{did}"]})["not_empty"] == [f"#{did}"]


class TestStatusEndpoint:
    """Verify the GET /status endpoint."""

    def test_status_returns_running(self, server):
        status_url = server.url.rsplit("/", 1)[0] + "/status"
        resp = httpx.get(status_url, timeout=5.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body["running"] is True
        assert "pid" in body
        assert "url" in body
        assert "collection" in body

    def test_status_has_uptime(self, server):
        status_url = server.url.rsplit("/", 1)[0] + "/status"
        resp = httpx.get(status_url, timeout=5.0)
        body = resp.json()
        assert "uptime" in body

    def test_status_no_embedding_without_model(self, server):
        # Embedding and index are always reported now; without a model the
        # embedding is unavailable and the index reports the unavailable state.
        status_url = server.url.rsplit("/", 1)[0] + "/status"
        resp = httpx.get(status_url, timeout=5.0)
        body = resp.json()
        assert body["embedding"]["available"] is False
        assert body["embedding"]["state"] == "not_configured"
        assert body["index"]["state"] == "unavailable"


class TestHttpShutdown:
    """Verify the POST /shutdown endpoint cleanly stops the server."""

    def test_shutdown_returns_ok_and_server_exits(self, server_factory):
        srv = server_factory("shutdown")
        shutdown_url = srv.url.rsplit("/", 1)[0] + "/shutdown"

        resp = httpx.post(shutdown_url, timeout=5.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "pid" in body

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                httpx.post(
                    srv.url,
                    json={"jsonrpc": "2.0", "id": 0, "method": "ping", "params": {}},
                    timeout=1.0,
                )
                time.sleep(0.1)
            except (httpx.ConnectError, httpx.RemoteProtocolError):
                break
        else:
            pytest.fail("Server did not exit after /shutdown")


class TestBatchLimits:
    """Verify the server enforces maximum batch sizes."""

    def test_upsert_notes_over_100_rejected(self, mcp):
        with pytest.raises(RuntimeError, match="at most 100"):
            mcp(
                "upsert_notes",
                {
                    "notes": [
                        {
                            "deck": "Batch",
                            "note_type": "Basic",
                            "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
                        }
                        for i in range(101)
                    ]
                },
            )

    def test_upsert_note_types_over_10_rejected(self, mcp):
        with pytest.raises(RuntimeError, match="at most 10"):
            mcp(
                "upsert_note_types",
                {
                    "note_types": [
                        {
                            "name": f"Type{i}",
                            "fields": ["F"],
                            "templates": [{"name": "C", "front": "{{F}}", "back": "{{F}}"}],
                            "css": "",
                        }
                        for i in range(11)
                    ]
                },
            )

    def test_delete_notes_over_100_rejected(self, mcp):
        with pytest.raises(RuntimeError, match="at most 100"):
            mcp("delete_notes", {"ids": list(range(101))})
