"""Integration tests for the collection_query (raw Anki search) MCP tool over transport."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestCollectionQuery:
    """collection_query (raw Anki search) over transport.

    Each test seeds its own note — the suite runs under xdist, which scatters
    tests across workers with independent collections, so nothing may rely on a
    sibling test's data.
    """

    def _make(self, mcp, front, tag):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Q",
                        "note_type": "Basic",
                        "fields": {"Front": front, "Back": "x"},
                        "tags": [tag],
                    }
                ]
            },
        )

    def test_query_by_tag(self, mcp):
        self._make(mcp, "qcard", "qtag")
        result = mcp("collection_query", {"query": "tag:qtag"})
        assert result["total"] == 1
        assert result["notes"][0]["content"]["Front"] == "qcard"

    def test_scheduling_predicate(self, mcp):
        # Raw predicates are accepted (the point of the tool); is:new matches a
        # freshly added note.
        self._make(mcp, "schedcard", "schedtag")
        result = mcp("collection_query", {"query": "is:new tag:schedtag", "fields": "meta"})
        assert result["total"] == 1
        assert result["notes"][0]["content"] is None

    def test_malformed_query_errors(self, mcp):
        with pytest.raises(RuntimeError):
            mcp("collection_query", {"query": "(unbalanced"})
