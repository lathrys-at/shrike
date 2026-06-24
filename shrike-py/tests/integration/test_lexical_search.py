"""Integration tests for lexical (substring + fuzzy) search via the FTS5 derived store."""

from __future__ import annotations

import time

import pytest

from tests.integration.conftest import search_until

pytestmark = pytest.mark.integration


class TestDerivedLexicalSearch:
    """Substring + fuzzy lexical search via the real FTS5 derived-text store.

    Runs against the shared no-embedding server: the derived store is independent of the embedder,
    so substring and the `fuzzy` signal work with semantic ranking off. The note is ingested by the
    upsert tool's incremental hook and dropped by the reset (delete_notes removes its derived rows).
    """

    def _derived(self, server, timeout: float = 10.0):
        """Wait for the boot build to finish; return the derived-status (or skip if no FTS5)."""
        from pathlib import Path

        from shrike.client import ShrikeClient

        client = ShrikeClient(server.url, autostart=False, state_dir=Path(server.state_dir))
        deadline = time.monotonic() + timeout
        der = client.status().derived
        while der.state != "ready" and time.monotonic() < deadline:
            time.sleep(0.1)
            der = client.status().derived
        if not der.fts5:
            pytest.skip("this SQLite build has no FTS5/trigram tokenizer")
        return der

    def test_fuzzy_typo_surfaces_note(self, mcp, server):
        self._derived(server)
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "FuzzySearch",
                        "note_type": "Basic",
                        "fields": {"Front": "Mitochondria are the powerhouse", "Back": "cell"},
                    }
                ]
            },
        )
        nid = created["results"][0]["id"]
        # A typo the note doesn't literally contain — only the trigram fuzzy signal can surface it.
        # The derived (FTS5) write rides the async ingest drain, so poll until it lands.
        matches = search_until(mcp, ["mitochndria"], lambda ms: any(m["id"] == nid for m in ms))
        hit = next((m for m in matches if m["id"] == nid), None)
        assert hit is not None, "fuzzy signal did not surface the typo'd query's note"
        assert "fuzzy" in [p["signal"] for p in hit["provenance"]]
        assert hit["fuzzy"]["source"] == "field"
        assert hit["fuzzy"]["ref"] == "Front"
        assert "Mitochondria" in hit["fuzzy"]["match"]["text"]

    def test_substring_via_store(self, mcp, server):
        self._derived(server)
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "SubSearch",
                        "note_type": "Basic",
                        "fields": {"Front": "Electron transport chain", "Back": "x"},
                    }
                ]
            },
        )
        nid = created["results"][0]["id"]
        # The derived (FTS5) write rides the async ingest drain, so poll until it lands.
        matches = search_until(mcp, ["transport"], lambda ms: any(m["id"] == nid for m in ms))
        hit = next((m for m in matches if m["id"] == nid), None)
        assert hit is not None
        assert hit["substring"]["ref"] == "Front"
        assert hit["substring"]["source"] == "field"
        assert "exact" in [p["signal"] for p in hit["provenance"]]


class TestSearchNotesNoIndex:
    def test_query_without_index_notes_unavailable(self, mcp):
        # No embedding index on this server: semantic ranking is skipped, but the
        # call still runs (exact substring needs no index) and says so.
        result = mcp("search_notes", {"queries": ["anything"]})
        assert "exact text matches" in result["message"]
        assert all(not g["matches"] for g in result["results"])

    def test_requires_queries_or_ids(self, mcp):
        with pytest.raises(RuntimeError, match="queries or ids"):
            mcp("search_notes", {})
