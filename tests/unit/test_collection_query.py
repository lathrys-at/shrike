"""Collection-layer tests for query (#97): raw Anki search expressions.

Exercises CollectionWrapper._query directly (no server). The query string goes
straight to col.find_notes, so this also pins the error type for malformed input.
"""

from __future__ import annotations

import json

import pytest
import shrike_native

from tests.unit.conftest import make_notes


def _add(wrapper, front, back="x", *, tags=None, deck="D"):
    results = make_notes(
        wrapper,
        [{"note_type": "Basic", "deck": deck, "fields": {"Front": front, "Back": back},
          "tags": list(tags or [])}],
    )
    return results[0]["id"]


def _query(wrapper, q, **kw):
    fields_mode = kw.pop("fields_mode", "full")
    limit = kw.pop("limit", 50)
    return wrapper.run_sync(
        lambda c: json.loads(c.query(q, with_fields=fields_mode == "full", limit=limit))
    )


class TestCollectionQuery:
    def test_match_by_tag(self, wrapper):
        a = _add(wrapper, "alpha", tags=["keep"])
        _add(wrapper, "beta", tags=["other"])
        result = _query(wrapper, "tag:keep")
        assert result["total"] == 1
        assert result["notes"][0]["id"] == a

    def test_match_by_deck(self, wrapper):
        _add(wrapper, "x1", deck="DeckA")
        _add(wrapper, "x2", deck="DeckB")
        result = _query(wrapper, "deck:DeckA")
        assert result["total"] == 1
        assert result["notes"][0]["deck"] == "DeckA"

    def test_match_by_nid(self, wrapper):
        nid = _add(wrapper, "byid")
        result = _query(wrapper, f"nid:{nid}")
        assert [n["id"] for n in result["notes"]] == [nid]

    def test_boolean_and_negation(self, wrapper):
        _add(wrapper, "v1", tags=["verb"])
        _add(wrapper, "v2", tags=["verb", "done"])
        result = _query(wrapper, "tag:verb -tag:done")
        assert result["total"] == 1
        assert result["notes"][0]["content"]["Front"] == "v1"

    def test_total_exceeds_limit(self, wrapper):
        for i in range(5):
            _add(wrapper, f"n{i}", tags=["batch"])
        result = _query(wrapper, "tag:batch", limit=2)
        assert result["total"] == 5
        assert len(result["notes"]) == 2
        assert result["limit"] == 2

    def test_fields_meta_drops_content(self, wrapper):
        _add(wrapper, "metacard", tags=["m"])
        full = _query(wrapper, "tag:m", fields_mode="full")["notes"][0]
        meta = _query(wrapper, "tag:m", fields_mode="meta")["notes"][0]
        assert full["content"]["Front"] == "metacard"
        assert "content" not in meta  # meta mode omits the field entirely

    def test_empty_result(self, wrapper):
        _add(wrapper, "lonely")
        result = _query(wrapper, "tag:nonexistent")
        assert result == {"notes": [], "total": 0, "limit": 50}

    def test_scheduling_predicate_runs(self, wrapper):
        # Freshly added notes are "new"; the predicate must be accepted, not rejected.
        _add(wrapper, "newcard")
        assert _query(wrapper, "is:new")["total"] == 1
        assert _query(wrapper, "is:due")["total"] == 0  # nothing due yet

    def test_malformed_query_raises_search_error(self, wrapper):
        with pytest.raises(shrike_native.NativeInputError):
            _query(wrapper, "(unbalanced")
