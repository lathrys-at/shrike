"""Tests for the shrike.schemas wire-contract models."""

from __future__ import annotations

import pytest

from shrike import schemas
from shrike.schemas import (
    CollectionInfo,
    DeleteNotesResponse,
    DeleteNoteTypesResponse,
    ListNotesResponse,
    SearchMatch,
    SearchResponse,
    ServerStatus,
    UpsertNotesResponse,
    UpsertNoteTypesResponse,
)

# Every tool response model — each must accept a bare {"error": ...} payload so
# the _safe_tool catch-all dict can be coerced by FastMCP into the declared type.
TOOL_RESPONSE_MODELS = [
    CollectionInfo,
    ListNotesResponse,
    SearchResponse,
    UpsertNotesResponse,
    UpsertNoteTypesResponse,
    DeleteNotesResponse,
    DeleteNoteTypesResponse,
]


@pytest.mark.parametrize("model", TOOL_RESPONSE_MODELS)
def test_error_only_payload_validates(model) -> None:
    """A {"error": ...} dict must validate into every tool response model."""
    obj = model.model_validate({"error": "Internal error: boom"})
    assert obj.error == "Internal error: boom"


def test_list_notes_response_parses_notes() -> None:
    resp = ListNotesResponse.model_validate(
        {
            "notes": [
                {
                    "id": 1,
                    "note_type": "Basic",
                    "deck": "Test",
                    "tags": ["a"],
                    "modified": "2026-01-01T00:00:00+00:00",
                    "content": {"Front": "Q", "Back": "A"},
                }
            ],
            "total": 1,
            "limit": 50,
        }
    )
    assert resp.notes[0].content == {"Front": "Q", "Back": "A"}
    assert resp.total == 1


def test_search_match_includes_score_and_note_fields() -> None:
    m = SearchMatch.model_validate(
        {
            "id": 7,
            "note_type": "Basic",
            "deck": "D",
            "tags": [],
            "modified": "2026-01-01T00:00:00+00:00",
            "score": 0.91,
        }
    )
    assert m.score == 0.91
    assert m.id == 7


def test_search_response_uses_message_not_underscore() -> None:
    resp = SearchResponse.model_validate({"results": [], "message": "building"})
    assert resp.message == "building"
    # The legacy underscore key is ignored (extra="ignore"), not surfaced.
    legacy = SearchResponse.model_validate({"results": [], "_message": "building"})
    assert legacy.message is None


def test_upsert_response_neighbor_shape() -> None:
    resp = UpsertNotesResponse.model_validate(
        {
            "results": [
                {
                    "status": "created",
                    "id": 1,
                    "neighbors": [{"id": 2, "score": 0.8, "tags": ["x"]}],
                }
            ]
        }
    )
    r = resp.results[0]
    assert r.neighbors is not None
    assert r.neighbors[0].id == 2
    assert r.neighbors[0].score == 0.8


def test_server_status_tolerates_degraded_daemon_dict() -> None:
    """The degraded daemon.server_status() shape (no embedding/index) parses."""
    status = ServerStatus.model_validate(
        {
            "running": True,
            "pid": 123,
            "url": "http://127.0.0.1:8372/mcp",
            "collection": "/c.anki2",
            "log_level": "info",
            "log_dir": "/logs",
            "started": "2026-01-01T00:00:00+00:00",
            "uptime": "5s",
        }
    )
    assert status.running is True
    assert status.embedding is None
    assert status.index is None


def test_schemas_module_has_no_shrike_imports() -> None:
    """schemas.py must stay a leaf module (pydantic only) to avoid import cycles."""
    from pathlib import Path

    src = (schemas.__file__ or "").strip()
    assert src.endswith("schemas.py")
    text = Path(src).read_text(encoding="utf-8")
    assert "from shrike" not in text
    assert "import shrike" not in text
