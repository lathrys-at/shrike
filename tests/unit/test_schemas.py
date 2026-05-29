"""Tests for the shrike.schemas wire-contract models."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from shrike import schemas
from shrike.schemas import (
    DeleteNoteTypeResult,
    ListNotesResponse,
    SearchMatch,
    SearchResponse,
    ServerStatus,
    UpsertNoteError,
    UpsertNoteOk,
    UpsertNoteResult,
    UpsertNotesResponse,
)


def test_result_union_discriminates_on_status() -> None:
    """A result dict resolves to the variant matching its status, with required fields."""
    ok = TypeAdapter(UpsertNoteResult).validate_python({"status": "created", "id": 1})
    assert isinstance(ok, UpsertNoteOk)
    assert ok.id == 1 and ok.neighbors == [] and ok.neighbors_unavailable is False

    err = TypeAdapter(UpsertNoteResult).validate_python(
        {"status": "error", "index": 2, "error": "bad"}
    )
    assert isinstance(err, UpsertNoteError)
    assert err.index == 2 and err.error == "bad"


def test_result_union_rejects_illegal_mix() -> None:
    """The success variant requires `id`; an error-shaped payload can't masquerade as it."""
    with pytest.raises(ValidationError):
        TypeAdapter(UpsertNoteResult).validate_python({"status": "created", "error": "x"})


def test_delete_result_union_variants() -> None:
    ta = TypeAdapter(DeleteNoteTypeResult)
    assert ta.validate_python({"status": "deleted", "id": 1, "name": "T"}).name == "T"
    assert ta.validate_python({"status": "not_found", "id": 2}).id == 2
    assert ta.validate_python({"status": "error", "id": 3, "name": "T", "error": "in use"}).error


def test_response_models_have_no_error_field() -> None:
    """Whole-call failures use MCP isError, so response models carry no `error`."""
    for model in (ListNotesResponse, SearchResponse, UpsertNotesResponse):
        assert "error" not in model.model_fields


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


def test_server_status_models_the_responsive_payload() -> None:
    """ServerStatus is exactly a responding server's /status report.

    embedding/index are required — the "not running" / "unresponsive" connection
    states are the CLI's concern, not optionals smuggled into this model.
    """
    status = ServerStatus.model_validate(
        {
            "running": True,
            "pid": 123,
            "url": "http://127.0.0.1:8372/mcp",
            "collection": "/c.anki2",
            "log_level": "info",
            "log_dir": "/logs",
            "uptime": "5s",
            "embedding": {"state": "not_configured"},
            "index": {"state": "unavailable"},
        }
    )
    assert status.pid == 123
    assert status.embedding.state == "not_configured"
    assert status.index.state == "unavailable"

    with pytest.raises(ValidationError):
        ServerStatus.model_validate({"running": True, "pid": 1})  # missing embedding/index


def test_schemas_module_has_no_shrike_imports() -> None:
    """schemas.py must stay a leaf module (pydantic only) to avoid import cycles."""
    from pathlib import Path

    src = (schemas.__file__ or "").strip()
    assert src.endswith("schemas.py")
    text = Path(src).read_text(encoding="utf-8")
    assert "from shrike" not in text
    assert "import shrike" not in text
