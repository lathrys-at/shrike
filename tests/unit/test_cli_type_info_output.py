"""Unit coverage for CLI type/info output branches (#107)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from shrike.cli import cli
from shrike.cli import client as cli_client
from shrike.cli.type_cmd import _resolve_note_type
from shrike.schemas import (
    CollectionInfo,
    DeleteNoteTypesResponse,
    NoteTypeDetail,
    NoteTypeInfo,
    Summary,
    TemplateInfo,
    UpsertNoteTypesResponse,
)


@pytest.fixture
def fake() -> MagicMock:
    return MagicMock(spec=cli_client.ShrikeClient)


@pytest.fixture
def run(tmp_path, fake):
    """Invoke the CLI with the client patched to `fake` and an empty config."""
    cfg = tmp_path / "config.yml"
    cfg.write_text("")
    runner = CliRunner()

    def _run(*args: str, **kwargs):
        with patch("shrike.client.ShrikeClient", return_value=fake):
            return runner.invoke(
                cli,
                ["--config", str(cfg), *args],
                catch_exceptions=False,
                **kwargs,
            )

    return _run


def _note_types() -> list[NoteTypeInfo]:
    return [
        NoteTypeInfo(name="123", id=10, fields=["Front"]),
        NoteTypeInfo(name="Basic", id=123, fields=["Front", "Back"]),
    ]


class TestResolveNoteType:
    def test_bare_numeric_prefers_id_over_name(self, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())

        assert _resolve_note_type(fake, "123") == (123, "Basic")
        fake.collection_info.assert_called_once_with(include=["note_types"])


class TestTypeList:
    def test_identifier_json_requests_details(self, run, fake):
        detail = NoteTypeDetail(
            templates=[TemplateInfo(name="Card 1", front="{{Front}}", back="{{Back}}")],
            css=".card { color: red; }",
        )
        fake.collection_info.side_effect = [
            CollectionInfo(note_types=_note_types()),
            CollectionInfo(note_types=[NoteTypeInfo(name="Basic", id=123, detail=detail)]),
        ]

        result = run("type", "list", "123", "--json")

        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["name"] == "Basic"
        fake.collection_info.assert_any_call(include=["note_types"])
        fake.collection_info.assert_any_call(include=["note_types"], note_type_details=["Basic"])


class TestTypeCreate:
    def test_invalid_json_input_errors(self, run, fake):
        result = run("type", "create", "--json-input", input="{")

        assert result.exit_code == 1
        assert "Invalid JSON input" in result.output
        fake.upsert_note_types.assert_not_called()

    def test_renders_created_result(self, run, fake):
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[{"status": "created", "id": 999, "name": "Vocab"}]
        )

        result = run(
            "type",
            "create",
            "--name",
            "Vocab",
            "--field",
            "Front",
            "--template",
            "Card 1:{{Front}}:{{Front}}",
        )

        assert result.exit_code == 0
        assert "Created note type" in result.output
        assert "#999" in result.output

    def test_renders_item_error(self, run, fake):
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[{"status": "error", "index": 0, "error": "missing templates"}]
        )

        result = run(
            "type",
            "create",
            "--name",
            "Broken",
            "--field",
            "Front",
            "--template",
            "Card 1:{{Front}}:{{Front}}",
        )

        assert result.exit_code == 0
        assert "missing templates" in result.stderr


class TestTypeUpdate:
    def test_invalid_json_input_errors_after_resolving_identifier(self, run, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())

        result = run("type", "update", "Basic", "--json-input", input="{")

        assert result.exit_code == 1
        assert "Invalid JSON input" in result.output
        fake.upsert_note_types.assert_not_called()

    def test_nothing_to_update_errors(self, run, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())

        result = run("type", "update", "Basic")

        assert result.exit_code == 2
        assert "Nothing to update" in result.output
        fake.upsert_note_types.assert_not_called()

    def test_renders_updated_result(self, run, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[{"status": "updated", "id": 123, "name": "Renamed"}]
        )

        result = run("type", "update", "Basic", "--name", "Renamed")

        assert result.exit_code == 0
        assert "Updated note type" in result.output
        assert "#123" in result.output

    def test_renders_item_error(self, run, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[{"status": "error", "index": 0, "error": "rename failed"}]
        )

        result = run("type", "update", "Basic", "--name", "Renamed")

        assert result.exit_code == 0
        assert "rename failed" in result.stderr


class TestTypeDelete:
    def test_prompt_cancel_skips_delete(self, run, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())

        result = run("type", "delete", "Basic", input="n\n")

        assert result.exit_code == 0
        assert "Cancelled" in result.output
        fake.delete_note_types.assert_not_called()

    def test_prompt_confirm_deletes(self, run, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        fake.delete_note_types.return_value = DeleteNoteTypesResponse(
            results=[{"status": "deleted", "id": 123, "name": "Basic"}]
        )

        result = run("type", "delete", "Basic", input="y\n")

        assert result.exit_code == 0
        assert "Deleted note type" in result.output
        assert "#123" in result.output
        fake.delete_note_types.assert_called_once_with([123])

    def test_renders_not_found_and_error_results(self, run, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        fake.delete_note_types.return_value = DeleteNoteTypesResponse(
            results=[
                {"status": "not_found", "id": 123},
                {
                    "status": "error",
                    "id": 10,
                    "name": "123",
                    "error": "note type is in use",
                },
            ]
        )

        result = run("type", "delete", "Basic", "#10", "--yes")

        assert result.exit_code == 0
        assert "Not found: #123" in result.output
        assert "note type is in use" in result.output
        fake.delete_note_types.assert_called_once_with([123, 10])


class TestInfo:
    def test_json_includes_detail_sections(self, run, fake):
        summary = Summary(
            path="/collection.anki2",
            created="2026-01-01",
            modified="2026-01-02",
            notes=1,
            cards=2,
            decks=1,
            note_types=1,
            tags=3,
            due_today=0,
        )
        fake.collection_info.return_value = CollectionInfo(
            summary=summary, note_types=_note_types(), tags=["marked"]
        )

        result = run("info", "--types", "--tags", "--json")

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["note_types"][0]["name"] == "123"
        assert data["tags"] == ["marked"]
        fake.collection_info.assert_called_once_with(
            include=["summary", "note_types", "tags"],
            note_type_details=None,
        )
