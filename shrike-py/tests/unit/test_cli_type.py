"""Unit coverage for `shrike type` command branches.

Covers the identifier-resolution edges (`_resolve_note_type` ID-not-found,
name-not-found-with-available-list, the `#id` strip, and the bare-numeric
id-over-name preference), the `type list` table / detail / empty / not-found /
JSON render forks, the inline `create`/`update` validation errors, the JSON
formatting variants, and the per-item delete error render. Driven through
Click's CliRunner with a mocked ShrikeClient — no server.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from shrike.cli import cli
from shrike.cli import client as cli_client
from shrike.cli.type_cmd import _parse_template, _resolve_note_type
from shrike.schemas import (
    CollectionInfo,
    DeleteNoteTypesResponse,
    NoteTypeDetail,
    NoteTypeInfo,
    TemplateInfo,
    UpsertNoteTypesResponse,
)


@pytest.fixture
def fake() -> MagicMock:
    return MagicMock(spec=cli_client.ShrikeClient)


@pytest.fixture
def run(tmp_path, fake):
    cfg = tmp_path / "config.yml"
    cfg.write_text("")
    runner = CliRunner()

    def _run(*args: str, **kwargs):
        with patch("shrike.client.ShrikeClient", return_value=fake):
            return runner.invoke(cli, ["--config", str(cfg), *args], **kwargs)

    return _run


@pytest.fixture
def run_strict(tmp_path, fake):
    """Invoke the CLI with exceptions propagated (no Click capture)."""
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
        NoteTypeInfo(name="Basic", id=10, type="standard", fields=["Front", "Back"]),
        NoteTypeInfo(name="Cloze", id=20, type="cloze", fields=["Text"]),
    ]


def _note_types_id_named() -> list[NoteTypeInfo]:
    # A note type literally named "123" alongside "Basic" (id=123): exercises
    # the bare-numeric id-over-name resolution preference.
    return [
        NoteTypeInfo(name="123", id=10, fields=["Front"]),
        NoteTypeInfo(name="Basic", id=123, fields=["Front", "Back"]),
    ]


class TestResolveNoteType:
    def test_numeric_id_not_found_errors(self, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        with pytest.raises(click.ClickException) as ei:
            _resolve_note_type(fake, "999")
        assert "ID 999 not found" in str(ei.value)

    def test_hash_numeric_id_resolves(self, fake) -> None:
        # The `#id` edge: a "#"-prefixed numeric resolves by id.
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        assert _resolve_note_type(fake, "#20") == (20, "Cloze")

    def test_unknown_name_lists_available(self, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        with pytest.raises(click.ClickException) as ei:
            _resolve_note_type(fake, "Nope")
        msg = str(ei.value)
        assert "'Nope' not found" in msg
        assert "Basic" in msg and "Cloze" in msg  # the available list

    def test_bare_numeric_prefers_id_over_name(self, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types_id_named())

        assert _resolve_note_type(fake, "123") == (123, "Basic")
        fake.collection_info.assert_called_once_with(include=["note_types"])


class TestParseTemplate:
    def test_bad_format_errors(self) -> None:
        with pytest.raises(click.BadParameter):
            _parse_template("missing-colons")

    def test_splits_on_first_two_colons_only(self) -> None:
        # Template HTML may carry ':', so only the first two delimiters split.
        out = _parse_template("Card 1:{{Front}}:{{Back}} a:b")
        assert out == {"name": "Card 1", "front": "{{Front}}", "back": "{{Back}} a:b"}


class TestTypeList:
    def test_no_identifier_empty_prints_placeholder(self, run, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(note_types=[])
        result = run("type", "list")
        assert result.exit_code == 0, result.output
        assert "No note types found" in result.output

    def test_no_identifier_renders_table(self, run, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        result = run("type", "list")
        assert result.exit_code == 0, result.output
        assert "Basic" in result.output and "Cloze" in result.output
        assert "#10" in result.output

    def test_no_identifier_json(self, run, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        result = run("--json", "type", "list")
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)[0]["name"] == "Basic"

    def test_identifier_detail_render(self, run, fake) -> None:
        detail = NoteTypeDetail(
            templates=[TemplateInfo(name="Card 1", front="{{Front}}", back="{{Back}}")],
            css=".card {}",
        )
        fake.collection_info.side_effect = [
            CollectionInfo(note_types=_note_types()),
            CollectionInfo(note_types=[NoteTypeInfo(name="Basic", id=10, detail=detail)]),
        ]
        result = run("type", "list", "Basic")
        assert result.exit_code == 0, result.output
        assert "Note Type" in result.output
        assert "Card 1" in result.output

    def test_identifier_detail_missing_errors(self, run, fake) -> None:
        # Resolves the id, but the detail fetch returns no detail → ClickException.
        fake.collection_info.side_effect = [
            CollectionInfo(note_types=_note_types()),
            CollectionInfo(note_types=[NoteTypeInfo(name="Basic", id=10, detail=None)]),
        ]
        result = run("type", "list", "Basic")
        assert result.exit_code != 0
        assert "'Basic' not found" in result.output

    def test_identifier_json_requests_details(self, run_strict, fake):
        detail = NoteTypeDetail(
            templates=[TemplateInfo(name="Card 1", front="{{Front}}", back="{{Back}}")],
            css=".card { color: red; }",
        )
        fake.collection_info.side_effect = [
            CollectionInfo(note_types=_note_types_id_named()),
            CollectionInfo(note_types=[NoteTypeInfo(name="Basic", id=123, detail=detail)]),
        ]

        result = run_strict("type", "list", "123", "--json")

        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["name"] == "Basic"
        fake.collection_info.assert_any_call(include=["note_types"])
        fake.collection_info.assert_any_call(include=["note_types"], note_type_details=["Basic"])


class TestTypeShow:
    def test_show_delegates_to_list(self, run, fake) -> None:
        detail = NoteTypeDetail(
            templates=[TemplateInfo(name="Card 1", front="{{Front}}", back="{{Back}}")],
            css="",
        )
        fake.collection_info.side_effect = [
            CollectionInfo(note_types=_note_types()),
            CollectionInfo(note_types=[NoteTypeInfo(name="Basic", id=10, detail=detail)]),
        ]
        result = run("type", "show", "Basic")
        assert result.exit_code == 0, result.output
        assert "Note Type" in result.output


class TestTypeCreate:
    def test_requires_name(self, run, fake) -> None:
        result = run("type", "create", "--field", "F", "--template", "C:f:b")
        assert result.exit_code != 0
        assert "--name is required" in result.output

    def test_requires_field(self, run, fake) -> None:
        result = run("type", "create", "--name", "X", "--template", "C:f:b")
        assert result.exit_code != 0
        assert "--field is required" in result.output

    def test_requires_template(self, run, fake) -> None:
        result = run("type", "create", "--name", "X", "--field", "F")
        assert result.exit_code != 0
        assert "--template is required" in result.output

    def test_cloze_flag_sets_is_cloze(self, run, fake) -> None:
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[{"status": "created", "id": 1, "name": "C"}]
        )
        result = run(
            "type",
            "create",
            "--name",
            "C",
            "--field",
            "Text",
            "--template",
            "C:{{Text}}:x",
            "--cloze",
        )
        assert result.exit_code == 0, result.output
        (note_types,), _ = fake.upsert_note_types.call_args
        assert note_types[0]["is_cloze"] is True

    def test_json_input_array_passthrough(self, run, fake) -> None:
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[{"status": "created", "id": 1, "name": "A"}]
        )
        payload = json.dumps([{"name": "A", "fields": ["F"], "templates": []}])
        result = run("type", "create", "--json-input", input=payload)
        assert result.exit_code == 0, result.output
        (note_types,), _ = fake.upsert_note_types.call_args
        assert note_types == [{"name": "A", "fields": ["F"], "templates": []}]

    def test_json_output(self, run, fake) -> None:
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[{"status": "created", "id": 7, "name": "V"}]
        )
        result = run(
            "--json", "type", "create", "--name", "V", "--field", "F", "--template", "C:f:b"
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["results"][0]["id"] == 7

    def test_mixed_results_loop_skips_unmatched_status(self, run, fake) -> None:
        # An `updated` status in a CREATE response matches neither the created nor
        # the error branch, so the render loop silently iterates past it to the
        # next result (the loop-continue arc).
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[
                {"status": "updated", "id": 5, "name": "Unexpected"},
                {"status": "created", "id": 1, "name": "A"},
            ]
        )
        payload = json.dumps(
            [
                {"name": "Unexpected", "fields": ["F"], "templates": []},
                {"name": "A", "fields": ["F"], "templates": []},
            ]
        )
        result = run("type", "create", "--json-input", input=payload)
        assert result.exit_code == 0, result.output
        assert "Created note type" in result.output
        # The unexpected `updated` result is not rendered by the create command.
        assert "Unexpected" not in result.output

    def test_invalid_json_input_errors(self, run_strict, fake):
        result = run_strict("type", "create", "--json-input", input="{")

        assert result.exit_code == 1
        assert "Invalid JSON input" in result.output
        fake.upsert_note_types.assert_not_called()

    def test_renders_created_result(self, run_strict, fake):
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[{"status": "created", "id": 999, "name": "Vocab"}]
        )

        result = run_strict(
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

    def test_renders_item_error(self, run_strict, fake):
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[{"status": "error", "index": 0, "error": "missing templates"}]
        )

        result = run_strict(
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
    def test_css_only_update_renders(self, run, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[{"status": "updated", "id": 10, "name": "Basic"}]
        )
        result = run("type", "update", "Basic", "--css", ".card {}")
        assert result.exit_code == 0, result.output
        assert "Updated note type" in result.output
        (note_types,), _ = fake.upsert_note_types.call_args
        assert note_types[0] == {"id": 10, "css": ".card {}"}

    def test_json_input_merges_id(self, run, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[{"status": "updated", "id": 10, "name": "Basic"}]
        )
        result = run(
            "type", "update", "Basic", "--json-input", input=json.dumps({"fields": ["A", "B"]})
        )
        assert result.exit_code == 0, result.output
        (note_types,), _ = fake.upsert_note_types.call_args
        assert note_types[0]["id"] == 10
        assert note_types[0]["fields"] == ["A", "B"]

    def test_json_output(self, run, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[{"status": "updated", "id": 10, "name": "Basic"}]
        )
        result = run("--json", "type", "update", "Basic", "--name", "Renamed")
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["results"][0]["status"] == "updated"

    def test_mixed_results_loop_skips_unmatched_status(self, run, fake) -> None:
        # A `created` status in an UPDATE response matches neither branch, so the
        # render loop iterates past it (the loop-continue arc).
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[
                {"status": "created", "id": 99, "name": "Unexpected"},
                {"status": "updated", "id": 10, "name": "Basic"},
            ]
        )
        result = run("type", "update", "Basic", "--name", "Renamed")
        assert result.exit_code == 0, result.output
        assert "Updated note type" in result.output
        assert "Unexpected" not in result.output

    def test_invalid_json_input_errors_after_resolving_identifier(self, run_strict, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types_id_named())

        result = run_strict("type", "update", "Basic", "--json-input", input="{")

        assert result.exit_code == 1
        assert "Invalid JSON input" in result.output
        fake.upsert_note_types.assert_not_called()

    def test_nothing_to_update_errors(self, run_strict, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types_id_named())

        result = run_strict("type", "update", "Basic")

        assert result.exit_code == 2
        assert "Nothing to update" in result.output
        fake.upsert_note_types.assert_not_called()

    def test_renders_updated_result(self, run_strict, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types_id_named())
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[{"status": "updated", "id": 123, "name": "Renamed"}]
        )

        result = run_strict("type", "update", "Basic", "--name", "Renamed")

        assert result.exit_code == 0
        assert "Updated note type" in result.output
        assert "#123" in result.output

    def test_renders_item_error(self, run_strict, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types_id_named())
        fake.upsert_note_types.return_value = UpsertNoteTypesResponse(
            results=[{"status": "error", "index": 0, "error": "rename failed"}]
        )

        result = run_strict("type", "update", "Basic", "--name", "Renamed")

        assert result.exit_code == 0
        assert "rename failed" in result.stderr


class TestTypeDelete:
    def test_json_output(self, run, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        fake.delete_note_types.return_value = DeleteNoteTypesResponse(
            results=[{"status": "deleted", "id": 10, "name": "Basic"}]
        )
        result = run("--json", "type", "delete", "Basic", "--yes")
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["results"][0]["status"] == "deleted"

    def test_per_item_error_rendered(self, run, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        fake.delete_note_types.return_value = DeleteNoteTypesResponse(
            results=[{"status": "error", "id": 10, "name": "Basic", "error": "in use"}]
        )
        result = run("type", "delete", "Basic", "--yes")
        assert result.exit_code == 0, result.output
        assert "in use" in result.output
        assert "#10" in result.output

    def test_mixed_deleted_and_error_loop(self, run, fake) -> None:
        # Two results force the delete render loop to iterate over both branches.
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types())
        fake.delete_note_types.return_value = DeleteNoteTypesResponse(
            results=[
                {"status": "deleted", "id": 10, "name": "Basic"},
                {"status": "error", "id": 20, "name": "Cloze", "error": "in use"},
            ]
        )
        result = run("type", "delete", "Basic", "Cloze", "--yes")
        assert result.exit_code == 0, result.output
        assert "Deleted note type" in result.output
        assert "in use" in result.output

    def test_prompt_cancel_skips_delete(self, run_strict, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types_id_named())

        result = run_strict("type", "delete", "Basic", input="n\n")

        assert result.exit_code == 0
        assert "Cancelled" in result.output
        fake.delete_note_types.assert_not_called()

    def test_prompt_confirm_deletes(self, run_strict, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types_id_named())
        fake.delete_note_types.return_value = DeleteNoteTypesResponse(
            results=[{"status": "deleted", "id": 123, "name": "Basic"}]
        )

        result = run_strict("type", "delete", "Basic", input="y\n")

        assert result.exit_code == 0
        assert "Deleted note type" in result.output
        assert "#123" in result.output
        fake.delete_note_types.assert_called_once_with([123])

    def test_renders_not_found_and_error_results(self, run_strict, fake):
        fake.collection_info.return_value = CollectionInfo(note_types=_note_types_id_named())
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

        result = run_strict("type", "delete", "Basic", "#10", "--yes")

        assert result.exit_code == 0
        assert "Not found: #123" in result.output
        assert "note type is in use" in result.output
        fake.delete_note_types.assert_called_once_with([123, 10])
