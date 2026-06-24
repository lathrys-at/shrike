"""Unit coverage for `shrike note` command rendering + identifier/error branches.

Drives `note list/show/create/update/delete` (and the non-JSON tails of
`tag`/`replace`) through Click's CliRunner with a mocked ShrikeClient — no
server. Targets the missed branches the per-command files don't reach: the
pretty table-vs-detail render fork, the `#id` identifier edge, empty-result
rendering, the inline-creation validation errors, and the JSON-vs-pretty
formatting variants.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from shrike.cli import cli
from shrike.cli import client as cli_client
from shrike.schemas import (
    DeleteNotesResponse,
    FindReplaceResponse,
    ListNotesResponse,
    Note,
    UpdateNoteTagsResponse,
    UpsertNotesResponse,
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
            return runner.invoke(cli, ["--config", str(cfg), *args], **kwargs)

    return _run


def _note(**kw) -> Note:
    base = {
        "id": 1,
        "note_type": "Basic",
        "deck": "Default",
        "tags": [],
        "modified": "2026-01-02T03:04:05",
        "content": None,
    }
    base.update(kw)
    return Note(**base)


class TestNoteList:
    def test_requires_a_filter(self, run, fake) -> None:
        result = run("note", "list")
        assert result.exit_code != 0
        assert "At least one filter is required" in result.output
        fake.list_notes.assert_not_called()

    def test_empty_result_renders_no_notes(self, run, fake) -> None:
        fake.list_notes.return_value = ListNotesResponse(notes=[], total=0)
        result = run("note", "list", "--deck", "Empty")
        assert result.exit_code == 0, result.output
        assert "No notes found" in result.output
        fake.list_notes.assert_called_once()

    def test_detail_render_includes_field_content_and_header(self, run, fake) -> None:
        # Full (non-brief) render: a note with content takes the note_detail branch,
        # and the header reports the deck/type/tag filters that were applied.
        fake.list_notes.return_value = ListNotesResponse(
            notes=[_note(deck="Bio", tags=["t1"], content={"Front": "Q", "Back": "A"})],
            total=1,
        )
        result = run("note", "list", "--deck", "Bio", "--type", "Basic", "--tags", "t1")
        assert result.exit_code == 0, result.output
        assert "in Bio" in result.output
        assert "of type Basic" in result.output
        assert "tagged" in result.output
        assert "Q" in result.output and "A" in result.output

    def test_partial_count_shows_n_of_total(self, run, fake) -> None:
        # total > len(notes) renders "1 of 5"; the brief flag forces the table path.
        fake.list_notes.return_value = ListNotesResponse(
            notes=[_note(content={"Front": "Q"})], total=5
        )
        result = run("note", "list", "--deck", "Bio", "--brief")
        assert result.exit_code == 0, result.output
        assert "1 of 5" in result.output
        assert "#1" in result.output
        assert fake.list_notes.call_args.kwargs["fields"] == "meta"

    def test_no_content_falls_back_to_table(self, run, fake) -> None:
        # Even without --brief, notes with no field content render as a table.
        fake.list_notes.return_value = ListNotesResponse(notes=[_note(content=None)], total=1)
        result = run("note", "list", "--deck", "Bio")
        assert result.exit_code == 0, result.output
        assert "#1" in result.output
        assert "Basic" in result.output

    def test_json_emits_raw_response(self, run, fake) -> None:
        fake.list_notes.return_value = ListNotesResponse(notes=[_note()], total=1)
        result = run("--json", "note", "list", "--deck", "Bio")
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["total"] == 1

    def test_ids_only_filter_has_no_filter_desc(self, run, fake) -> None:
        # Filtering by --ids alone leaves deck/type/tags unset, so the header
        # carries no " in/of/tagged …" descriptor (the empty filter_desc branch).
        fake.list_notes.return_value = ListNotesResponse(notes=[_note(id=7)], total=1)
        result = run("note", "list", "--ids", "7")
        assert result.exit_code == 0, result.output
        assert "in " not in result.output
        assert "of type" not in result.output
        assert "tagged" not in result.output
        assert fake.list_notes.call_args.kwargs["ids"] == [7]


class TestNoteShow:
    def test_not_found_errors(self, run, fake) -> None:
        fake.list_notes.return_value = ListNotesResponse(notes=[], total=0)
        result = run("note", "show", "42")
        assert result.exit_code != 0
        assert "#42 not found" in result.output

    def test_hash_prefix_id_is_accepted(self, run, fake) -> None:
        # The `#id` identifier edge: NOTE_ID strips the leading '#'.
        fake.list_notes.return_value = ListNotesResponse(notes=[_note(id=7)], total=1)
        result = run("note", "show", "#7")
        assert result.exit_code == 0, result.output
        assert fake.list_notes.call_args.kwargs["ids"] == [7]
        assert "#7" in result.output

    def test_json_emits_response(self, run, fake) -> None:
        fake.list_notes.return_value = ListNotesResponse(notes=[_note(id=7)], total=1)
        result = run("--json", "note", "show", "7")
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["notes"][0]["id"] == 7


class TestNoteCreate:
    def test_inline_requires_deck(self, run, fake) -> None:
        result = run("note", "create", "--type", "Basic", "-f", "Front=Q")
        assert result.exit_code != 0
        assert "--deck is required" in result.output
        fake.upsert_notes.assert_not_called()

    def test_inline_requires_type(self, run, fake) -> None:
        result = run("note", "create", "--deck", "D", "-f", "Front=Q")
        assert result.exit_code != 0
        assert "--type is required" in result.output

    def test_inline_requires_a_field(self, run, fake) -> None:
        result = run("note", "create", "--deck", "D", "--type", "Basic")
        assert result.exit_code != 0
        assert "At least one field is required" in result.output

    def test_bad_field_format_errors(self, run, fake) -> None:
        result = run("note", "create", "--deck", "D", "--type", "Basic", "-f", "noequals")
        assert result.exit_code != 0
        assert "Invalid field format" in result.output

    def test_inline_builds_note_with_tags(self, run, fake) -> None:
        fake.upsert_notes.return_value = UpsertNotesResponse(
            results=[{"status": "created", "id": 55}]
        )
        result = run(
            "note", "create", "--deck", "D", "--type", "Basic", "-f", "Front=Q", "--tags", "a,b"
        )
        assert result.exit_code == 0, result.output
        assert "Created note" in result.output and "#55" in result.output
        (notes,), _ = fake.upsert_notes.call_args
        assert notes[0] == {
            "deck": "D",
            "note_type": "Basic",
            "fields": {"Front": "Q"},
            "tags": ["a", "b"],
        }

    def test_json_input_bad_json_errors(self, run, fake) -> None:
        result = run("note", "create", "--json-input", input="{not json")
        assert result.exit_code == 1
        assert "Invalid JSON input" in result.output
        fake.upsert_notes.assert_not_called()

    def test_json_input_single_object_wrapped_in_list(self, run, fake) -> None:
        # A bare object (not an array) on stdin is wrapped into a one-element list.
        fake.upsert_notes.return_value = UpsertNotesResponse(
            results=[{"status": "created", "id": 1}]
        )
        payload = json.dumps({"deck": "D", "note_type": "Basic", "fields": {"Front": "Q"}})
        result = run("note", "create", "--json-input", input=payload)
        assert result.exit_code == 0, result.output
        (notes,), _ = fake.upsert_notes.call_args
        assert isinstance(notes, list) and len(notes) == 1

    def test_json_input_array_passes_through(self, run, fake) -> None:
        # An array on stdin is forwarded as-is (the list-already branch).
        fake.upsert_notes.return_value = UpsertNotesResponse(
            results=[{"status": "created", "id": 1}, {"status": "created", "id": 2}]
        )
        payload = json.dumps(
            [
                {"deck": "D", "note_type": "Basic", "fields": {"Front": "Q1"}},
                {"deck": "D", "note_type": "Basic", "fields": {"Front": "Q2"}},
            ]
        )
        result = run("note", "create", "--json-input", input=payload)
        assert result.exit_code == 0, result.output
        (notes,), _ = fake.upsert_notes.call_args
        assert len(notes) == 2

    def test_json_output(self, run, fake) -> None:
        fake.upsert_notes.return_value = UpsertNotesResponse(
            results=[{"status": "created", "id": 9}]
        )
        result = run("--json", "note", "create", "--deck", "D", "--type", "Basic", "-f", "Front=Q")
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["results"][0]["id"] == 9


class TestNoteUpdate:
    def test_nothing_to_update_errors(self, run, fake) -> None:
        result = run("note", "update", "123")
        assert result.exit_code != 0
        assert "Nothing to update" in result.output
        fake.upsert_notes.assert_not_called()

    def test_builds_payload_from_field_tags_deck(self, run, fake) -> None:
        fake.upsert_notes.return_value = UpsertNotesResponse(
            results=[{"status": "updated", "id": 123}]
        )
        result = run("note", "update", "123", "-f", "Back=New", "--tags", "x,y", "--deck", "Other")
        assert result.exit_code == 0, result.output
        assert "Updated note" in result.output and "#123" in result.output
        (notes,), _ = fake.upsert_notes.call_args
        assert notes[0] == {
            "id": 123,
            "fields": {"Back": "New"},
            "tags": ["x", "y"],
            "deck": "Other",
        }

    def test_json_output(self, run, fake) -> None:
        fake.upsert_notes.return_value = UpsertNotesResponse(
            results=[{"status": "updated", "id": 123}]
        )
        result = run("--json", "note", "update", "123", "-f", "Back=New")
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["results"][0]["status"] == "updated"


class TestNoteDelete:
    def test_confirm_cancel_skips_delete(self, run, fake) -> None:
        result = run("note", "delete", "1", "2", input="n\n")
        assert result.exit_code == 0
        assert "Cancelled" in result.output
        fake.delete_notes.assert_not_called()

    def test_confirm_yes_deletes_and_renders(self, run, fake) -> None:
        fake.delete_notes.return_value = DeleteNotesResponse(deleted=[1, 2], not_found=[])
        result = run("note", "delete", "1", "2", input="y\n")
        assert result.exit_code == 0, result.output
        assert "Deleted 2 note(s)" in result.output
        fake.delete_notes.assert_called_once_with([1, 2])

    def test_yes_flag_renders_deleted_and_not_found(self, run, fake) -> None:
        fake.delete_notes.return_value = DeleteNotesResponse(deleted=[1], not_found=[99])
        result = run("note", "delete", "1", "99", "--yes")
        assert result.exit_code == 0, result.output
        assert "Deleted 1 note(s)" in result.output
        assert "Not found: 99" in result.output

    def test_json_output(self, run, fake) -> None:
        fake.delete_notes.return_value = DeleteNotesResponse(deleted=[1], not_found=[])
        result = run("--json", "note", "delete", "1", "--yes")
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["deleted"] == [1]

    def test_all_not_found_omits_deleted_line(self, run, fake) -> None:
        # Nothing deleted (the empty-deleted branch): only the not-found line shows.
        fake.delete_notes.return_value = DeleteNotesResponse(deleted=[], not_found=[7])
        result = run("note", "delete", "7", "--yes")
        assert result.exit_code == 0, result.output
        assert "Deleted" not in result.output
        assert "Not found: 7" in result.output


class TestNoteTagRender:
    def test_non_json_reports_count_and_not_found(self, run, fake) -> None:
        # The pretty tail: "Updated tags on N note(s)." plus the not-found advisory.
        fake.update_note_tags.return_value = UpdateNoteTagsResponse(
            notes_modified=1, not_found=[42]
        )
        result = run("note", "tag", "1", "42", "--add", "x")
        assert result.exit_code == 0, result.output
        assert "Updated tags on 1 note(s)" in result.output
        assert "Not found" in result.output and "#42" in result.output

    def test_json_output(self, run, fake) -> None:
        fake.update_note_tags.return_value = UpdateNoteTagsResponse(notes_modified=2, not_found=[])
        result = run("--json", "note", "tag", "1", "--add", "x")
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["notes_modified"] == 2


class TestNoteReplaceRender:
    def test_json_mode_applies_directly(self, run, fake) -> None:
        # JSON mode is non-interactive: one call, dry_run honored, raw JSON out.
        fake.find_replace_notes.return_value = FindReplaceResponse(notes_changed=3, dry_run=False)
        result = run("--json", "note", "replace", "a", "b", "--deck", "D")
        assert result.exit_code == 0, result.output
        assert fake.find_replace_notes.call_count == 1
        assert fake.find_replace_notes.call_args.kwargs["dry_run"] is False
        assert json.loads(result.output)["notes_changed"] == 3

    def test_no_matches_short_circuits(self, run, fake) -> None:
        fake.find_replace_notes.return_value = FindReplaceResponse(notes_changed=0, dry_run=True)
        result = run("note", "replace", "a", "b", "--deck", "D")
        assert result.exit_code == 0, result.output
        assert "No matching notes" in result.output
        # Only the preview ran; no apply call.
        assert fake.find_replace_notes.call_count == 1

    def test_preview_truncates_extra_with_more_line(self, run, fake) -> None:
        # notes_changed exceeds the sample count → the "… and N more" line.
        preview = FindReplaceResponse(
            notes_changed=5,
            dry_run=True,
            samples=[{"id": 1, "field": "Front", "before": "teh", "after": "the"}],
        )
        applied = FindReplaceResponse(notes_changed=5, dry_run=False)

        def side(search, replace, *, dry_run=False, **_):
            return preview if dry_run else applied

        fake.find_replace_notes.side_effect = side
        result = run("note", "replace", "teh", "the", "--deck", "D", "--yes")
        assert result.exit_code == 0, result.output
        assert "and 4 more" in result.output
        assert "Replaced in 5" in result.output

    def test_preview_all_sampled_has_no_more_line(self, run, fake) -> None:
        # When the samples cover every change, extra == 0 → no "… and N more" line.
        preview = FindReplaceResponse(
            notes_changed=1,
            dry_run=True,
            samples=[{"id": 1, "field": "Front", "before": "teh", "after": "the"}],
        )
        applied = FindReplaceResponse(notes_changed=1, dry_run=False)

        def side(search, replace, *, dry_run=False, **_):
            return preview if dry_run else applied

        fake.find_replace_notes.side_effect = side
        result = run("note", "replace", "teh", "the", "--deck", "D", "--yes")
        assert result.exit_code == 0, result.output
        assert "more" not in result.output
        assert "Replaced in 1" in result.output


# `shrike note replace` CLI argument handling with the client stubbed.

_REPLACE_PREVIEW = FindReplaceResponse(
    notes_changed=2,
    dry_run=True,
    samples=[{"id": 1, "field": "Front", "before": "teh", "after": "the"}],
)
_REPLACE_APPLIED = FindReplaceResponse(notes_changed=2, dry_run=False)


def _run_replace(args, **kwargs):
    def side(search, replace, *, dry_run=False, **_):
        return _REPLACE_PREVIEW if dry_run else _REPLACE_APPLIED

    with patch("shrike.client.ShrikeClient.find_replace_notes", side_effect=side) as m:
        result = CliRunner().invoke(cli, ["note", "replace", *args], **kwargs)
    return result, m


class TestNoteReplaceCLI:
    def test_requires_scope(self):
        result, m = _run_replace(["teh", "the"])
        assert result.exit_code != 0
        assert "scope" in result.output.lower()
        m.assert_not_called()

    def test_dry_run_only_previews(self):
        result, m = _run_replace(["teh", "the", "--deck", "Bio", "--dry-run"])
        assert result.exit_code == 0
        assert m.call_count >= 1
        assert all(c.kwargs.get("dry_run") is True for c in m.call_args_list)  # no apply

    def test_apply_with_yes(self):
        result, m = _run_replace(["teh", "the", "--deck", "Bio", "--yes"])
        assert result.exit_code == 0
        assert any(c.kwargs.get("dry_run") is False for c in m.call_args_list)  # applied
        assert "Replaced in 2" in result.output

    def test_confirm_cancel(self):
        result, m = _run_replace(["teh", "the", "--deck", "Bio"], input="n\n")
        assert "Cancelled" in result.output
        assert all(c.kwargs.get("dry_run") is True for c in m.call_args_list)  # no apply


# `shrike note tag` CLI argument handling: the set-XOR-add/remove rule and the
# `--set ""` clear path, stubbing ShrikeClient.update_note_tags.

_TAG_OK = UpdateNoteTagsResponse(notes_modified=1, not_found=[])


def _run_tag(*args: str):
    with patch("shrike.client.ShrikeClient.update_note_tags", return_value=_TAG_OK) as m:
        result = CliRunner().invoke(cli, ["note", "tag", *args])
    return result, m


class TestNoteTagValidation:
    def test_set_with_add_is_error(self):
        result, m = _run_tag("123", "--set", "a", "--add", "b")
        assert result.exit_code != 0
        assert "cannot be combined" in result.output
        m.assert_not_called()

    def test_no_mode_is_error(self):
        result, m = _run_tag("123")
        assert result.exit_code != 0
        assert "Specify one of" in result.output
        m.assert_not_called()


class TestNoteTagDispatch:
    def test_set_passes_full_list(self):
        result, m = _run_tag("123", "--set", "a,b")
        assert result.exit_code == 0, result.output
        _, kwargs = m.call_args
        assert kwargs["set"] == ["a", "b"]

    def test_empty_set_clears(self):
        # `--set ""` is a clear, distinct from not passing --set.
        result, m = _run_tag("123", "--set", "")
        assert result.exit_code == 0, result.output
        _, kwargs = m.call_args
        assert kwargs["set"] == []

    def test_add_and_remove_combine(self):
        result, m = _run_tag("123", "--add", "jp", "--add", "verbs", "--remove", "jp-verbs")
        assert result.exit_code == 0, result.output
        _, kwargs = m.call_args
        assert kwargs["add"] == ["jp", "verbs"]
        assert kwargs["remove"] == ["jp-verbs"]
        assert "set" not in kwargs
