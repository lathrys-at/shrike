"""CLI argument handling for `shrike note tag` (#73).

Exercises the set-XOR-add/remove rule and the `--set ""` clear path without a
live server, by stubbing ShrikeClient.update_note_tags.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from shrike.cli import cli
from shrike.schemas import UpdateNoteTagsResponse

_OK = UpdateNoteTagsResponse(notes_modified=1, not_found=[])


def _run(*args: str):
    with patch("shrike.client.ShrikeClient.update_note_tags", return_value=_OK) as m:
        result = CliRunner().invoke(cli, ["note", "tag", *args])
    return result, m


class TestNoteTagValidation:
    def test_set_with_add_is_error(self):
        result, m = _run("123", "--set", "a", "--add", "b")
        assert result.exit_code != 0
        assert "cannot be combined" in result.output
        m.assert_not_called()

    def test_no_mode_is_error(self):
        result, m = _run("123")
        assert result.exit_code != 0
        assert "Specify one of" in result.output
        m.assert_not_called()


class TestNoteTagDispatch:
    def test_set_passes_full_list(self):
        result, m = _run("123", "--set", "a,b")
        assert result.exit_code == 0, result.output
        _, kwargs = m.call_args
        assert kwargs["set"] == ["a", "b"]

    def test_empty_set_clears(self):
        # `--set ""` is a clear, distinct from not passing --set.
        result, m = _run("123", "--set", "")
        assert result.exit_code == 0, result.output
        _, kwargs = m.call_args
        assert kwargs["set"] == []

    def test_add_and_remove_combine(self):
        result, m = _run("123", "--add", "jp", "--add", "verbs", "--remove", "jp-verbs")
        assert result.exit_code == 0, result.output
        _, kwargs = m.call_args
        assert kwargs["add"] == ["jp", "verbs"]
        assert kwargs["remove"] == ["jp-verbs"]
        assert "set" not in kwargs
