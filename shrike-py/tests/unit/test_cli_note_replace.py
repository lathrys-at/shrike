"""CLI handling for `shrike note replace` (#85), with the client stubbed."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from shrike.cli import cli
from shrike.schemas import FindReplaceResponse

_PREVIEW = FindReplaceResponse(
    notes_changed=2,
    dry_run=True,
    samples=[{"id": 1, "field": "Front", "before": "teh", "after": "the"}],
)
_APPLIED = FindReplaceResponse(notes_changed=2, dry_run=False)


def _run(args, **kwargs):
    def side(search, replace, *, dry_run=False, **_):
        return _PREVIEW if dry_run else _APPLIED

    with patch("shrike.client.ShrikeClient.find_replace_notes", side_effect=side) as m:
        result = CliRunner().invoke(cli, ["note", "replace", *args], **kwargs)
    return result, m


class TestNoteReplaceCLI:
    def test_requires_scope(self):
        result, m = _run(["teh", "the"])
        assert result.exit_code != 0
        assert "scope" in result.output.lower()
        m.assert_not_called()

    def test_dry_run_only_previews(self):
        result, m = _run(["teh", "the", "--deck", "Bio", "--dry-run"])
        assert result.exit_code == 0
        assert m.call_count >= 1
        assert all(c.kwargs.get("dry_run") is True for c in m.call_args_list)  # no apply

    def test_apply_with_yes(self):
        result, m = _run(["teh", "the", "--deck", "Bio", "--yes"])
        assert result.exit_code == 0
        assert any(c.kwargs.get("dry_run") is False for c in m.call_args_list)  # applied
        assert "Replaced in 2" in result.output

    def test_confirm_cancel(self):
        result, m = _run(["teh", "the", "--deck", "Bio"], input="n\n")
        assert "Cancelled" in result.output
        assert all(c.kwargs.get("dry_run") is True for c in m.call_args_list)  # no apply
