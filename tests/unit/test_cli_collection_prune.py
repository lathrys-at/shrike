"""CLI handling for `shrike collection prune` (#89), with the client stubbed."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from shrike.cli import cli
from shrike.schemas import CollectionPruneResponse


def _resp(dry_run):
    return CollectionPruneResponse(
        dry_run=dry_run,
        unused_tags={"removed": 1, "tags": ["orphan"]},
        empty_notes={"removed": [123]},
        empty_cards={"cards_removed": 0, "notes_deleted": []},
    )


def _run(args, **kwargs):
    def side(*, dry_run=True, **_):
        return _resp(dry_run)

    with patch("shrike.client.ShrikeClient.prune", side_effect=side) as m:
        result = CliRunner().invoke(cli, ["collection", "prune", *args], **kwargs)
    return result, m


class TestCollectionPruneCLI:
    def test_default_previews_only(self):
        result, m = _run([])
        assert result.exit_code == 0
        assert m.call_count == 1  # one dry-run call, no apply
        assert m.call_args.kwargs["dry_run"] is True
        assert "Preview only" in result.output
        assert "orphan" in result.output

    def test_no_flags_passes_all_false(self):
        # None selected -> server defaults to all; CLI forwards all-False.
        _, m = _run([])
        assert m.call_args.kwargs["unused_tags"] is False
        assert m.call_args.kwargs["empty_notes"] is False
        assert m.call_args.kwargs["empty_cards"] is False

    def test_flag_selection_forwarded(self):
        _, m = _run(["--unused-tags"])
        assert m.call_args.kwargs["unused_tags"] is True
        assert m.call_args.kwargs["empty_notes"] is False

    def test_apply_with_yes(self):
        result, m = _run(["--apply", "--yes"])
        assert result.exit_code == 0
        assert any(c.kwargs.get("dry_run") is False for c in m.call_args_list)  # applied
        assert "Pruned." in result.output

    def test_apply_confirm_cancel(self):
        result, m = _run(["--apply"], input="n\n")
        assert "Cancelled" in result.output
        assert all(c.kwargs.get("dry_run") is True for c in m.call_args_list)  # never applied

    def test_json_apply_is_noninteractive(self):
        result, m = _run(["--json", "--apply"])
        assert result.exit_code == 0
        assert m.call_count == 1
        assert m.call_args.kwargs["dry_run"] is False  # applied directly, no preview/confirm
