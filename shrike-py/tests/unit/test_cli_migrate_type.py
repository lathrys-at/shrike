"""CLI handling for `shrike note migrate-type`, with the client stubbed."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from shrike.cli import cli
from shrike.schemas import MigrateNoteTypeResponse


def _resp(dry_run):
    return MigrateNoteTypeResponse(
        changed=[1, 2],
        from_note_type="Basic",
        to_note_type="Cloze",
        dropped_fields=["Back"],
        new_empty_fields=["Extra"],
        dry_run=dry_run,
    )


def _run(args, **kwargs):
    def side(note_ids, to, field_map, *, template_map=None, dry_run=False):
        return _resp(dry_run)

    with patch("shrike.client.ShrikeClient.migrate_note_type", side_effect=side) as m:
        result = CliRunner().invoke(cli, ["note", "migrate-type", *args], **kwargs)
    return result, m


class TestMigrateTypeCLI:
    def test_requires_map(self):
        result, m = _run(["1", "--to", "Cloze"])
        assert result.exit_code != 0
        assert "--map" in result.output

    def test_map_parsed_into_dict(self):
        _run(["1", "--to", "Cloze", "--map", "Front=Text", "--map", "Back=Back Extra", "--yes"])
        # first (preview) call carries the parsed field map
        with patch("shrike.client.ShrikeClient.migrate_note_type") as m:
            m.return_value = _resp(True)
            CliRunner().invoke(
                cli,
                ["note", "migrate-type", "1", "--to", "Cloze", "--map", "Front=Text", "--dry-run"],
            )
        assert m.call_args.args[2] == {"Front": "Text"}

    def test_dry_run_previews_drops_no_apply(self):
        result, m = _run(["1", "--to", "Cloze", "--map", "Front=Text", "--dry-run"])
        assert result.exit_code == 0
        assert "Back" in result.output  # dropped field shown
        assert all(c.kwargs.get("dry_run") is True for c in m.call_args_list)  # never applied

    def test_apply_with_yes(self):
        result, m = _run(["1", "--to", "Cloze", "--map", "Front=Text", "--yes"])
        assert result.exit_code == 0
        assert any(c.kwargs.get("dry_run") is False for c in m.call_args_list)
        assert "Migrated 2" in result.output

    def test_confirm_cancel(self):
        result, m = _run(["1", "--to", "Cloze", "--map", "Front=Text"], input="n\n")
        assert "Cancelled" in result.output
        assert all(c.kwargs.get("dry_run") is True for c in m.call_args_list)

    def test_template_map_forwarded(self):
        _, m = _run(
            ["1", "--to", "Cloze", "--map", "Front=Text", "--template-map", "Card 1=Cloze", "--yes"]
        )
        assert m.call_args_list[0].kwargs["template_map"] == {"Card 1": "Cloze"}

    def test_json_applies_directly(self):
        result, m = _run(["--json", "1", "--to", "Cloze", "--map", "Front=Text"])
        assert result.exit_code == 0
        assert m.call_count == 1
        assert m.call_args.kwargs["dry_run"] is False

    def test_clean_migration_omits_drop_and_empty_lines(self):
        # No dropped/new-empty fields → neither advisory line is printed (the
        # negative side of both branches), just the count + type transition.
        clean = MigrateNoteTypeResponse(
            changed=[1],
            from_note_type="Basic",
            to_note_type="Cloze",
            dropped_fields=[],
            new_empty_fields=[],
            dry_run=False,
        )
        with patch("shrike.client.ShrikeClient.migrate_note_type", return_value=clean):
            result = CliRunner().invoke(
                cli, ["note", "migrate-type", "1", "--to", "Cloze", "--map", "Front=Text", "--yes"]
            )
        assert result.exit_code == 0, result.output
        assert "drops (content lost)" not in result.output
        assert "empty in target" not in result.output
        assert "Migrated 1" in result.output
