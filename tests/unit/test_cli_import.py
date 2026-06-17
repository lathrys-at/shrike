"""CLI behavior for `shrike collection import` (#72 S3; rehomed in #683), client stubbed."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from shrike.cli import cli
from shrike.schemas import ImportPackageResponse


def _run(tmp_path, args, **kwargs):
    cfg = tmp_path / "config.yml"
    cfg.write_text("")
    fake = MagicMock()
    fake.import_package.return_value = ImportPackageResponse(
        new=3, updated=1, found_notes=4, reindexed=True
    )
    with patch("shrike.client.ShrikeClient", return_value=fake):
        result = CliRunner().invoke(
            cli, ["--config", str(cfg), "collection", "import", *args], **kwargs
        )
    return result, fake


class TestImportCommand:
    def test_sends_absolutized_path_and_defaults(self, tmp_path):
        result, fake = _run(tmp_path, ["deck.apkg"])
        assert result.exit_code == 0, result.output
        args, kwargs = fake.import_package.call_args
        # Path is absolutized client-side.
        assert os.path.isabs(args[0])
        assert args[0].endswith("deck.apkg")
        # Defaults.
        assert kwargs["update_notes"] == "if_newer"
        assert kwargs["update_notetypes"] == "if_newer"
        assert kwargs["with_scheduling"] is False
        assert kwargs["merge_notetypes"] is False

    def test_renders_summary(self, tmp_path):
        result, _ = _run(tmp_path, ["/abs/deck.apkg"])
        assert result.exit_code == 0, result.output
        assert "3" in result.output  # new
        assert "imported" in result.output.lower()

    def test_options_forwarded(self, tmp_path):
        result, fake = _run(
            tmp_path,
            [
                "/abs/deck.apkg",
                "--update-notes",
                "always",
                "--update-notetypes",
                "never",
                "--with-scheduling",
                "--merge-notetypes",
            ],
        )
        assert result.exit_code == 0, result.output
        _, kwargs = fake.import_package.call_args
        assert kwargs["update_notes"] == "always"
        assert kwargs["update_notetypes"] == "never"
        assert kwargs["with_scheduling"] is True
        assert kwargs["merge_notetypes"] is True

    def test_bad_condition_rejected_by_click(self, tmp_path):
        result, fake = _run(tmp_path, ["/abs/deck.apkg", "--update-notes", "bogus"])
        assert result.exit_code != 0
        assert "bogus" in result.output or "invalid" in result.output.lower()
        fake.import_package.assert_not_called()

    def test_json_output(self, tmp_path):
        result, _ = _run(tmp_path, ["/abs/deck.apkg", "--json"])
        assert result.exit_code == 0, result.output
        assert '"new": 3' in result.output
        assert '"reindexed": true' in result.output
