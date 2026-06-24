"""CLI integration tests for `shrike note migrate-type`."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestMigrateType:
    """`shrike note migrate-type` via CLI. Self-seeding (shared-server reset)."""

    def _basic(self, runner, deck, front, back="x"):
        runner.json(
            ["note", "create", "--deck", deck, "--type", "Basic"]
            + ["-f", f"Front={front}", "-f", f"Back={back}"]
        )
        return runner.json(["note", "list", "--deck", deck])["notes"][0]["id"]

    def test_apply_json(self, runner):
        nid = self._basic(runner, "MT", "mt-front", "mt-back")
        result = runner.json(
            ["note", "migrate-type", str(nid), "--to", "Cloze"]
            + ["--map", "Front=Text", "--map", "Back=Back Extra"]
        )
        assert result["dry_run"] is False
        assert result["to_note_type"] == "Cloze"
        note = runner.json(["note", "show", str(nid)])["notes"][0]
        assert note["note_type"] == "Cloze"
        assert note["content"]["Text"] == "mt-front"

    def test_dry_run_shows_drop(self, runner):
        nid = self._basic(runner, "MTD", "mtd-front", "mtd-back")
        result = runner.invoke(
            ["note", "migrate-type", str(nid), "--to", "Cloze", "--map", "Front=Text", "--dry-run"]
        )
        assert result.exit_code == 0
        assert "Back" in result.output  # dropped field
        assert runner.json(["note", "show", str(nid)])["notes"][0]["note_type"] == "Basic"
