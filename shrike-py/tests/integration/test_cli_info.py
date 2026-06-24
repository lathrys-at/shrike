"""CLI integration tests for `shrike collection info`."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestInfo:
    def test_summary_default(self, runner):
        result = runner.invoke(["collection", "info"])
        assert result.exit_code == 0
        assert "Collection" in result.output
        assert "Notes" in result.output

    def test_summary_json(self, runner):
        data = runner.json(["collection", "info"])
        assert "summary" in data
        assert "notes" in data["summary"]
        assert "cards" in data["summary"]

    def test_decks_only(self, runner):
        result = runner.invoke(["collection", "info", "--decks"])
        assert result.exit_code == 0
        assert "Default" in result.output

    def test_types_only(self, runner):
        result = runner.invoke(["collection", "info", "--types"])
        assert result.exit_code == 0
        assert "Basic" in result.output

    def test_stats_only(self, runner):
        result = runner.invoke(["collection", "info", "--stats"])
        assert result.exit_code == 0

    def test_stats_reflect_notes(self, runner):
        runner.json(
            [
                "note",
                "create",
                "--deck",
                "Default",
                "--type",
                "Basic",
                "-f",
                "Front=Q",
                "-f",
                "Back=A",
            ]
        )
        data = runner.json(["collection", "info", "--stats"])
        assert data["stats"]["total_notes"] >= 1

    def test_tags_reflect_notes(self, runner):
        runner.json(
            [
                "note",
                "create",
                "--deck",
                "Default",
                "--type",
                "Basic",
                "-f",
                "Front=Q2",
                "-f",
                "Back=A2",
                "--tags",
                "mytag",
            ]
        )
        data = runner.json(["collection", "info", "--tags"])
        assert "mytag" in data["tags"]
