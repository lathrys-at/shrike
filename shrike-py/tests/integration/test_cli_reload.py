"""CLI integration tests for `shrike collection reload`."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestCollectionReload:
    """`shrike collection reload` via CLI (the /reload control endpoint)."""

    def test_reload_json(self, runner):
        result = runner.json(["collection", "reload"])
        assert result["status"] == "reloaded"
        assert isinstance(result["col_mod"], int)
        # No embedder in the non-embedding lane, so nothing to rebuild.
        assert result["rebuilding"] is False
        # Collection is still usable through the re-opened handle.
        summary = runner.json(["collection", "info"])["summary"]
        assert "notes" in summary

    def test_reload_then_write(self, runner):
        runner.json(["collection", "reload"])
        # A write after a reload goes to the re-opened collection.
        runner.json(
            ["note", "create", "--deck", "RLD", "--type", "Basic"]
            + ["-f", "Front=after-reload", "-f", "Back=x"]
        )
        assert runner.json(["note", "list", "--deck", "RLD"])["total"] == 1

    def test_reload_pretty(self, runner):
        result = runner.invoke(["collection", "reload"])
        assert result.exit_code == 0
        assert "Reloaded" in result.output
