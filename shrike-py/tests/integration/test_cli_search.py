"""CLI integration tests for `shrike search` (substring on a no-embedding server)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestNoteSearch:
    """`shrike search` on a server with no embeddings: exact substring still works."""

    def _make(self, runner, front: str) -> None:
        runner.json(
            ["note", "create", "--deck", "S", "--type", "Basic"]
            + ["-f", f"Front={front}", "-f", "Back=x"]
        )

    def test_substring_finds_note_json(self, runner):
        self._make(runner, "mitochondria powerhouse")
        self._make(runner, "ribosome protein")
        data = runner.json(["search", "mitochondria"])
        matches = data["results"][0]["matches"]
        assert len(matches) == 1
        # CLI --json drops null fields, so an exact-only hit has no `score` key.
        assert matches[0].get("score") is None
        assert matches[0]["substring"]["matched_fields"] == ["Front"]

    def test_substring_pretty_shows_snippet(self, runner):
        self._make(runner, "electron transport chain")
        result = runner.invoke(["search", "transport"])
        assert result.exit_code == 0
        assert "transport" in result.output
        # semantic ranking is unavailable on this server; that's surfaced (the
        # message wraps in the panel, so match on a single unbroken word)
        assert "unavailable" in result.output.lower()

    def test_requires_an_argument(self, runner):
        result = runner.invoke(["search"])
        assert result.exit_code != 0
