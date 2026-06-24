"""CLI integration tests for `shrike search query` (raw Anki search)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestCollectionQuery:
    """`shrike search query` via CLI. Each test seeds its own note (xdist)."""

    def _make(self, runner, deck, front, tag):
        runner.json(
            ["note", "create", "--deck", deck, "--type", "Basic"]
            + ["-f", f"Front={front}", "-f", "Back=x", "--tags", tag]
        )

    def test_query_json(self, runner):
        self._make(runner, "CQ", "cq-card", "cqtag")
        result = runner.json(["search", "query", "tag:cqtag"])
        assert result["total"] == 1
        assert result["notes"][0]["content"]["Front"] == "cq-card"

    def test_query_brief_pretty(self, runner):
        self._make(runner, "CQB", "cqb-card", "cqbrief")
        result = runner.invoke(["search", "query", "tag:cqbrief", "--brief"])
        assert result.exit_code == 0
        assert "cqbrief" in result.output  # header echoes the expression

    def test_malformed_query_errors(self, runner):
        result = runner.invoke(["search", "query", "(unbalanced"])
        assert result.exit_code != 0
