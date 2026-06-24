"""CLI integration tests for global output modes (--json, --pretty, --no-pretty)."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration


class TestOutputModes:
    """Test --json, --pretty, --no-pretty across commands."""

    def test_json_flag(self, runner):
        result = runner.invoke(["--json", "collection", "info"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "summary" in data

    def test_no_pretty(self, runner):
        result = runner.invoke(["--no-pretty", "collection", "info"])
        assert result.exit_code == 0

    def test_json_pretty_conflict(self, runner):
        result = runner.invoke(["collection", "info", "--json", "--pretty"])
        assert result.exit_code != 0
