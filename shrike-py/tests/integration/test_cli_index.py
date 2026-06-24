"""CLI integration tests for `shrike server index save`."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestIndexSave:
    """`shrike server index save` against a server with no embedding/index configured."""

    def test_save_empty_json(self, runner):
        data = runner.json(["server", "index", "save"])
        # No embedding service → no index has been built, nothing to persist.
        assert data["status"] == "empty"

    def test_save_empty_pretty(self, runner):
        result = runner.invoke(["server", "index", "save"])
        assert result.exit_code == 0
        assert "no index" in result.output.lower()
