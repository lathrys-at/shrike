"""CLI integration tests for `shrike completion` (shell completion scripts)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestCompletion:
    """Shell completion script generation."""

    def test_zsh(self, runner):
        result = runner.invoke(["completion", "zsh"])
        assert result.exit_code == 0
        assert "#compdef shrike" in result.output

    def test_bash(self, runner):
        result = runner.invoke(["completion", "bash"])
        assert result.exit_code == 0
        assert "_shrike_completion" in result.output

    def test_fish(self, runner):
        result = runner.invoke(["completion", "fish"])
        assert result.exit_code == 0
        assert "complete" in result.output
        assert "shrike" in result.output

    def test_invalid_shell(self, runner):
        result = runner.invoke(["completion", "powershell"])
        assert result.exit_code != 0
