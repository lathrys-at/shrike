"""CLI integration tests for `shrike collection prune`."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestCollectionPrune:
    """`shrike collection prune` via CLI."""

    def test_preview_lists_unused_tag(self, runner):
        runner.json(
            ["note", "create", "--deck", "CP", "--type", "Basic"]
            + ["-f", "Front=cp-preview", "-f", "Back=a", "--tags", "cporphan"]
        )
        nid = runner.json(["note", "list", "--deck", "CP"])["notes"][0]["id"]
        runner.json(["note", "tag", str(nid), "--set", ""])  # orphan "cporphan"

        # --dry-run previews only: JSON shows it as a dry run, nothing mutated.
        preview = runner.json(["collection", "prune", "--unused-tags", "--dry-run"])
        assert preview["dry_run"] is True
        assert "cporphan" in preview["unused_tags"]["tags"]
        # Still in the registry — preview mutated nothing.
        assert "cporphan" in runner.json(["collection", "info", "--tags"])["tags"]

    def test_pretty_dry_run_previews_without_applying(self, runner):
        # Seed an orphan tag so there's something to preview (the per-test reset
        # means we can't rely on another test's data).
        runner.json(
            ["note", "create", "--deck", "CPP", "--type", "Basic"]
            + ["-f", "Front=cpp", "-f", "Back=a", "--tags", "cpporphan"]
        )
        nid = runner.json(["note", "list", "--deck", "CPP"])["notes"][0]["id"]
        runner.json(["note", "tag", str(nid), "--set", ""])

        result = runner.invoke(["collection", "prune", "--unused-tags", "--dry-run"])
        assert result.exit_code == 0
        assert "cpporphan" in result.output
        # --dry-run mutates nothing — the orphan tag is still registered.
        assert "cpporphan" in runner.json(["collection", "info", "--tags"])["tags"]

    def test_apply_clears_unused_tag(self, runner):
        runner.json(
            ["note", "create", "--deck", "CPA", "--type", "Basic"]
            + ["-f", "Front=cp-apply", "-f", "Back=a", "--tags", "cpapply"]
        )
        nid = runner.json(["note", "list", "--deck", "CPA"])["notes"][0]["id"]
        runner.json(["note", "tag", str(nid), "--set", ""])

        # Applies by default (JSON mode is non-interactive — no confirm needed).
        applied = runner.json(["collection", "prune", "--unused-tags"])
        assert applied["dry_run"] is False
        assert "cpapply" not in runner.json(["collection", "info", "--tags"])["tags"]
