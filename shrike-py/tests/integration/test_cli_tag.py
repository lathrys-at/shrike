"""CLI integration tests for collection-level tag ops (`shrike collection tag`)."""

from __future__ import annotations

import pytest

from tests.integration.conftest import _unique_front

pytestmark = pytest.mark.integration


class TestTagGroup:
    """Collection-level tag ops: rename and clean."""

    def _make(self, runner, tags):
        created = runner.json(
            [
                "note",
                "create",
                "--deck",
                "Default",
                "--type",
                "Basic",
                "-f",
                _unique_front(),
                "-f",
                "Back=A",
                "--tags",
                tags,
            ]
        )
        return str(created["results"][0]["id"])

    def _tags(self, runner, note_id):
        return sorted(runner.json(["note", "show", note_id])["notes"][0]["tags"])

    def test_rename_collection_wide(self, runner):
        id1 = self._make(runner, "history::ww2")
        id2 = self._make(runner, "history::ww2,other")

        data = runner.json(["collection", "tag", "rename", "history::ww2", "history::wwii"])
        assert data["notes_modified"] == 2
        assert "history::wwii" in self._tags(runner, id1)
        assert "history::wwii" in self._tags(runner, id2)

    def test_rename_scoped_is_exact(self, runner):
        note_id = self._make(runner, "jp,jp-verbs")
        data = runner.json(["collection", "tag", "rename", "jp", "japanese", "--note", note_id])
        assert data["notes_modified"] == 1
        assert self._tags(runner, note_id) == ["japanese", "jp-verbs"]
