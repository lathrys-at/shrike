"""CLI integration tests for the `shrike note` command group."""

from __future__ import annotations

import json

import pytest

from tests.integration.conftest import _unique_front

pytestmark = pytest.mark.integration


class TestNoteCreate:
    """Note creation via CLI — inline fields and JSON stdin."""

    def test_create_pretty(self, runner):
        result = runner.invoke(
            [
                "note",
                "create",
                "--deck",
                "Test",
                "--type",
                "Basic",
                "-f",
                "Front=What is Shrike?",
                "-f",
                "Back=An Anki manager",
            ]
        )
        assert result.exit_code == 0
        assert "Created" in result.output

    def test_create_json(self, runner):
        data = runner.json(
            [
                "note",
                "create",
                "--deck",
                "Test",
                "--type",
                "Basic",
                "-f",
                "Front=Question",
                "-f",
                "Back=Answer",
            ]
        )
        assert data["results"][0]["status"] == "created"
        assert "id" in data["results"][0]

    def test_create_with_tags(self, runner):
        data = runner.json(
            [
                "note",
                "create",
                "--deck",
                "Test",
                "--type",
                "Basic",
                "-f",
                "Front=Tagged Q",
                "-f",
                "Back=Tagged A",
                "--tags",
                "alpha,beta",
            ]
        )
        assert data["results"][0]["status"] == "created"

        note_id = str(data["results"][0]["id"])
        note_data = runner.json(["note", "show", note_id])
        tags = set(note_data["notes"][0]["tags"])
        assert "alpha" in tags
        assert "beta" in tags

    def test_create_bulk_stdin(self, runner):
        notes = json.dumps(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": f"Bulk Q{i}", "Back": f"Bulk A{i}"},
                }
                for i in range(5)
            ]
        )
        result = runner.invoke(["note", "create", "--json-input"], input=notes)
        assert result.exit_code == 0

        data = runner.json(["note", "list", "--deck", "Test"])
        assert data["total"] >= 5

    def test_create_auto_creates_deck(self, runner):
        runner.json(
            [
                "note",
                "create",
                "--deck",
                "Brand New Deck",
                "--type",
                "Basic",
                "-f",
                "Front=Q",
                "-f",
                "Back=A",
            ]
        )
        data = runner.json(["note", "list", "--deck", "Brand New Deck"])
        assert data["total"] == 1


class TestNoteListAndShow:
    """Listing and showing notes via CLI."""

    def test_list_empty_deck(self, runner):
        data = runner.json(["note", "list", "--deck", "Empty"])
        assert data["notes"] == []
        assert data["total"] == 0

    def test_list_empty_pretty(self, runner):
        result = runner.invoke(["note", "list", "--deck", "Empty"])
        assert result.exit_code == 0

    def test_list_by_deck(self, runner):
        for i in range(3):
            runner.json(
                [
                    "note",
                    "create",
                    "--deck",
                    "ListDeck",
                    "--type",
                    "Basic",
                    "-f",
                    f"Front=Q{i}",
                    "-f",
                    f"Back=A{i}",
                ]
            )
        data = runner.json(["note", "list", "--deck", "ListDeck"])
        assert data["total"] == 3

    def test_list_by_tags(self, runner):
        runner.json(
            [
                "note",
                "create",
                "--deck",
                "Default",
                "--type",
                "Basic",
                "-f",
                "Front=Tagged",
                "-f",
                "Back=Note",
                "--tags",
                "findme",
            ]
        )
        runner.json(
            [
                "note",
                "create",
                "--deck",
                "Default",
                "--type",
                "Basic",
                "-f",
                "Front=Other",
                "-f",
                "Back=Note",
                "--tags",
                "other",
            ]
        )
        data = runner.json(["note", "list", "--tags", "findme"])
        assert data["total"] == 1

    def test_show_pretty(self, runner):
        created = runner.json(
            [
                "note",
                "create",
                "--deck",
                "Default",
                "--type",
                "Basic",
                "-f",
                "Front=Show Q",
                "-f",
                "Back=Show A",
            ]
        )
        note_id = str(created["results"][0]["id"])
        result = runner.invoke(["note", "show", note_id])
        assert result.exit_code == 0
        assert note_id in result.output
        assert "Show Q" in result.output

    def test_show_json(self, runner):
        created = runner.json(
            [
                "note",
                "create",
                "--deck",
                "Default",
                "--type",
                "Basic",
                "-f",
                "Front=JSON Q",
                "-f",
                "Back=JSON A",
            ]
        )
        note_id = str(created["results"][0]["id"])
        data = runner.json(["note", "show", note_id])
        assert len(data["notes"]) == 1
        assert data["notes"][0]["content"]["Front"] == "JSON Q"

    def test_show_nonexistent(self, runner):
        result = runner.invoke(["note", "show", "999999999"])
        assert result.exit_code != 0

    def test_list_meta_flag(self, runner):
        runner.json(
            [
                "note",
                "create",
                "--deck",
                "MetaDeck",
                "--type",
                "Basic",
                "-f",
                "Front=Q",
                "-f",
                "Back=A",
            ]
        )
        data = runner.json(["note", "list", "--deck", "MetaDeck", "--brief"])
        assert data["total"] == 1
        note = data["notes"][0]
        assert "content" not in note
        assert "note_type" in note

    def test_list_since(self, runner):
        # End-to-end plumbing of `--since` both ways (the boundary logic itself
        # is unit-tested deterministically, so no real-time sleep here).
        runner.json(
            [
                "note",
                "create",
                "--deck",
                "SinceDeck",
                "--type",
                "Basic",
                "-f",
                "Front=Note",
                "-f",
                "Back=Body",
            ]
        )
        present = runner.json(
            ["note", "list", "--deck", "SinceDeck", "--since", "2000-01-01T00:00:00Z"]
        )
        assert present["total"] == 1
        assert present["notes"][0]["content"]["Front"] == "Note"

        excluded = runner.json(
            ["note", "list", "--deck", "SinceDeck", "--since", "2099-01-01T00:00:00Z"]
        )
        assert excluded["total"] == 0


class TestNoteUpdate:
    """Updating notes via CLI."""

    def test_update_field(self, runner):
        created = runner.json(
            [
                "note",
                "create",
                "--deck",
                "Default",
                "--type",
                "Basic",
                "-f",
                "Front=Original Q",
                "-f",
                "Back=Original A",
            ]
        )
        note_id = str(created["results"][0]["id"])

        result = runner.invoke(["note", "update", note_id, "-f", "Back=Updated A"])
        assert result.exit_code == 0
        assert "Updated" in result.output

        data = runner.json(["note", "show", note_id])
        assert data["notes"][0]["content"]["Back"] == "Updated A"
        assert data["notes"][0]["content"]["Front"] == "Original Q"

    def test_update_json(self, runner):
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
            ]
        )
        note_id = str(created["results"][0]["id"])

        data = runner.json(["note", "update", note_id, "-f", "Back=New A"])
        assert data["results"][0]["status"] == "updated"

    def test_update_tags(self, runner):
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
                "old",
            ]
        )
        note_id = str(created["results"][0]["id"])

        data = runner.json(["note", "update", note_id, "--tags", "new,replaced"])
        assert data["results"][0]["status"] == "updated"

        note = runner.json(["note", "show", note_id])["notes"][0]
        assert "new" in note["tags"]
        assert "replaced" in note["tags"]
        assert "old" not in note["tags"]

    def test_update_deck(self, runner):
        created = runner.json(
            [
                "note",
                "create",
                "--deck",
                "OrigDeck",
                "--type",
                "Basic",
                "-f",
                _unique_front(),
                "-f",
                "Back=A",
            ]
        )
        note_id = str(created["results"][0]["id"])

        data = runner.json(["note", "update", note_id, "--deck", "MovedDeck"])
        assert data["results"][0]["status"] == "updated"

        note = runner.json(["note", "show", note_id])["notes"][0]
        assert note["deck"] == "MovedDeck"


class TestNoteTag:
    """set/add/remove tag editing via CLI."""

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

    def test_set_replaces_across_multiple_notes(self, runner):
        id1 = self._make(runner, "old1")
        id2 = self._make(runner, "old2")

        data = runner.json(["note", "tag", id1, id2, "--set", "shared,history"])
        assert data["notes_modified"] == 2

        for note_id in (id1, id2):
            assert self._tags(runner, note_id) == ["history", "shared"]

    def test_set_clears_with_empty_set(self, runner):
        note_id = self._make(runner, "keep,me")
        runner.json(["note", "tag", note_id, "--set", ""])
        assert self._tags(runner, note_id) == []

    def test_add_leaves_others_intact(self, runner):
        note_id = self._make(runner, "base")
        runner.json(["note", "tag", note_id, "--add", "extra"])
        assert self._tags(runner, note_id) == ["base", "extra"]

    def test_remove_leaves_others_intact(self, runner):
        note_id = self._make(runner, "base,drop")
        runner.json(["note", "tag", note_id, "--remove", "drop"])
        assert self._tags(runner, note_id) == ["base"]

    def test_add_and_remove_combine(self, runner):
        note_id = self._make(runner, "jp-verbs,keep")
        runner.json(
            ["note", "tag", note_id, "--add", "jp", "--add", "verbs", "--remove", "jp-verbs"]
        )
        assert self._tags(runner, note_id) == ["jp", "keep", "verbs"]

    def test_requires_a_mode(self, runner):
        note_id = self._make(runner, "x")
        result = runner.invoke(["note", "tag", note_id])
        assert result.exit_code != 0

    def test_set_and_add_conflict(self, runner):
        note_id = self._make(runner, "x")
        result = runner.invoke(["note", "tag", note_id, "--set", "a", "--add", "b"])
        assert result.exit_code != 0


class TestNoteDelete:
    """Deleting notes via CLI."""

    def test_delete_pretty(self, runner):
        created = runner.json(
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
        note_id = str(created["results"][0]["id"])

        result = runner.invoke(["note", "delete", note_id, "--yes"])
        assert result.exit_code == 0
        assert "Deleted" in result.output

    def test_delete_json(self, runner):
        created = runner.json(
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
        note_id = str(created["results"][0]["id"])

        data = runner.json(["note", "delete", note_id, "--yes"])
        assert len(data["deleted"]) == 1

    def test_delete_verified_gone(self, runner):
        created = runner.json(
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
        note_id = str(created["results"][0]["id"])

        runner.json(["note", "delete", note_id, "--yes"])
        result = runner.invoke(["note", "show", note_id])
        assert result.exit_code != 0

    def test_delete_multiple_ids(self, runner):
        ids = []
        for i in range(3):
            created = runner.json(
                [
                    "note",
                    "create",
                    "--deck",
                    "Default",
                    "--type",
                    "Basic",
                    "-f",
                    f"Front=Multi{i}",
                    "-f",
                    f"Back=Del{i}",
                ]
            )
            ids.append(str(created["results"][0]["id"]))

        data = runner.json(["note", "delete", *ids, "--yes"])
        assert len(data["deleted"]) == 3


class TestNoteReplace:
    """`note replace` find-and-replace via CLI."""

    def _make(self, runner, deck: str, front: str, back: str = "x") -> None:
        runner.json(
            ["note", "create", "--deck", deck, "--type", "Basic"]
            + ["-f", f"Front={front}", "-f", f"Back={back}"]
        )

    def test_apply_json(self, runner):
        self._make(runner, "Rep", "teh cell", "teh power")
        applied = runner.json(["note", "replace", "teh", "the", "--deck", "Rep"])
        assert applied["dry_run"] is False
        assert applied["notes_changed"] == 1
        note = runner.json(["note", "list", "--deck", "Rep"])["notes"][0]
        assert note["content"]["Front"] == "the cell"
        assert note["content"]["Back"] == "the power"

    def test_dry_run_changes_nothing(self, runner):
        self._make(runner, "Rep2", "teh keep")
        dry = runner.json(["note", "replace", "teh", "the", "--deck", "Rep2", "--dry-run"])
        assert dry["dry_run"] is True
        assert dry["notes_changed"] == 1
        note = runner.json(["note", "list", "--deck", "Rep2"])["notes"][0]
        assert note["content"]["Front"] == "teh keep"  # untouched

    def test_pretty_confirm_apply(self, runner):
        self._make(runner, "Rep3", "teh pretty")
        result = runner.invoke(["note", "replace", "teh", "the", "--deck", "Rep3"], input="y\n")
        assert result.exit_code == 0
        assert "Replaced in 1" in result.output

    def test_requires_scope(self, runner):
        result = runner.invoke(["note", "replace", "a", "b"])
        assert result.exit_code != 0
