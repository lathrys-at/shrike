"""CLI integration tests — exercise every CLI command against a live server.

Each test class gets its own isolated server with a fresh collection.
"""

from __future__ import annotations

import itertools
import json

import pytest

pytestmark = pytest.mark.integration

# Unique first-field values for setup notes. Each test class shares one
# collection, and `shrike note create` defaults to on_duplicate="error", so
# notes created purely as fixtures must not collide on their first field.
_front_counter = itertools.count()


def _unique_front() -> str:
    return f"Front=Q{next(_front_counter)}"


class TestInfo:
    def test_summary_default(self, runner):
        result = runner.invoke(["info"])
        assert result.exit_code == 0
        assert "Collection" in result.output
        assert "Notes" in result.output

    def test_summary_json(self, runner):
        data = runner.json(["info"])
        assert "summary" in data
        assert "notes" in data["summary"]
        assert "cards" in data["summary"]

    def test_decks_only(self, runner):
        result = runner.invoke(["info", "--decks"])
        assert result.exit_code == 0
        assert "Default" in result.output

    def test_types_only(self, runner):
        result = runner.invoke(["info", "--types"])
        assert result.exit_code == 0
        assert "Basic" in result.output

    def test_stats_only(self, runner):
        result = runner.invoke(["info", "--stats"])
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
        data = runner.json(["info", "--stats"])
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
        data = runner.json(["info", "--tags"])
        assert "mytag" in data["tags"]


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

        data = runner.json(["tag", "rename", "history::ww2", "history::wwii"])
        assert data["notes_modified"] == 2
        assert "history::wwii" in self._tags(runner, id1)
        assert "history::wwii" in self._tags(runner, id2)

    def test_rename_scoped_is_exact(self, runner):
        note_id = self._make(runner, "jp,jp-verbs")
        data = runner.json(["tag", "rename", "jp", "japanese", "--note", note_id])
        assert data["notes_modified"] == 1
        assert self._tags(runner, note_id) == ["japanese", "jp-verbs"]


class TestDeckGroup:
    """Deck lifecycle via CLI."""

    def _deck_names(self, runner):
        return {d["name"] for d in runner.json(["info", "--decks"])["decks"]}

    def test_create(self, runner):
        data = runner.json(["deck", "create", "CLIDeck::Sub"])
        assert data["results"][0]["status"] == "created"
        assert "CLIDeck::Sub" in self._deck_names(runner)

    def test_rename(self, runner):
        runner.json(["deck", "create", "RenmeFrom"])
        data = runner.json(["deck", "rename", "RenmeFrom", "RenmeTo"])
        assert data["results"][0]["status"] == "updated"
        names = self._deck_names(runner)
        assert "RenmeTo" in names and "RenmeFrom" not in names

    def test_rename_unknown_errors(self, runner):
        result = runner.invoke(["deck", "rename", "NoSuchDeck", "Whatever"])
        assert result.exit_code != 0

    def test_delete_empty(self, runner):
        runner.json(["deck", "create", "DeleteMe"])
        data = runner.json(["deck", "delete", "DeleteMe", "--yes"])
        assert data["deleted"] == ["DeleteMe"]
        assert "DeleteMe" not in self._deck_names(runner)

    def test_delete_non_empty_refused(self, runner):
        runner.json(
            [
                "note",
                "create",
                "--deck",
                "FullDeck",
                "--type",
                "Basic",
                "-f",
                "Front=Q",
                "-f",
                "Back=A",
            ]
        )
        result = runner.invoke(["deck", "delete", "FullDeck", "--yes"])
        assert result.exit_code != 0
        assert "not empty" in result.output.lower()
        assert "FullDeck" in self._deck_names(runner)

    def _deck_id(self, runner, name):
        return str(
            next(d["id"] for d in runner.json(["info", "--decks"])["decks"] if d["name"] == name)
        )

    def test_rename_and_delete_by_id(self, runner):
        created = runner.json(["deck", "create", "IDDeck"])
        did = str(created["results"][0]["id"])
        # rename by #id (deck id is unchanged by a rename)
        data = runner.json(["deck", "rename", f"#{did}", "IDDeck2"])
        assert data["results"][0]["status"] == "updated"
        assert "IDDeck2" in self._deck_names(runner)
        # delete the (empty) deck by bare numeric id
        out = runner.json(["deck", "delete", did, "--yes"])
        assert out["deleted"] == [did]

    def test_note_list_by_deck_id(self, runner):
        runner.json(
            ["note", "create", "--deck", "NLD", "--type", "Basic", "-f", "Front=q", "-f", "Back=a"]
        )
        did = self._deck_id(runner, "NLD")
        assert runner.json(["note", "list", "--deck", f"#{did}"])["total"] == 1


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


class TestNoteSearch:
    """`note search` on a server with no embeddings: exact substring still works."""

    def _make(self, runner, front: str) -> None:
        runner.json(
            ["note", "create", "--deck", "S", "--type", "Basic"]
            + ["-f", f"Front={front}", "-f", "Back=x"]
        )

    def test_substring_finds_note_json(self, runner):
        self._make(runner, "mitochondria powerhouse")
        self._make(runner, "ribosome protein")
        data = runner.json(["note", "search", "mitochondria"])
        matches = data["results"][0]["matches"]
        assert len(matches) == 1
        # CLI --json drops null fields, so an exact-only hit has no `score` key.
        assert matches[0].get("score") is None
        assert matches[0]["substring"]["matched_fields"] == ["Front"]

    def test_substring_pretty_shows_snippet(self, runner):
        self._make(runner, "electron transport chain")
        result = runner.invoke(["note", "search", "transport"])
        assert result.exit_code == 0
        assert "transport" in result.output
        # semantic ranking is unavailable on this server; that's surfaced (the
        # message wraps in the panel, so match on a single unbroken word)
        assert "unavailable" in result.output.lower()

    def test_requires_an_argument(self, runner):
        result = runner.invoke(["note", "search"])
        assert result.exit_code != 0


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


class TestIndexSave:
    """`shrike index save` against a server with no embedding/index configured."""

    def test_save_empty_json(self, runner):
        data = runner.json(["index", "save"])
        # No embedding service → no index has been built, nothing to persist.
        assert data["status"] == "empty"

    def test_save_empty_pretty(self, runner):
        result = runner.invoke(["index", "save"])
        assert result.exit_code == 0
        assert "no index" in result.output.lower()


class TestTypeList:
    """Listing and inspecting note types via CLI."""

    def test_list_pretty(self, runner):
        result = runner.invoke(["type", "list"])
        assert result.exit_code == 0
        assert "Basic" in result.output
        assert "Cloze" in result.output

    def test_list_json(self, runner):
        data = runner.json(["type", "list"])
        names = {nt["name"] for nt in data}
        assert "Basic" in names
        assert "Cloze" in names

    def test_show_pretty(self, runner):
        result = runner.invoke(["type", "show", "Basic"])
        assert result.exit_code == 0
        assert "Front" in result.output
        assert "Back" in result.output

    def test_show_json(self, runner):
        data = runner.json(["type", "show", "Basic"])
        assert data["name"] == "Basic"
        assert "fields" in data
        assert "templates" in data["detail"]


class TestTypeCreateAndUpdate:
    """Creating and modifying note types via CLI."""

    def test_create_pretty(self, runner):
        result = runner.invoke(
            [
                "type",
                "create",
                "--name",
                "CLIType",
                "--field",
                "Question",
                "--field",
                "Answer",
                "--template",
                "Card 1:{{Question}}:{{FrontSide}}<hr>{{Answer}}",
            ]
        )
        assert result.exit_code == 0
        assert "Created" in result.output

    def test_create_json(self, runner):
        data = runner.json(
            [
                "type",
                "create",
                "--name",
                "CLIType2",
                "--field",
                "Term",
                "--field",
                "Definition",
                "--template",
                "Card 1:{{Term}}:{{FrontSide}}<hr>{{Definition}}",
                "--css",
                ".card { font-size: 18px; }",
            ]
        )
        assert data["results"][0]["status"] == "created"

    def test_create_then_show(self, runner):
        runner.invoke(
            [
                "type",
                "create",
                "--name",
                "Inspectable",
                "--field",
                "Q",
                "--field",
                "A",
                "--template",
                "Card 1:{{Q}}:{{A}}",
            ]
        )
        data = runner.json(["type", "show", "Inspectable"])
        assert data["name"] == "Inspectable"
        assert "Q" in data["fields"]
        assert "A" in data["fields"]

    def test_update_css(self, runner):
        runner.invoke(
            [
                "type",
                "create",
                "--name",
                "Updatable",
                "--field",
                "F",
                "--template",
                "Card 1:{{F}}:{{F}}",
            ]
        )
        data = runner.json(["type", "show", "Updatable"])
        type_id = str(data["id"])

        result = runner.invoke(["type", "update", type_id, "--css", ".card { color: red; }"])
        assert result.exit_code == 0
        assert "Updated" in result.output

    def test_update_name(self, runner):
        runner.invoke(
            [
                "type",
                "create",
                "--name",
                "BeforeRename",
                "--field",
                "F",
                "--template",
                "Card 1:{{F}}:{{F}}",
            ]
        )
        data = runner.json(["type", "show", "BeforeRename"])
        type_id = str(data["id"])

        upd = runner.json(["type", "update", type_id, "--name", "AfterRename"])
        assert upd["results"][0]["status"] == "updated"

    def test_create_note_with_custom_type(self, runner):
        runner.invoke(
            [
                "type",
                "create",
                "--name",
                "Vocab",
                "--field",
                "Word",
                "--field",
                "Meaning",
                "--template",
                "Card 1:{{Word}}:{{Meaning}}",
            ]
        )
        data = runner.json(
            [
                "note",
                "create",
                "--deck",
                "Default",
                "--type",
                "Vocab",
                "-f",
                "Word=shrike",
                "-f",
                "Meaning=A predatory songbird",
            ]
        )
        assert data["results"][0]["status"] == "created"

    def test_update_json_input(self, runner):
        runner.invoke(
            [
                "type",
                "create",
                "--name",
                "JsonUpdatable",
                "--field",
                "X",
                "--field",
                "Y",
                "--template",
                "Card 1:{{X}}:{{Y}}",
            ]
        )
        data = runner.json(["type", "show", "JsonUpdatable"])
        type_id = str(data["id"])

        json_payload = json.dumps(
            {
                "fields": ["X", "Y", "Z"],
                "templates": [{"name": "Card 1", "front": "{{X}}", "back": "{{Y}} {{Z}}"}],
            }
        )
        upd = runner.json(["type", "update", type_id, "--json-input"], input=json_payload)
        assert upd["results"][0]["status"] == "updated"

        updated = runner.json(["type", "show", "JsonUpdatable"])
        assert "Z" in updated["fields"]


class TestTypeShowByID:
    """Showing note types by numeric ID."""

    def test_show_by_id_pretty(self, runner):
        data = runner.json(["type", "list"])
        basic = next(nt for nt in data if nt["name"] == "Basic")
        result = runner.invoke(["type", "show", str(basic["id"])])
        assert result.exit_code == 0
        assert "Basic" in result.output
        assert "Front" in result.output

    def test_show_by_id_json(self, runner):
        data = runner.json(["type", "list"])
        basic = next(nt for nt in data if nt["name"] == "Basic")
        shown = runner.json(["type", "show", str(basic["id"])])
        assert shown["name"] == "Basic"
        assert "templates" in shown["detail"]


class TestTypeUpdateByName:
    """Updating note types by name instead of ID."""

    def test_update_by_name(self, runner):
        runner.invoke(
            [
                "type",
                "create",
                "--name",
                "NameUpdatable",
                "--field",
                "F",
                "--template",
                "Card 1:{{F}}:{{F}}",
            ]
        )
        result = runner.invoke(
            ["type", "update", "NameUpdatable", "--css", ".card { color: green; }"]
        )
        assert result.exit_code == 0
        assert "Updated" in result.output

        data = runner.json(["type", "show", "NameUpdatable"])
        assert "color: green" in data["detail"]["css"]


class TestTypeDelete:
    """Deleting note types by name or ID."""

    def _create_type(self, runner, name):
        data = runner.json(
            [
                "type",
                "create",
                "--name",
                name,
                "--field",
                "F",
                "--template",
                "Card 1:{{F}}:{{F}}",
            ]
        )
        return data["results"][0]["id"]

    def test_delete_by_name(self, runner):
        self._create_type(runner, "DeleteByName")
        data = runner.json(["type", "delete", "DeleteByName", "-y"])
        assert data["results"][0]["status"] == "deleted"

        types = runner.json(["type", "list"])
        assert not any(nt["name"] == "DeleteByName" for nt in types)

    def test_delete_by_id(self, runner):
        nt_id = self._create_type(runner, "DeleteByID")
        data = runner.json(["type", "delete", str(nt_id), "-y"])
        assert data["results"][0]["status"] == "deleted"

    def test_delete_type_with_notes_fails(self, runner):
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
        data = runner.json(["type", "delete", "Basic", "-y"])
        assert data["results"][0]["status"] == "error"
        assert "note(s) use this type" in data["results"][0]["error"]

    def test_delete_nonexistent_name(self, runner):
        result = runner.invoke(["type", "delete", "Nonexistent", "-y"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_delete_nonexistent_id(self, runner):
        result = runner.invoke(["type", "delete", "9999999999", "-y"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


class TestOutputModes:
    """Test --json, --pretty, --no-pretty across commands."""

    def test_json_flag(self, runner):
        result = runner.invoke(["--json", "info"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "summary" in data

    def test_no_pretty(self, runner):
        result = runner.invoke(["--no-pretty", "info"])
        assert result.exit_code == 0

    def test_json_pretty_conflict(self, runner):
        result = runner.invoke(["info", "--json", "--pretty"])
        assert result.exit_code != 0


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
