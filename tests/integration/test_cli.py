"""CLI integration tests — exercise every CLI command against a live server.

Each test class gets its own isolated server with a fresh collection.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration


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

    def test_list_by_query(self, runner):
        runner.json(
            [
                "note",
                "create",
                "--deck",
                "QueryDeck",
                "--type",
                "Basic",
                "-f",
                "Front=mitochondria",
                "-f",
                "Back=powerhouse",
            ]
        )
        runner.json(
            [
                "note",
                "create",
                "--deck",
                "QueryDeck",
                "--type",
                "Basic",
                "-f",
                "Front=ribosome",
                "-f",
                "Back=protein",
            ]
        )
        data = runner.json(["note", "list", "--query", "mitochondria"])
        assert data["total"] == 1

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
        import time

        runner.json(
            [
                "note",
                "create",
                "--deck",
                "SinceDeck",
                "--type",
                "Basic",
                "-f",
                "Front=Old",
                "-f",
                "Back=Note",
            ]
        )
        time.sleep(1)
        from datetime import UTC, datetime

        cutoff = datetime.now(UTC).isoformat()
        time.sleep(1)
        runner.json(
            [
                "note",
                "create",
                "--deck",
                "SinceDeck",
                "--type",
                "Basic",
                "-f",
                "Front=New",
                "-f",
                "Back=Note",
            ]
        )
        data = runner.json(["note", "list", "--deck", "SinceDeck", "--since", cutoff])
        assert data["total"] == 1
        assert data["notes"][0]["content"]["Front"] == "New"


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
                "Front=Q",
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
                "Front=Q",
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
                "Front=Q",
                "-f",
                "Back=A",
            ]
        )
        note_id = str(created["results"][0]["id"])

        data = runner.json(["note", "update", note_id, "--deck", "MovedDeck"])
        assert data["results"][0]["status"] == "updated"

        note = runner.json(["note", "show", note_id])["notes"][0]
        assert note["deck"] == "MovedDeck"


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
    def test_search_stub(self, runner):
        result = runner.invoke(["note", "search", "test query"])
        assert result.exit_code != 0 or "not available" in result.output.lower()


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
