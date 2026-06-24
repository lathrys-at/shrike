"""CLI integration tests for the `shrike type` command group."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration


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
