"""CLI integration tests — exercise every CLI command against a live server."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from shrike.cli import cli


@pytest.fixture(scope="module")
def cli_config(server, tmp_path_factory):
    """Create an isolated config file pointing at the test server."""
    config_dir = tmp_path_factory.mktemp("cli-config")
    config_path = config_dir / "config.yml"
    config_path.write_text(
        f"server:\n"
        f"  host: 127.0.0.1\n"
        f"  port: {server['port']}\n"
        f"collection: {server['collection_path']}\n"
        f"logging:\n"
        f"  dir: {server['log_dir']}\n"
    )
    return config_path


@pytest.fixture(scope="module")
def runner(server, cli_config):
    """CliRunner that targets the test server with an isolated config."""
    url = server["url"]
    config = str(cli_config)

    class ServerRunner:
        def __init__(self) -> None:
            self._runner = CliRunner()
            self._url = url
            self._config = config

        def invoke(self, args: list[str], **kwargs) -> object:
            return self._runner.invoke(
                cli,
                ["--config", self._config, "--url", self._url, *args],
                catch_exceptions=False,
                **kwargs,
            )

        def json(self, args: list[str], **kwargs) -> dict:
            result = self.invoke(["--json", *args], **kwargs)
            assert result.exit_code == 0, result.output
            return json.loads(result.output)

    return ServerRunner()


@pytest.mark.integration
class TestInfo:
    def test_info_pretty(self, runner):
        result = runner.invoke(["info"])
        assert result.exit_code == 0
        assert "Basic" in result.output
        assert "Default" in result.output

    def test_info_json(self, runner):
        data = runner.json(["info"])
        assert "note_types" in data
        assert "decks" in data
        assert "tags" in data
        assert "stats" in data
        assert any(nt["name"] == "Basic" for nt in data["note_types"])

    def test_info_decks_only(self, runner):
        result = runner.invoke(["info", "--decks"])
        assert result.exit_code == 0
        assert "Default" in result.output

    def test_info_types_only(self, runner):
        result = runner.invoke(["info", "--types"])
        assert result.exit_code == 0
        assert "Basic" in result.output

    def test_info_stats_only(self, runner):
        result = runner.invoke(["info", "--stats"])
        assert result.exit_code == 0
        assert "Notes" in result.output or "notes" in result.output.lower()


@pytest.mark.integration
class TestNoteLifecycle:
    """Test the full note create -> list -> show -> update -> delete cycle."""

    def test_01_list_empty(self, runner):
        result = runner.invoke(["note", "list", "--deck", "Default"])
        assert result.exit_code == 0

    def test_02_list_empty_json(self, runner):
        data = runner.json(["note", "list", "--deck", "Default"])
        assert data["notes"] == []
        assert data["total"] == 0

    def test_03_create_note(self, runner):
        result = runner.invoke(
            [
                "note",
                "create",
                "--deck",
                "CLITest",
                "--type",
                "Basic",
                "-f",
                "Front=What is CLI testing?",
                "-f",
                "Back=Testing commands end-to-end",
            ]
        )
        assert result.exit_code == 0
        assert "Created" in result.output

    def test_04_create_note_json(self, runner):
        data = runner.json(
            [
                "note",
                "create",
                "--deck",
                "CLITest",
                "--type",
                "Basic",
                "-f",
                "Front=Second question",
                "-f",
                "Back=Second answer",
                "--tags",
                "test,cli",
            ]
        )
        assert data["results"][0]["status"] == "created"

    def test_05_list_notes(self, runner):
        data = runner.json(["note", "list", "--deck", "CLITest"])
        assert data["total"] == 2
        assert len(data["notes"]) == 2

    def test_06_show_note(self, runner):
        data = runner.json(["note", "list", "--deck", "CLITest"])
        note_id = str(data["notes"][0]["id"])

        result = runner.invoke(["note", "show", note_id])
        assert result.exit_code == 0
        assert note_id in result.output

    def test_07_show_note_json(self, runner):
        data = runner.json(["note", "list", "--deck", "CLITest"])
        note_id = str(data["notes"][0]["id"])

        note_data = runner.json(["note", "show", note_id])
        assert "notes" in note_data
        assert len(note_data["notes"]) == 1
        assert note_data["notes"][0]["id"] == data["notes"][0]["id"]

    def test_08_update_note(self, runner):
        data = runner.json(["note", "list", "--deck", "CLITest"])
        note_id = str(data["notes"][0]["id"])

        result = runner.invoke(
            [
                "note",
                "update",
                note_id,
                "-f",
                "Back=Updated answer",
            ]
        )
        assert result.exit_code == 0
        assert "Updated" in result.output

    def test_09_update_verified(self, runner):
        data = runner.json(["note", "list", "--deck", "CLITest"])
        note_id = str(data["notes"][0]["id"])

        note_data = runner.json(["note", "show", note_id])
        note = note_data["notes"][0]
        assert note["content"]["Back"] == "Updated answer"

    def test_10_list_by_tags(self, runner):
        data = runner.json(["note", "list", "--tags", "test"])
        assert data["total"] >= 1

    def test_11_delete_note(self, runner):
        data = runner.json(["note", "list", "--deck", "CLITest"])
        note_id = str(data["notes"][0]["id"])

        result = runner.invoke(["note", "delete", note_id, "--yes"])
        assert result.exit_code == 0
        assert "Deleted" in result.output

    def test_12_delete_note_json(self, runner):
        data = runner.json(["note", "list", "--deck", "CLITest"])
        assert data["total"] >= 1
        note_id = str(data["notes"][0]["id"])

        del_data = runner.json(["note", "delete", note_id, "--yes"])
        assert len(del_data["deleted"]) >= 1

    def test_13_show_nonexistent(self, runner):
        result = runner.invoke(["note", "show", "999999999"])
        assert result.exit_code != 0

    def test_14_create_bulk_stdin(self, runner):
        notes = json.dumps(
            [
                {
                    "deck": "CLITest",
                    "note_type": "Basic",
                    "fields": {"Front": f"Bulk Q{i}", "Back": f"Bulk A{i}"},
                }
                for i in range(3)
            ]
        )
        result = runner.invoke(["note", "create", "--json-input"], input=notes)
        assert result.exit_code == 0


@pytest.mark.integration
class TestNoteSearch:
    def test_search_stub(self, runner):
        result = runner.invoke(["note", "search", "test query"])
        assert result.exit_code != 0 or "not available" in result.output.lower()


@pytest.mark.integration
class TestTypeLifecycle:
    def test_01_list_types(self, runner):
        result = runner.invoke(["type", "list"])
        assert result.exit_code == 0
        assert "Basic" in result.output

    def test_02_list_types_json(self, runner):
        data = runner.json(["type", "list"])
        assert any(nt["name"] == "Basic" for nt in data)

    def test_03_show_type(self, runner):
        result = runner.invoke(["type", "show", "Basic"])
        assert result.exit_code == 0
        assert "Front" in result.output
        assert "Back" in result.output

    def test_04_show_type_json(self, runner):
        data = runner.json(["type", "show", "Basic"])
        assert data["name"] == "Basic"
        assert "fields" in data
        assert "templates" in data

    def test_05_create_type(self, runner):
        result = runner.invoke(
            [
                "type",
                "create",
                "--name",
                "CLITestType",
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

    def test_06_create_type_json(self, runner):
        data = runner.json(
            [
                "type",
                "create",
                "--name",
                "CLITestType2",
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

    def test_07_show_created_type(self, runner):
        data = runner.json(["type", "show", "CLITestType"])
        assert data["name"] == "CLITestType"
        assert "Question" in data["fields"]
        assert "Answer" in data["fields"]

    def test_08_update_type(self, runner):
        data = runner.json(["type", "show", "CLITestType"])
        type_id = str(data["id"])

        result = runner.invoke(
            [
                "type",
                "update",
                type_id,
                "--css",
                ".card { color: red; }",
            ]
        )
        assert result.exit_code == 0
        assert "Updated" in result.output

    def test_09_update_type_json(self, runner):
        data = runner.json(["type", "show", "CLITestType"])
        type_id = str(data["id"])

        upd = runner.json(["type", "update", type_id, "--name", "CLITestTypeRenamed"])
        assert upd["results"][0]["status"] == "updated"


@pytest.mark.integration
class TestOutputModes:
    """Test --json, --pretty, --no-pretty across commands."""

    def test_json_flag(self, runner):
        result = runner.invoke(["--json", "info"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "note_types" in data

    def test_no_pretty(self, runner):
        result = runner.invoke(["--no-pretty", "info"])
        assert result.exit_code == 0

    def test_json_pretty_conflict(self, runner):
        result = runner.invoke(["info", "--json", "--pretty"])
        assert result.exit_code != 0
