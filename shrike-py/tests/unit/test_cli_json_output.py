"""CLI ``--json`` output carries no explicit nulls.

Since the kernel read wire serializes unset Options as explicit ``null``,
the "MCP wire carries nulls, CLI --json doesn't" split hinges entirely on the
``exclude_none=True`` in ``output._to_jsonable``. These tests pin that invariant
at the command layer: a response model with unset optional sections must render
with those keys absent, not ``null``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from shrike.cli import cli
from shrike.cli import client as cli_client
from shrike.schemas import CollectionInfo, ListNotesResponse, Note, Summary


@pytest.fixture
def fake() -> MagicMock:
    return MagicMock(spec=cli_client.ShrikeClient)


@pytest.fixture
def run(tmp_path, fake):
    """Invoke the CLI with the client patched to `fake` and an empty config."""
    cfg = tmp_path / "config.yml"
    cfg.write_text("")
    runner = CliRunner()

    def _run(*args: str, **kwargs):
        with patch("shrike.client.ShrikeClient", return_value=fake):
            return runner.invoke(
                cli,
                ["--config", str(cfg), *args],
                catch_exceptions=False,
                **kwargs,
            )

    return _run


def _assert_no_nulls(data: Any, path: str = "$") -> None:
    """Recursively assert no value anywhere in the JSON tree is null."""
    assert data is not None, f"explicit null at {path}"
    if isinstance(data, dict):
        for key, value in data.items():
            _assert_no_nulls(value, f"{path}.{key}")
    elif isinstance(data, list):
        for i, value in enumerate(data):
            _assert_no_nulls(value, f"{path}[{i}]")


def _summary() -> Summary:
    return Summary(
        path="/collection.anki2",
        created="2026-01-01",
        modified="2026-01-02",
        notes=1,
        cards=2,
        decks=1,
        note_types=1,
        tags=0,
        due_today=0,
    )


class TestJsonNullStripping:
    def test_info_omits_unrequested_sections(self, run, fake):
        # Only summary requested: the other CollectionInfo sections stay None
        # and must not appear as explicit nulls in --json output.
        fake.collection_info.return_value = CollectionInfo(summary=_summary())

        result = run("--json", "collection", "info")

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["summary"]["path"] == "/collection.anki2"
        for section in ("note_types", "decks", "tags", "stats"):
            assert section not in data
        _assert_no_nulls(data)

    def test_note_list_brief_omits_content(self, run, fake):
        # Meta mode: Note.content is None and must be absent, not null.
        fake.list_notes.return_value = ListNotesResponse(
            notes=[
                Note(
                    id=1,
                    note_type="Basic",
                    deck="Default",
                    tags=["t"],
                    modified="2026-01-02T00:00:00",
                )
            ],
            total=1,
            limit=50,
        )

        result = run("note", "list", "--deck", "Default", "--brief", "--json")

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["notes"][0]["id"] == 1
        assert "content" not in data["notes"][0]
        _assert_no_nulls(data)
