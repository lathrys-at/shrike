"""Unit coverage for CLI type/info output branches (#107)."""

from __future__ import annotations

import json
from unittest.mock import patch

from click.testing import CliRunner

from shrike.cli import cli
from shrike.cli.type_cmd import _resolve_note_type
from shrike.client import ShrikeClient
from shrike.schemas import CollectionInfo, NoteTypeDetail, NoteTypeInfo, Summary, TemplateInfo


def _runner(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("")
    return CliRunner(), ["--config", str(cfg)]


def _note_types() -> list[NoteTypeInfo]:
    return [
        NoteTypeInfo(name="123", id=10, fields=["Front"]),
        NoteTypeInfo(name="Basic", id=123, fields=["Front", "Back"]),
    ]


def test_resolve_note_type_bare_numeric_prefers_id_over_name():
    client = ShrikeClient("http://127.0.0.1:8372/mcp", autostart=False)
    with patch.object(
        ShrikeClient,
        "collection_info",
        return_value=CollectionInfo(note_types=_note_types()),
    ):
        assert _resolve_note_type(client, "123") == (123, "Basic")


def test_type_list_identifier_json_requests_details(tmp_path):
    runner, base_args = _runner(tmp_path)
    detail = NoteTypeDetail(
        templates=[TemplateInfo(name="Card 1", front="{{Front}}", back="{{Back}}")],
        css=".card { color: red; }",
    )

    with patch.object(
        ShrikeClient,
        "collection_info",
        side_effect=[
            CollectionInfo(note_types=_note_types()),
            CollectionInfo(note_types=[NoteTypeInfo(name="Basic", id=123, detail=detail)]),
        ],
    ) as collection_info:
        result = runner.invoke(cli, [*base_args, "type", "list", "123", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["name"] == "Basic"
    collection_info.assert_any_call(include=["note_types"], note_type_details=["Basic"])


def test_info_json_includes_detail_sections(tmp_path):
    runner, base_args = _runner(tmp_path)
    summary = Summary(
        path="/collection.anki2",
        created="2026-01-01",
        modified="2026-01-02",
        notes=1,
        cards=2,
        decks=1,
        note_types=1,
        tags=3,
        due_today=0,
    )

    with patch.object(
        ShrikeClient,
        "collection_info",
        return_value=CollectionInfo(summary=summary, note_types=_note_types(), tags=["marked"]),
    ) as collection_info:
        result = runner.invoke(cli, [*base_args, "info", "--types", "--tags", "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["note_types"][0]["name"] == "123"
    assert data["tags"] == ["marked"]
    collection_info.assert_called_once_with(
        include=["summary", "note_types", "tags"],
        note_type_details=None,
    )
