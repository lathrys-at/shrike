"""Integration tests for the note-type lifecycle MCP tools over HTTP transport."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestNoteTypeLifecycle:
    """Create, inspect, use, and update custom note types."""

    def test_create_and_inspect(self, mcp):
        result = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Custom",
                        "fields": ["Term", "Definition", "Example"],
                        "templates": [
                            {
                                "name": "Forward",
                                "front": "<div>{{Term}}</div>",
                                "back": "{{FrontSide}}<hr>{{Definition}}<br>{{Example}}",
                            }
                        ],
                        "css": ".card { font-family: sans-serif; }",
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "created"

        info = mcp(
            "collection_info",
            {"include": ["note_types"], "note_type_details": ["Custom"]},
        )
        nt = next(nt for nt in info["note_types"] if nt["name"] == "Custom")
        assert nt["fields"] == ["Term", "Definition", "Example"]
        assert len(nt["detail"]["templates"]) == 1
        assert "font-family" in nt["detail"]["css"]

    def test_create_note_with_custom_type(self, mcp):
        mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Vocab",
                        "fields": ["Word", "Meaning"],
                        "templates": [
                            {"name": "Card 1", "front": "{{Word}}", "back": "{{Meaning}}"}
                        ],
                        "css": ".card { font-size: 16px; }",
                    }
                ]
            },
        )
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Vocab",
                        "fields": {"Word": "shrike", "Meaning": "A predatory songbird"},
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "created"

    def test_update_css(self, mcp):
        created = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Styled",
                        "fields": ["Q", "A"],
                        "templates": [{"name": "Card 1", "front": "{{Q}}", "back": "{{A}}"}],
                        "css": ".card { color: black; }",
                    }
                ]
            },
        )
        nt_id = created["results"][0]["id"]

        mcp(
            "upsert_note_types",
            {"note_types": [{"id": nt_id, "css": ".card { color: red; }"}]},
        )
        info = mcp(
            "collection_info",
            {"include": ["note_types"], "note_type_details": ["Styled"]},
        )
        nt = next(nt for nt in info["note_types"] if nt["name"] == "Styled")
        assert "color: red" in nt["detail"]["css"]

    def test_update_name(self, mcp):
        created = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "OldName",
                        "fields": ["F"],
                        "templates": [{"name": "Card 1", "front": "{{F}}", "back": "{{F}}"}],
                        "css": "",
                    }
                ]
            },
        )
        nt_id = created["results"][0]["id"]

        result = mcp("upsert_note_types", {"note_types": [{"id": nt_id, "name": "NewName"}]})
        assert result["results"][0]["status"] == "updated"

        info = mcp("collection_info", {"include": ["note_types"]})
        names = {nt["name"] for nt in info["note_types"]}
        assert "NewName" in names
        assert "OldName" not in names

    def test_field_update_preserves_note_data(self, mcp):
        # Regression: updating a note type's fields must not blank every note's
        # content or delete its cards. Rename a field and confirm the existing
        # note keeps its data under the new field name.
        created = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Preserve",
                        "fields": ["Front", "Back"],
                        "templates": [{"name": "C", "front": "{{Front}}", "back": "{{Back}}"}],
                        "css": "",
                    }
                ]
            },
        )
        nt_id = created["results"][0]["id"]
        note = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Preserve",
                        "note_type": "Preserve",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        note_id = note["results"][0]["id"]

        mcp("upsert_note_types", {"note_types": [{"id": nt_id, "fields": ["Frente", "Back"]}]})

        listed = mcp("list_notes", {"ids": [note_id]})["notes"][0]
        assert listed["content"] == {"Frente": "Q", "Back": "A"}

    def test_field_ops_move_rename_remove(self, mcp):
        # update_note_type_fields: identity-based ops preserve data.
        mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Ops",
                        "fields": ["A", "B", "C"],
                        "templates": [{"name": "C1", "front": "{{A}}", "back": "{{B}}{{C}}"}],
                        "css": "",
                    }
                ]
            },
        )
        note = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Ops",
                        "note_type": "Ops",
                        "fields": {"A": "va", "B": "vb", "C": "vc"},
                    }
                ]
            },
        )
        note_id = note["results"][0]["id"]

        result = mcp(
            "update_note_type_fields",
            {
                "note_type": "Ops",
                "operations": [
                    {"op": "reposition", "name": "C", "position": 0},
                    {"op": "rename", "name": "A", "new_name": "Alpha"},
                    {"op": "remove", "name": "B"},
                    {"op": "add", "name": "D"},
                ],
            },
        )
        assert result["fields"] == ["C", "Alpha", "D"]
        listed = mcp("list_notes", {"ids": [note_id]})["notes"][0]
        assert listed["content"] == {"C": "vc", "Alpha": "va", "D": ""}

    def test_field_ops_invalid_is_atomic(self, mcp):
        mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Atomic",
                        "fields": ["X", "Y"],
                        "templates": [{"name": "C1", "front": "{{X}}", "back": "{{Y}}"}],
                        "css": "",
                    }
                ]
            },
        )
        with pytest.raises(RuntimeError, match="not found"):
            mcp(
                "update_note_type_fields",
                {
                    "note_type": "Atomic",
                    "operations": [
                        {"op": "rename", "name": "X", "new_name": "Xx"},
                        {"op": "remove", "name": "Nope"},
                    ],
                },
            )
        info = mcp("collection_info", {"include": ["note_types"]})
        atomic = next(nt for nt in info["note_types"] if nt["name"] == "Atomic")
        assert atomic["fields"] == ["X", "Y"]  # unchanged

    def test_upsert_field_reorder_rejected(self, mcp):
        # The position-keyed upsert refuses a move (would mislabel data) and
        # redirects to update_note_type_fields.
        created = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "NoReorder",
                        "fields": ["A", "B"],
                        "templates": [{"name": "C1", "front": "{{A}}", "back": "{{B}}"}],
                        "css": "",
                    }
                ]
            },
        )
        nt_id = created["results"][0]["id"]
        result = mcp("upsert_note_types", {"note_types": [{"id": nt_id, "fields": ["B", "A"]}]})
        assert result["results"][0]["status"] == "error"
        assert "update_note_type_fields" in result["results"][0]["error"]

    def test_template_ops_move_rename_remove(self, mcp):
        # update_note_type_templates: identity-based ops preserve cards.
        mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Tmpl",
                        "fields": ["F"],
                        "templates": [
                            {"name": "Ta", "front": "Ta {{F}}", "back": "{{F}}"},
                            {"name": "Tb", "front": "Tb {{F}}", "back": "{{F}}"},
                            {"name": "Tc", "front": "Tc {{F}}", "back": "{{F}}"},
                        ],
                        "css": "",
                    }
                ]
            },
        )
        note = mcp(
            "upsert_notes",
            {"notes": [{"deck": "Tmpl", "note_type": "Tmpl", "fields": {"F": "x"}}]},
        )
        note_id = note["results"][0]["id"]
        assert mcp("list_notes", {"ids": [note_id]})["total"] == 1

        result = mcp(
            "update_note_type_templates",
            {
                "note_type": "Tmpl",
                "operations": [
                    {"op": "reposition", "name": "Tc", "position": 0},
                    {"op": "rename", "name": "Ta", "new_name": "Alpha"},
                    {"op": "remove", "name": "Tb"},
                    {"op": "add", "name": "Td", "front": "Td {{F}}", "back": "{{F}}"},
                ],
            },
        )
        assert result["templates"] == ["Tc", "Alpha", "Td"]
        # note still has one card per surviving/added template (3)
        details = mcp("collection_info", {"include": ["note_types"], "note_type_details": ["Tmpl"]})
        tmpl = next(nt for nt in details["note_types"] if nt["name"] == "Tmpl")
        assert [t["name"] for t in tmpl["detail"]["templates"]] == ["Tc", "Alpha", "Td"]

    def test_upsert_template_reorder_rejected(self, mcp):
        created = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "NoTmplReorder",
                        "fields": ["F"],
                        "templates": [
                            {"name": "Ta", "front": "Ta {{F}}", "back": "{{F}}"},
                            {"name": "Tb", "front": "Tb {{F}}", "back": "{{F}}"},
                        ],
                        "css": "",
                    }
                ]
            },
        )
        nt_id = created["results"][0]["id"]
        result = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "id": nt_id,
                        "templates": [
                            {"name": "Tb", "front": "Tb {{F}}", "back": "{{F}}"},
                            {"name": "Ta", "front": "Ta {{F}}", "back": "{{F}}"},
                        ],
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "error"
        assert "update_note_type_templates" in result["results"][0]["error"]

    def test_duplicate_name_rejected(self, mcp):
        result = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Basic",
                        "fields": ["A"],
                        "templates": [{"name": "C", "front": "{{A}}", "back": "{{A}}"}],
                        "css": "",
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "error"

    def test_find_replace_note_types(self, mcp):
        # findAndReplaceInModels: literal find/replace across a model's template
        # HTML and CSS, scoped by front/back/css, returning a count.
        mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "FRModel",
                        "fields": ["Old", "New"],
                        "templates": [{"name": "C", "front": "{{Old}}", "back": "ans {{Old}}"}],
                        "css": ".card { color: red; }",
                    }
                ]
            },
        )
        result = mcp(
            "find_replace_note_types",
            {"note_type": "FRModel", "search": "{{Old}}", "replace": "{{New}}"},
        )
        assert result["replacements"] == 2
        assert result["templates_changed"] == ["C"]
        assert result["css_changed"] is False

        # A CSS-only replace, with the templates excluded.
        css = mcp(
            "find_replace_note_types",
            {
                "note_type": "FRModel",
                "search": "red",
                "replace": "blue",
                "front": False,
                "back": False,
            },
        )
        assert css["replacements"] == 1
        assert css["css_changed"] is True

        details = mcp(
            "collection_info", {"include": ["note_types"], "note_type_details": ["FRModel"]}
        )
        nt = next(n for n in details["note_types"] if n["name"] == "FRModel")
        assert nt["detail"]["templates"][0]["front"] == "{{New}}"
        assert "color: blue" in nt["detail"]["css"]

    def test_find_replace_note_types_unknown(self, mcp):
        with pytest.raises(RuntimeError, match="not found"):
            mcp(
                "find_replace_note_types",
                {"note_type": "DoesNotExist", "search": "a", "replace": "b"},
            )

    def test_update_note_type_field_metadata(self, mcp):
        mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "FM",
                        "fields": ["F"],
                        "templates": [{"name": "C", "front": "{{F}}", "back": "{{F}}"}],
                        "css": "",
                    }
                ]
            },
        )
        result = mcp(
            "update_note_type_field_metadata",
            {"note_type": "FM", "fields": [{"name": "F", "size": 28, "description": "the prompt"}]},
        )
        assert result["fields_updated"] == ["F"]

        details = mcp("collection_info", {"include": ["note_types"], "note_type_details": ["FM"]})
        nt = next(n for n in details["note_types"] if n["name"] == "FM")
        meta = {f["name"]: f for f in nt["detail"]["fields"]}
        assert meta["F"]["size"] == 28
        assert meta["F"]["description"] == "the prompt"
