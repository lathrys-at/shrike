from __future__ import annotations

import pytest

from tests.unit._native_shims import (
    NoteTypeOpError,
    find_and_replace_note_types,
    upsert_note_types,
)


def _make_model(wrapper, *, templates, css, fields=("F",)):
    def build(c):
        res = upsert_note_types(
            c, [{"name": "FR", "fields": list(fields), "templates": templates, "css": css}]
        )
        assert res[0]["status"] != "error", res[0].get("error")
        return res[0]["id"]

    return wrapper.run_sync(build)


async def _detail(wrapper):
    info = await wrapper.get_collection_info(include=["note_types"], note_type_details=["FR"])
    nt = next(n for n in info["note_types"] if n["name"] == "FR")
    return nt["detail"]


def _replace(wrapper, **kw):
    return wrapper.run_sync(lambda c: find_and_replace_note_types(c, "FR", **kw))


class TestFindAndReplaceNoteTypes:
    """find_and_replace_note_types rewrites template HTML and CSS in place,
    scoped by the front/back/css selectors, returning a replacement count. No
    note field values are involved."""

    async def test_replace_in_front_and_back(self, wrapper):
        # Old and New are both real fields (Anki rejects a template that
        # references a missing field, at create *and* on save).
        _make_model(
            wrapper,
            fields=("Old", "New"),
            templates=[{"name": "C", "front": "{{Old}}", "back": "see {{Old}} again"}],
            css="",
        )
        res = _replace(wrapper, search="{{Old}}", replacement="{{New}}")
        assert res["replacements"] == 2
        assert res["templates_changed"] == ["C"]
        assert res["css_changed"] is False
        detail = await _detail(wrapper)
        assert detail["templates"][0]["front"] == "{{New}}"
        assert detail["templates"][0]["back"] == "see {{New}} again"

    async def test_replace_in_css(self, wrapper):
        _make_model(
            wrapper,
            templates=[{"name": "C", "front": "{{F}}", "back": "{{F}}"}],
            css=".card { color: red; }",
        )
        res = _replace(wrapper, search="red", replacement="blue", front=False, back=False)
        assert res["replacements"] == 1
        assert res["templates_changed"] == []
        assert res["css_changed"] is True
        detail = await _detail(wrapper)
        assert "color: blue" in detail["css"]

    async def test_selectors_scope_the_search(self, wrapper):
        # "z" appears on front, back, and css; only the front is selected.
        _make_model(
            wrapper,
            templates=[{"name": "C", "front": "z {{F}}", "back": "z"}],
            css=".card { /* z */ }",
        )
        res = _replace(wrapper, search="z", replacement="y", back=False, css=False)
        assert res["replacements"] == 1
        detail = await _detail(wrapper)
        assert detail["templates"][0]["front"] == "y {{F}}"
        assert detail["templates"][0]["back"] == "z"
        assert "/* z */" in detail["css"]

    async def test_match_case_default_is_sensitive(self, wrapper):
        _make_model(
            wrapper,
            templates=[{"name": "C", "front": "Red red RED {{F}}", "back": "{{F}}"}],
            css="",
        )
        res = _replace(wrapper, search="red", replacement="blue")
        assert res["replacements"] == 1  # only the exact-case "red"
        detail = await _detail(wrapper)
        assert detail["templates"][0]["front"] == "Red blue RED {{F}}"

    async def test_match_case_false(self, wrapper):
        _make_model(
            wrapper,
            templates=[{"name": "C", "front": "Red red RED {{F}}", "back": "{{F}}"}],
            css="",
        )
        res = _replace(wrapper, search="red", replacement="x", match_case=False)
        assert res["replacements"] == 3

    async def test_literal_replacement_is_not_group_ref(self, wrapper):
        # A literal replace containing "\1" must be inserted verbatim, not as a
        # regex backreference.
        _make_model(
            wrapper,
            templates=[{"name": "C", "front": "A {{F}}", "back": "{{F}}"}],
            css="",
        )
        _replace(wrapper, search="A", replacement=r"\1B")
        detail = await _detail(wrapper)
        assert detail["templates"][0]["front"] == r"\1B {{F}}"

    async def test_regex_with_capture_group(self, wrapper):
        _make_model(
            wrapper,
            fields=("Old", "F"),
            templates=[{"name": "C", "front": "{{Old}} {{F}}", "back": "{{Old}}"}],
            css="",
        )
        # Match the Old field ref only, leaving {{F}} on the front intact (so the
        # front keeps a field reference and stays valid).
        res = _replace(wrapper, search=r"\{\{(Old)\}\}", replacement=r"[\1]", regex=True)
        assert res["replacements"] == 2
        detail = await _detail(wrapper)
        assert detail["templates"][0]["front"] == "[Old] {{F}}"
        assert detail["templates"][0]["back"] == "[Old]"

    async def test_multiple_templates(self, wrapper):
        _make_model(
            wrapper,
            templates=[
                {"name": "C1", "front": "z {{F}}", "back": "{{F}}"},
                {"name": "C2", "front": "{{F}} two", "back": "z {{F}}"},
                {"name": "C3", "front": "{{F}} three", "back": "{{F}}"},
            ],
            css="",
        )
        res = _replace(wrapper, search="z", replacement="q")
        assert res["replacements"] == 2
        assert res["templates_changed"] == ["C1", "C2"]

    async def test_no_match_makes_no_change(self, wrapper):
        _make_model(
            wrapper,
            templates=[{"name": "C", "front": "{{F}}", "back": "{{F}}"}],
            css=".card {}",
        )
        res = _replace(wrapper, search="absent", replacement="x")
        assert res["replacements"] == 0
        assert res["templates_changed"] == []
        assert res["css_changed"] is False

    async def test_unknown_note_type_raises(self, wrapper):
        with pytest.raises(NoteTypeOpError, match="not found"):
            wrapper.run_sync(
                lambda c: find_and_replace_note_types(c, "Nope", search="a", replacement="b")
            )

    async def test_invalid_regex_raises(self, wrapper):
        _make_model(
            wrapper,
            templates=[{"name": "C", "front": "{{F}}", "back": "{{F}}"}],
            css="",
        )
        with pytest.raises(NoteTypeOpError, match="invalid regex"):
            _replace(wrapper, search="(unclosed", replacement="x", regex=True)
