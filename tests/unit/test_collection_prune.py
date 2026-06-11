"""Collection-layer tests for prune (#89): unused tags, empty notes, empty cards.

Exercises CollectionWrapper._prune / _find_empty_notes and the media-safe blank
rule (embed_text.field_is_blank) directly, no server.
"""

from __future__ import annotations

import json

import pytest

from tests.oracles.embed_text_oracle import field_is_blank
from tests.unit.conftest import make_notes


def _add_note(wrapper, fields, *, tags=None, model="Basic", deck="D"):
    results = make_notes(
        wrapper,
        [{"note_type": model, "deck": deck, "fields": dict(fields), "tags": list(tags or [])}],
    )
    assert results[0]["status"] == "created", results[0]
    return results[0]["id"]


def _blank_note(wrapper, *, model="Basic"):
    """A note that started valid then had every field cleared (an empty note)."""
    nid = _add_note(wrapper, {"Front": "tmp", "Back": "x"} if model == "Basic" else {"Text": "t"})
    wrapper.run_sync(lambda c: _clear(c, nid))
    return nid


def _clear(c, nid):
    _, _, fields, _ = c.get_note(nid)
    c.update_note(nid, ["" for _ in fields])


def _prune(wrapper, **kw):
    kw.setdefault("unused_tags", False)
    kw.setdefault("empty_notes", False)
    kw.setdefault("empty_cards", False)
    kw.setdefault("unused_media", False)
    kw.setdefault("dry_run", True)

    def run(c):
        result = json.loads(
            c.prune(
                kw["unused_tags"],
                kw["empty_notes"],
                kw["empty_cards"],
                kw["unused_media"],
                kw["dry_run"],
            )
        )
        removed = result.pop("removed_note_ids")
        return result, removed

    return wrapper.run_sync(run)


def _find_empty(wrapper):
    """Empty-note ids via a dry-run prune (the native core owns the scan)."""
    result, _ = _prune(wrapper, empty_notes=True, dry_run=True)
    return result["empty_notes"]["removed"]


def _note_exists(wrapper, nid):
    return bool(wrapper.run_sync(lambda c: c.find_notes(f"nid:{nid}")))


class TestFieldIsBlank:
    @pytest.mark.parametrize("value", ["", "   ", "<br>", "<div></div>", "&nbsp;", "\xa0"])
    def test_blank(self, value):
        assert field_is_blank(value) is True

    @pytest.mark.parametrize(
        "value", ["hello", "<b>x</b>", "<img src='a.png'>", "[sound:a.mp3]", "<audio src=x>"]
    )
    def test_not_blank(self, value):
        assert field_is_blank(value) is False


class TestEmptyNotes:
    def test_finds_only_blank_notes(self, wrapper):
        blank = _blank_note(wrapper)
        full = _add_note(wrapper, {"Front": "Q", "Back": "A"})
        found = _find_empty(wrapper)
        assert blank in found
        assert full not in found

    def test_media_only_note_is_kept(self, wrapper):
        # Front blank but Back has an image -> note has content, not empty.
        nid = _add_note(wrapper, {"Front": "x", "Back": "y"})
        wrapper.run_sync(lambda c: _set(c, nid, {"Front": "", "Back": "<img src='pic.png'>"}))
        found = _find_empty(wrapper)
        assert nid not in found

    def test_content_in_any_field_keeps_note(self, wrapper):
        nid = _add_note(wrapper, {"Front": "x", "Back": "y"})
        wrapper.run_sync(lambda c: _set(c, nid, {"Front": "", "Back": "still here"}))
        assert _find_empty(wrapper) == []

    def test_dry_run_reports_but_keeps(self, wrapper):
        blank = _blank_note(wrapper)
        result, removed = _prune(wrapper, empty_notes=True, dry_run=True)
        assert result["empty_notes"]["removed"] == [blank]
        assert removed == [blank]
        assert _note_exists(wrapper, blank)  # nothing deleted

    def test_apply_removes(self, wrapper):
        blank = _blank_note(wrapper)
        result, removed = _prune(wrapper, empty_notes=True, dry_run=False)
        assert result["empty_notes"]["removed"] == [blank]
        assert not _note_exists(wrapper, blank)


class TestUnusedTags:
    def test_enumerates_and_clears(self, wrapper):
        _add_note(wrapper, {"Front": "Q", "Back": "A"}, tags=["used", "willremove"])
        # Orphan "willremove" in the registry by removing it from the note.
        nid = wrapper.run_sync(lambda c: c.find_notes("tag:willremove")[0])
        wrapper.run_sync(lambda c: c.update_note_tags([nid], remove=["willremove"]))

        preview, _ = _prune(wrapper, unused_tags=True, dry_run=True)
        assert preview["unused_tags"]["tags"] == ["willremove"]
        assert "willremove" in wrapper.run_sync(
            lambda c: json.loads(c.collection_info(["tags"], []))["tags"]
        )  # not cleared yet

        applied, _ = _prune(wrapper, unused_tags=True, dry_run=False)
        assert applied["unused_tags"]["removed"] == 1
        assert "willremove" not in wrapper.run_sync(
            lambda c: json.loads(c.collection_info(["tags"], []))["tags"]
        )
        assert "used" in wrapper.run_sync(
            lambda c: json.loads(c.collection_info(["tags"], []))["tags"]
        )

    def test_parent_tag_with_child_notes_is_kept(self, wrapper):
        # A note tagged only "parent::child" keeps "parent" as in-use (hierarchy).
        _add_note(wrapper, {"Front": "Q", "Back": "A"}, tags=["parent::child"])
        preview, _ = _prune(wrapper, unused_tags=True, dry_run=True)
        assert "parent" not in preview["unused_tags"]["tags"]
        assert "parent::child" not in preview["unused_tags"]["tags"]


class TestEmptyCards:
    def test_finds_and_removes_empty_cloze_card(self, wrapper):
        nid = _add_note(wrapper, {"Text": "{{c1::A}} and {{c2::B}}"}, model="Cloze")
        assert len(wrapper.run_sync(lambda c: c.cards_of_note(nid))) == 2
        # Drop c2 -> its card becomes empty.
        wrapper.run_sync(lambda c: _set(c, nid, {"Text": "{{c1::A}} only"}))

        preview, _ = _prune(wrapper, empty_cards=True, dry_run=True)
        assert preview["empty_cards"]["cards_removed"] == 1
        assert len(wrapper.run_sync(lambda c: c.cards_of_note(nid))) == 2  # untouched

        applied, _ = _prune(wrapper, empty_cards=True, dry_run=False)
        assert applied["empty_cards"]["cards_removed"] == 1
        assert len(wrapper.run_sync(lambda c: c.cards_of_note(nid))) == 1


class TestPruneOrdering:
    def test_tag_freed_by_empty_note_is_cleared_same_call(self, wrapper):
        # A tag that lives only on an empty note: removing the note frees the tag,
        # and unused-tags (run last on apply) clears it in the same call.
        blank = _blank_note(wrapper)
        wrapper.run_sync(lambda c: c.update_note_tags([blank], add=["lonely"]))
        result, removed = _prune(wrapper, empty_notes=True, unused_tags=True, dry_run=False)
        assert blank in removed
        assert not _note_exists(wrapper, blank)
        assert "lonely" not in wrapper.run_sync(
            lambda c: json.loads(c.collection_info(["tags"], []))["tags"]
        )


def _set(c, nid, fields):
    result = json.loads(
        c.upsert_notes(json.dumps([{"id": nid, "fields": fields}]), "allow", False)
    )[0]
    assert result["status"] == "updated", result
