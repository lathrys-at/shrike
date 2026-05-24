from __future__ import annotations


class TestCollectionInfo:
    def test_returns_all_sections_by_default(self, wrapper):
        info = wrapper.get_collection_info()
        assert "note_types" in info
        assert "decks" in info
        assert "tags" in info
        assert "stats" in info

    def test_include_filters_sections(self, wrapper):
        info = wrapper.get_collection_info(include=["decks"])
        assert "decks" in info
        assert "note_types" not in info
        assert "tags" not in info
        assert "stats" not in info

    def test_default_note_types_present(self, wrapper):
        info = wrapper.get_collection_info(include=["note_types"])
        names = {nt["name"] for nt in info["note_types"]}
        assert "Basic" in names
        assert "Cloze" in names

    def test_note_type_fields(self, wrapper):
        info = wrapper.get_collection_info(include=["note_types"])
        basic = next(nt for nt in info["note_types"] if nt["name"] == "Basic")
        assert basic["fields"] == ["Front", "Back"]
        assert basic["type"] == "standard"

    def test_cloze_type(self, wrapper):
        info = wrapper.get_collection_info(include=["note_types"])
        cloze = next(nt for nt in info["note_types"] if nt["name"] == "Cloze")
        assert cloze["type"] == "cloze"

    def test_note_type_details_includes_templates(self, wrapper):
        info = wrapper.get_collection_info(
            include=["note_types"], note_type_details=["Basic"]
        )
        basic = next(nt for nt in info["note_types"] if nt["name"] == "Basic")
        assert "templates" in basic
        assert "css" in basic
        assert len(basic["templates"]) >= 1
        tmpl = basic["templates"][0]
        assert "front" in tmpl
        assert "back" in tmpl
        assert "name" in tmpl

    def test_note_type_details_omitted_by_default(self, wrapper):
        info = wrapper.get_collection_info(include=["note_types"])
        basic = next(nt for nt in info["note_types"] if nt["name"] == "Basic")
        assert "templates" not in basic
        assert "css" not in basic

    def test_default_deck_present(self, wrapper):
        info = wrapper.get_collection_info(include=["decks"])
        names = {d["name"] for d in info["decks"]}
        assert "Default" in names

    def test_deck_note_count(self, wrapper, basic_note):
        info = wrapper.get_collection_info(include=["decks"])
        test_deck = next(d for d in info["decks"] if d["name"] == "Test")
        assert test_deck["note_count"] == 1

    def test_tags_empty_initially(self, wrapper):
        info = wrapper.get_collection_info(include=["tags"])
        assert info["tags"] == []

    def test_stats_empty_collection(self, wrapper):
        info = wrapper.get_collection_info(include=["stats"])
        stats = info["stats"]
        assert stats["total_notes"] == 0
        assert stats["total_cards"] == 0

    def test_stats_after_adding_note(self, wrapper, basic_note):
        info = wrapper.get_collection_info(include=["stats"])
        stats = info["stats"]
        assert stats["total_notes"] == 1
        assert stats["total_cards"] >= 1
