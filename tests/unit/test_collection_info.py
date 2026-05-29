from __future__ import annotations


class TestCollectionInfo:
    async def test_returns_summary_by_default(self, wrapper):
        info = await wrapper.get_collection_info()
        assert "summary" in info
        summary = info["summary"]
        assert "path" in summary
        assert "notes" in summary
        assert "cards" in summary
        assert "decks" in summary
        assert "note_types" in summary

    async def test_include_filters_sections(self, wrapper):
        info = await wrapper.get_collection_info(include=["decks"])
        assert "decks" in info
        assert "note_types" not in info
        assert "tags" not in info
        assert "stats" not in info

    async def test_default_note_types_present(self, wrapper):
        info = await wrapper.get_collection_info(include=["note_types"])
        names = {nt["name"] for nt in info["note_types"]}
        assert "Basic" in names
        assert "Cloze" in names

    async def test_note_type_fields(self, wrapper):
        info = await wrapper.get_collection_info(include=["note_types"])
        basic = next(nt for nt in info["note_types"] if nt["name"] == "Basic")
        assert basic["fields"] == ["Front", "Back"]
        assert basic["type"] == "standard"

    async def test_cloze_type(self, wrapper):
        info = await wrapper.get_collection_info(include=["note_types"])
        cloze = next(nt for nt in info["note_types"] if nt["name"] == "Cloze")
        assert cloze["type"] == "cloze"

    async def test_note_type_details_includes_templates(self, wrapper):
        info = await wrapper.get_collection_info(
            include=["note_types"], note_type_details=["Basic"]
        )
        basic = next(nt for nt in info["note_types"] if nt["name"] == "Basic")
        assert "css" in basic["detail"]
        assert len(basic["detail"]["templates"]) >= 1
        tmpl = basic["detail"]["templates"][0]
        assert "front" in tmpl
        assert "back" in tmpl
        assert "name" in tmpl

    async def test_note_type_details_omitted_by_default(self, wrapper):
        info = await wrapper.get_collection_info(include=["note_types"])
        basic = next(nt for nt in info["note_types"] if nt["name"] == "Basic")
        assert "detail" not in basic

    async def test_default_deck_present(self, wrapper):
        info = await wrapper.get_collection_info(include=["decks"])
        names = {d["name"] for d in info["decks"]}
        assert "Default" in names

    async def test_deck_note_count(self, wrapper, basic_note):
        info = await wrapper.get_collection_info(include=["decks"])
        test_deck = next(d for d in info["decks"] if d["name"] == "Test")
        assert test_deck["note_count"] == 1

    async def test_tags_empty_initially(self, wrapper):
        info = await wrapper.get_collection_info(include=["tags"])
        assert info["tags"] == []

    async def test_stats_empty_collection(self, wrapper):
        info = await wrapper.get_collection_info(include=["stats"])
        stats = info["stats"]
        assert stats["total_notes"] == 0
        assert stats["total_cards"] == 0

    async def test_stats_after_adding_note(self, wrapper, basic_note):
        info = await wrapper.get_collection_info(include=["stats"])
        stats = info["stats"]
        assert stats["total_notes"] == 1
        assert stats["total_cards"] >= 1

    async def test_nested_decks_not_double_counted(self, wrapper):
        """Regression: top-level deck_due_tree counts already roll up their
        subdecks. Summing recursively over every node double-counts nested
        decks, so the rolled-up totals must sum over top-level decks only."""
        notes = [
            {"deck": "Parent", "note_type": "Basic", "fields": {"Front": f"P{i}", "Back": "x"}}
            for i in range(2)
        ] + [
            {
                "deck": "Parent::Child",
                "note_type": "Basic",
                "fields": {"Front": f"C{i}", "Back": "x"},
            }
            for i in range(3)
        ]
        await wrapper.upsert_notes(notes)

        stats = (await wrapper.get_collection_info(include=["stats"]))["stats"]
        # 5 brand-new cards total; the nested deck must not inflate new_cards.
        assert stats["new_cards"] == 5
        assert stats["total_notes"] == 5
        # Per-deck summary reports each node's own (hierarchical) note count.
        assert stats["decks_summary"]["Parent::Child"]["notes"] == 3
        assert stats["decks_summary"]["Parent"]["notes"] == 5

        # The summary's due_today rolls up the same way (0 here: new cards
        # aren't yet in review/learn), and must not be inflated by nesting.
        summary = (await wrapper.get_collection_info(include=["summary"]))["summary"]
        assert summary["due_today"] == 0
