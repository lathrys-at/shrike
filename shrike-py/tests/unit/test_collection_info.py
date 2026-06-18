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
        assert info["decks"] is not None
        # Unrequested sections read as None — explicit nulls on the raw wire
        # since the #391 to_wire retirement (.get() is the stable form).
        assert info.get("note_types") is None
        assert info.get("tags") is None
        assert info.get("stats") is None

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
        # No detail unless requested (explicit null on the raw wire, #391).
        assert basic.get("detail") is None

    async def test_default_deck_present(self, wrapper):
        info = await wrapper.get_collection_info(include=["decks"])
        names = {d["name"] for d in info["decks"]}
        assert "Default" in names

    async def test_deck_note_count(self, wrapper, basic_note):
        info = await wrapper.get_collection_info(include=["decks"])
        test_deck = next(d for d in info["decks"] if d["name"] == "Test")
        assert test_deck["note_count"] == 1

    async def test_deck_note_counts_match_find_notes(self, wrapper):
        """The single-pass note-count rollup that replaces the per-deck
        find_notes query must agree with find_notes("deck:NAME") across nested
        decks and cards parked in a filtered deck (the odid path)."""
        notes = (
            [
                {"deck": "Lang", "note_type": "Basic", "fields": {"Front": f"p{i}", "Back": "x"}}
                for i in range(2)
            ]
            + [
                {
                    "deck": "Lang::Japanese",
                    "note_type": "Basic",
                    "fields": {"Front": f"c{i}", "Back": "x"},
                }
                for i in range(3)
            ]
            + [{"deck": "Other", "note_type": "Basic", "fields": {"Front": "o", "Back": "x"}}]
        )
        await wrapper.upsert_notes(notes)

        # Park every card in a filtered deck: cards get did=Filt, odid=original,
        # which Anki's deck: search attributes to *both* decks.
        # Filtered decks are created by Anki desktop, not Shrike: author one
        # with the pip-anki ORACLE in a subprocess on the released file, then
        # let the wrapper re-acquire and observe it.
        import subprocess
        import sys

        wrapper.release_now()
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "from anki.collection import Collection\n"
                    "col = Collection(sys.argv[1])\n"
                    'fid = col.decks.new_filtered("Filt")\n'
                    "col.sched.rebuild_filtered_deck(fid)\n"
                    "col.close()\n"
                ),
                wrapper._path,
            ],
            check=True,
        )
        await wrapper.reopen()

        info = await wrapper.get_collection_info(include=["decks"])
        by_name = {d["name"]: d["note_count"] for d in info["decks"]}

        for name, count in by_name.items():
            expected = await wrapper.run(lambda c, n=name: len(c.find_notes(f'"deck:{n}"')))
            assert count == expected, f"{name}: rollup={count} find_notes={expected}"

        # Sanity: the nested rollup and the filtered deck are both non-trivial.
        assert by_name["Lang"] == 5
        assert by_name["Lang::Japanese"] == 3
        assert by_name["Filt"] == 6

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
