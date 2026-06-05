from __future__ import annotations


class TestCreateNotes:
    async def test_create_basic(self, wrapper):
        results = await wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Q", "Back": "A"},
                }
            ]
        )
        assert len(results) == 1
        assert results[0]["status"] == "created"
        assert isinstance(results[0]["id"], int)

    async def test_create_with_tags(self, wrapper):
        results = await wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Q", "Back": "A"},
                    "tags": ["tag1", "tag2"],
                }
            ]
        )
        nid = results[0]["id"]
        note = (await wrapper.list_notes(ids=[nid]))["notes"][0]
        assert set(note["tags"]) == {"tag1", "tag2"}

    async def test_create_deck_auto_created(self, wrapper):
        results = await wrapper.upsert_notes(
            [
                {
                    "deck": "New::Nested::Deck",
                    "note_type": "Basic",
                    "fields": {"Front": "Q", "Back": "A"},
                }
            ]
        )
        assert results[0]["status"] == "created"
        note = (await wrapper.list_notes(ids=[results[0]["id"]]))["notes"][0]
        assert note["deck"] == "New::Nested::Deck"

    async def test_create_missing_note_type(self, wrapper):
        results = await wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Nonexistent",
                    "fields": {"Front": "Q", "Back": "A"},
                }
            ]
        )
        assert results[0]["status"] == "error"
        assert "not found" in results[0]["error"].lower()

    async def test_create_missing_deck(self, wrapper):
        results = await wrapper.upsert_notes(
            [
                {
                    "note_type": "Basic",
                    "fields": {"Front": "Q", "Back": "A"},
                }
            ]
        )
        assert results[0]["status"] == "error"

    async def test_create_missing_fields(self, wrapper):
        results = await wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                }
            ]
        )
        assert results[0]["status"] == "error"

    async def test_create_invalid_field_name(self, wrapper):
        results = await wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Q", "Wrong": "A"},
                }
            ]
        )
        assert results[0]["status"] == "error"
        assert "Wrong" in results[0]["error"]

    async def test_create_bulk(self, wrapper):
        notes = [
            {
                "deck": "Test",
                "note_type": "Basic",
                "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
            }
            for i in range(10)
        ]
        results = await wrapper.upsert_notes(notes)
        assert len(results) == 10
        assert all(r["status"] == "created" for r in results)

    async def test_partial_failure(self, wrapper):
        """One bad note in a batch should not block the others."""
        results = await wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Q1", "Back": "A1"},
                },
                {
                    "deck": "Test",
                    "note_type": "Nonexistent",
                    "fields": {"Front": "Q2", "Back": "A2"},
                },
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Q3", "Back": "A3"},
                },
            ]
        )
        assert results[0]["status"] == "created"
        assert results[1]["status"] == "error"
        assert results[2]["status"] == "created"


class TestUpdateNotes:
    async def test_update_fields(self, wrapper, basic_note):
        results = await wrapper.upsert_notes([{"id": basic_note, "fields": {"Back": "Four"}}])
        assert results[0]["status"] == "updated"

        note = (await wrapper.list_notes(ids=[basic_note]))["notes"][0]
        assert note["content"]["Back"] == "Four"
        assert note["content"]["Front"] == "What is 2+2?"  # unchanged

    async def test_update_tags(self, wrapper, basic_note):
        results = await wrapper.upsert_notes([{"id": basic_note, "tags": ["new-tag"]}])
        assert results[0]["status"] == "updated"

        note = (await wrapper.list_notes(ids=[basic_note]))["notes"][0]
        assert note["tags"] == ["new-tag"]

    async def test_update_move_deck(self, wrapper, basic_note):
        results = await wrapper.upsert_notes([{"id": basic_note, "deck": "Other"}])
        assert results[0]["status"] == "updated"

        note = (await wrapper.list_notes(ids=[basic_note]))["notes"][0]
        assert note["deck"] == "Other"

    async def test_update_nonexistent_note(self, wrapper):
        results = await wrapper.upsert_notes([{"id": 9999999999999, "fields": {"Front": "Q"}}])
        assert results[0]["status"] == "error"
        assert "not found" in results[0]["error"].lower()

    async def test_update_cannot_change_note_type(self, wrapper, basic_note):
        results = await wrapper.upsert_notes([{"id": basic_note, "note_type": "Cloze"}])
        assert results[0]["status"] == "error"
        assert "cannot change" in results[0]["error"].lower()

    async def test_update_invalid_field_name(self, wrapper, basic_note):
        results = await wrapper.upsert_notes(
            [{"id": basic_note, "fields": {"Nonexistent": "value"}}]
        )
        assert results[0]["status"] == "error"
        assert results[0]["reason"] == "unknown_field"


async def _count(wrapper) -> int:
    return await wrapper.run(lambda c: len(c.find_notes("")))


class TestDuplicatePolicy:
    """on_duplicate governs exact first-field duplicates (Anki's rule)."""

    BASIC = {"deck": "Test", "note_type": "Basic", "fields": {"Front": "Dup", "Back": "A"}}

    async def test_duplicate_errors_by_default(self, wrapper):
        first = await wrapper.upsert_notes([self.BASIC])
        assert first[0]["status"] == "created"

        again = await wrapper.upsert_notes(
            [{**self.BASIC, "fields": {"Front": "Dup", "Back": "B"}}]
        )
        assert again[0]["status"] == "error"
        assert again[0]["reason"] == "duplicate"
        assert await _count(wrapper) == 1  # not written

    async def test_duplicate_skip(self, wrapper):
        await wrapper.upsert_notes([self.BASIC])
        result = await wrapper.upsert_notes([self.BASIC], on_duplicate="skip")
        assert result[0]["status"] == "skipped"
        assert result[0]["reason"] == "duplicate"
        assert await _count(wrapper) == 1

    async def test_duplicate_allow(self, wrapper):
        await wrapper.upsert_notes([self.BASIC])
        result = await wrapper.upsert_notes([self.BASIC], on_duplicate="allow")
        assert result[0]["status"] == "created"
        assert await _count(wrapper) == 2

    async def test_empty_first_field_always_errors(self, wrapper):
        # Structural problems are rejected regardless of on_duplicate.
        result = await wrapper.upsert_notes(
            [{"deck": "Test", "note_type": "Basic", "fields": {"Front": "", "Back": "A"}}],
            on_duplicate="allow",
        )
        assert result[0]["status"] == "error"
        assert result[0]["reason"] == "empty"
        assert await _count(wrapper) == 0

    async def test_missing_cloze_errors(self, wrapper):
        result = await wrapper.upsert_notes(
            [{"deck": "Test", "note_type": "Cloze", "fields": {"Text": "no cloze here"}}]
        )
        assert result[0]["status"] == "error"
        assert result[0]["reason"] == "missing_cloze"

    async def test_unknown_note_type_reason(self, wrapper):
        result = await wrapper.upsert_notes(
            [{"deck": "Test", "note_type": "Nope", "fields": {"Front": "Q", "Back": "A"}}]
        )
        assert result[0]["status"] == "error"
        assert result[0]["reason"] == "unknown_note_type"


class TestDryRun:
    """dry_run validates every note and writes nothing."""

    async def test_would_create_writes_nothing(self, wrapper):
        result = await wrapper.upsert_notes(
            [{"deck": "Test", "note_type": "Basic", "fields": {"Front": "Q", "Back": "A"}}],
            dry_run=True,
        )
        assert result[0] == {"status": "ok", "index": 0, "action": "create"}
        assert await _count(wrapper) == 0

    async def test_reports_duplicate_without_writing(self, wrapper):
        await wrapper.upsert_notes(
            [{"deck": "Test", "note_type": "Basic", "fields": {"Front": "Dup", "Back": "A"}}]
        )
        result = await wrapper.upsert_notes(
            [{"deck": "Test", "note_type": "Basic", "fields": {"Front": "Dup", "Back": "B"}}],
            dry_run=True,
        )
        assert result[0]["status"] == "error"
        assert result[0]["reason"] == "duplicate"
        assert await _count(wrapper) == 1  # only the original

    async def test_would_update_does_not_change_note(self, wrapper, basic_note):
        result = await wrapper.upsert_notes(
            [{"id": basic_note, "fields": {"Back": "Changed"}}], dry_run=True
        )
        assert result[0] == {"status": "ok", "index": 0, "action": "update"}
        note = (await wrapper.list_notes(ids=[basic_note]))["notes"][0]
        assert note["content"]["Back"] == "4"  # unchanged

    async def test_mixed_sanity_check(self, wrapper, basic_note):
        # basic_note has Front "What is 2+2?"; default on_duplicate=error.
        result = await wrapper.upsert_notes(
            [
                {"deck": "Test", "note_type": "Basic", "fields": {"Front": "Fresh", "Back": "x"}},
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "What is 2+2?", "Back": "y"},
                },
                {"deck": "Test", "note_type": "Basic", "fields": {"Front": "", "Back": "z"}},
            ],
            dry_run=True,
        )
        assert [r["status"] for r in result] == ["ok", "error", "error"]
        assert [r.get("reason") for r in result] == [None, "duplicate", "empty"]
        assert await _count(wrapper) == 1  # only basic_note
