from __future__ import annotations


class TestDeleteNotes:
    def test_delete_existing(self, wrapper, basic_note):
        result = wrapper.delete_notes([basic_note])
        assert basic_note in result["deleted"]
        assert result["not_found"] == []

        # Verify note is gone
        list_result = wrapper.list_notes(ids=[basic_note])
        assert list_result["total"] == 0

    def test_delete_nonexistent(self, wrapper):
        result = wrapper.delete_notes([9999999999999])
        assert result["deleted"] == []
        assert 9999999999999 in result["not_found"]

    def test_delete_mixed(self, wrapper, basic_note):
        result = wrapper.delete_notes([basic_note, 9999999999999])
        assert basic_note in result["deleted"]
        assert 9999999999999 in result["not_found"]

    def test_delete_multiple(self, wrapper):
        results = wrapper.upsert_notes([
            {
                "deck": "Test",
                "note_type": "Basic",
                "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
            }
            for i in range(3)
        ])
        ids = [r["id"] for r in results]
        result = wrapper.delete_notes(ids)
        assert set(result["deleted"]) == set(ids)
        assert result["not_found"] == []
