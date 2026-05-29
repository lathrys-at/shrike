"""Tests for ShrikeClient transparent batching logic."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from shrike.client import ShrikeClient


@pytest.fixture()
def client():
    return ShrikeClient("http://fake:9999/mcp", autostart=False)


class TestUpsertNotesBatching:
    def test_small_batch_single_call(self, client):
        notes = [
            {"deck": "D", "note_type": "Basic", "fields": {"Front": f"Q{i}", "Back": f"A{i}"}}
            for i in range(5)
        ]
        mock_response = {"results": [{"status": "created", "id": i} for i in range(5)]}

        with patch.object(client, "call", return_value=mock_response) as mock_call:
            result = client.upsert_notes(notes)

        mock_call.assert_called_once_with("upsert_notes", {"notes": notes})
        assert len(result["results"]) == 5

    def test_large_batch_split_into_chunks(self, client):
        notes = [
            {"deck": "D", "note_type": "Basic", "fields": {"Front": f"Q{i}", "Back": f"A{i}"}}
            for i in range(250)
        ]

        call_count = 0
        chunks_received: list[int] = []

        def fake_call(tool_name, args):
            nonlocal call_count
            call_count += 1
            chunk = args["notes"]
            chunks_received.append(len(chunk))
            return {"results": [{"status": "created", "id": i} for i in range(len(chunk))]}

        with patch.object(client, "call", side_effect=fake_call):
            result = client.upsert_notes(notes)

        assert call_count == 3
        assert chunks_received == [100, 100, 50]
        assert len(result["results"]) == 250


class TestUpsertNoteTypesBatching:
    def test_small_batch_single_call(self, client):
        types = [{"name": f"T{i}", "fields": ["F"], "templates": [], "css": ""} for i in range(5)]
        mock_response = {"results": [{"status": "created", "id": i} for i in range(5)]}

        with patch.object(client, "call", return_value=mock_response) as mock_call:
            result = client.upsert_note_types(types)

        mock_call.assert_called_once_with("upsert_note_types", {"note_types": types})
        assert len(result["results"]) == 5

    def test_large_batch_split_into_chunks(self, client):
        types = [{"name": f"T{i}", "fields": ["F"], "templates": [], "css": ""} for i in range(25)]

        call_count = 0
        chunks_received: list[int] = []

        def fake_call(tool_name, args):
            nonlocal call_count
            call_count += 1
            chunk = args["note_types"]
            chunks_received.append(len(chunk))
            return {"results": [{"status": "created", "id": i} for i in range(len(chunk))]}

        with patch.object(client, "call", side_effect=fake_call):
            result = client.upsert_note_types(types)

        assert call_count == 3
        assert chunks_received == [10, 10, 5]
        assert len(result["results"]) == 25


class TestDeleteNotesBatching:
    def test_small_batch_single_call(self, client):
        ids = list(range(50))
        mock_response = {"deleted": ids, "not_found": []}

        with patch.object(client, "call", return_value=mock_response) as mock_call:
            result = client.delete_notes(ids)

        mock_call.assert_called_once_with("delete_notes", {"ids": ids})
        assert result["deleted"] == ids

    def test_large_batch_split_into_chunks(self, client):
        ids = list(range(250))

        call_count = 0

        def fake_call(tool_name, args):
            nonlocal call_count
            call_count += 1
            chunk_ids = args["ids"]
            return {"deleted": chunk_ids, "not_found": []}

        with patch.object(client, "call", side_effect=fake_call):
            result = client.delete_notes(ids)

        assert call_count == 3
        assert len(result["deleted"]) == 250
        assert result["not_found"] == []

    def test_large_batch_merges_not_found(self, client):
        ids = list(range(150))

        def fake_call(tool_name, args):
            chunk_ids = args["ids"]
            return {"deleted": chunk_ids[:-1], "not_found": [chunk_ids[-1]]}

        with patch.object(client, "call", side_effect=fake_call):
            result = client.delete_notes(ids)

        assert len(result["deleted"]) == 148
        assert len(result["not_found"]) == 2
