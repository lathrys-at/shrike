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
        mock_response = {
            "results": [{"status": "created", "id": i, "name": f"T{i}"} for i in range(5)]
        }

        with patch.object(client, "_call", return_value=mock_response) as mock_call:
            result = client.upsert_notes(notes)

        mock_call.assert_called_once_with(
            "upsert_notes",
            {
                "notes": notes,
                "top_k_neighbors": 5,
                "on_duplicate": "error",
                "dry_run": False,
            },
        )
        assert len(result.results) == 5

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
            return {
                "results": [{"status": "created", "id": i, "name": "T"} for i in range(len(chunk))]
            }

        with patch.object(client, "_call", side_effect=fake_call):
            result = client.upsert_notes(notes)

        assert call_count == 3
        assert chunks_received == [100, 100, 50]
        assert len(result.results) == 250

    def test_passes_policy_and_restores_dry_run(self, client):
        notes = [
            {"deck": "D", "note_type": "Basic", "fields": {"Front": f"Q{i}"}} for i in range(150)
        ]

        def fake_call(tool_name, args):
            # Server echoes dry_run, but _batched_call drops it when merging.
            return {
                "results": [
                    {"status": "ok", "index": i, "action": "create"}
                    for i in range(len(args["notes"]))
                ],
                "dry_run": args["dry_run"],
            }

        with patch.object(client, "_call", side_effect=fake_call) as mock_call:
            result = client.upsert_notes(notes, on_duplicate="skip", dry_run=True)

        # Policy flags reach every batch.
        for call in mock_call.call_args_list:
            assert call.args[1]["on_duplicate"] == "skip"
            assert call.args[1]["dry_run"] is True
        # dry_run survives the chunk merge.
        assert result.dry_run is True
        assert len(result.results) == 150


class TestUpsertNoteTypesBatching:
    def test_small_batch_single_call(self, client):
        types = [{"name": f"T{i}", "fields": ["F"], "templates": [], "css": ""} for i in range(5)]
        mock_response = {
            "results": [{"status": "created", "id": i, "name": f"T{i}"} for i in range(5)]
        }

        with patch.object(client, "_call", return_value=mock_response) as mock_call:
            result = client.upsert_note_types(types)

        mock_call.assert_called_once_with("upsert_note_types", {"note_types": types})
        assert len(result.results) == 5

    def test_large_batch_split_into_chunks(self, client):
        types = [{"name": f"T{i}", "fields": ["F"], "templates": [], "css": ""} for i in range(25)]

        call_count = 0
        chunks_received: list[int] = []

        def fake_call(tool_name, args):
            nonlocal call_count
            call_count += 1
            chunk = args["note_types"]
            chunks_received.append(len(chunk))
            return {
                "results": [{"status": "created", "id": i, "name": "T"} for i in range(len(chunk))]
            }

        with patch.object(client, "_call", side_effect=fake_call):
            result = client.upsert_note_types(types)

        assert call_count == 3
        assert chunks_received == [10, 10, 5]
        assert len(result.results) == 25


class TestDeleteNotesBatching:
    def test_small_batch_single_call(self, client):
        ids = list(range(50))
        mock_response = {"deleted": ids, "not_found": []}

        with patch.object(client, "_call", return_value=mock_response) as mock_call:
            result = client.delete_notes(ids)

        mock_call.assert_called_once_with("delete_notes", {"ids": ids})
        assert result.deleted == ids

    def test_large_batch_split_into_chunks(self, client):
        ids = list(range(250))

        call_count = 0

        def fake_call(tool_name, args):
            nonlocal call_count
            call_count += 1
            chunk_ids = args["ids"]
            return {"deleted": chunk_ids, "not_found": []}

        with patch.object(client, "_call", side_effect=fake_call):
            result = client.delete_notes(ids)

        assert call_count == 3
        assert len(result.deleted) == 250
        assert result.not_found == []

    def test_large_batch_merges_not_found(self, client):
        ids = list(range(150))

        def fake_call(tool_name, args):
            chunk_ids = args["ids"]
            return {"deleted": chunk_ids[:-1], "not_found": [chunk_ids[-1]]}

        with patch.object(client, "_call", side_effect=fake_call):
            result = client.delete_notes(ids)

        assert len(result.deleted) == 148
        assert len(result.not_found) == 2


class TestStoreMediaBatching:
    def test_index_rebased_across_chunks(self, client):
        # The server assigns `index` per chunk (0..n within each batch); the client
        # must re-base it to the global request position when results span batches.
        items = [{"data": "eA==", "filename": f"f{i}.png"} for i in range(15)]

        def fake_call(tool_name, args):
            chunk = args["items"]
            return {
                "results": [
                    {
                        "status": "stored",
                        "index": i,  # per-chunk index, restarts at 0 each batch
                        "filename": item["filename"],
                        "size_bytes": 1,
                        "deduped": False,
                    }
                    for i, item in enumerate(chunk)
                ]
            }

        with patch.object(client, "_call", side_effect=fake_call):
            result = client.store_media(items)

        assert [r.index for r in result.results] == list(range(15))
        assert [r.filename for r in result.results] == [f"f{i}.png" for i in range(15)]
