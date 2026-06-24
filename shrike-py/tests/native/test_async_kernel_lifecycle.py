"""Embedder-slot, wrapper-over-kernel, and cooperative-reopen lifecycle."""

from __future__ import annotations

import asyncio
import json

import pytest

shrike_native = pytest.importorskip("shrike_native")

pytestmark = pytest.mark.skipif(
    not hasattr(shrike_native, "async_kernel_open"),
    reason="anki-core build required (scripts/build-native.sh)",
)

from .conftest import _Backend, _open  # noqa: E402


class TestEmbedderSlot:
    def test_detach_degrades_and_reattach_recovers(self, tmp_path) -> None:
        async def flow():
            backend = _Backend()
            kernel = await _open(tmp_path, backend)
            await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")

            kernel.detach_embedder()
            assert json.loads(kernel.index_status_json())["state"] == "unavailable"
            # Creates still work and lexical search still serves.
            results = await kernel.upsert_notes(
                [(basic, 1, ["paris is the capital of france", "geo"], [])], "error"
            )
            assert results[0][0] == "created"
            # The lexical (derived) row lands off the drain on the compute pool;
            # settle the maintained tail before the read-after-write search.
            await kernel.settle()
            hits = await kernel.search("capital of france", 5)
            assert hits[0][0] == results[0][1]
            assert all(s != "text" for s, _ in hits[0][2])

            # Re-attach (a fresh capture, like an embedding restart): the index
            # watermark stayed put, so reindex embeds the note created while
            # detached.
            kernel.attach_embedder(shrike_native.PyEmbedder.capture(backend))
            assert json.loads(kernel.index_status_json())["state"] == "ready"
            assert await kernel.reindex_if_needed()
            hits = await kernel.search("capital of france", 5)
            assert any(s == "text" for s, _ in hits[0][2])
            await kernel.close()

        asyncio.run(flow())


class TestWrapperOverKernel:
    def test_wrapper_ops_serialize_through_the_kernel(self, tmp_path) -> None:
        from shrike.harness.collection import CollectionWrapper

        async def flow():
            kernel = await shrike_native.async_kernel_open(
                str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
            )
            wrapper = CollectionWrapper.over_kernel(kernel, str(tmp_path / "collection.anki2"))
            # The wrapper's async surface rides run_job over the shared core.
            assert await wrapper.col_mod() >= 0
            notes = await wrapper.upsert_notes(
                [
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {"Front": "via wrapper", "Back": "b"},
                    }
                ]
            )
            assert notes[0]["status"] == "created"
            # The kernel sees the same collection (one shared core).
            assert await kernel.col_mod() >= 0

            # Loop-free phases don't exist in kernel mode.
            with pytest.raises(RuntimeError, match="kernel mode"):
                wrapper.run_sync(lambda c: c.col_mod())
            with pytest.raises(RuntimeError, match="kernel mode"):
                wrapper.release_now()

            wrapper.close()  # must NOT close the kernel's core
            assert await kernel.col_mod() >= 0
            await kernel.close()

        asyncio.run(flow())


class TestCooperativeReopen:
    def test_kernel_writes_self_heal_after_release(self, tmp_path) -> None:
        # An idle release closes the collection; kernel write ops must not
        # error CollectionNotOpen — kernel-side ensure_open reopens on demand.
        async def flow():
            kernel = await _open(tmp_path, _Backend())
            await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")

            await kernel.release()
            results = await kernel.upsert_notes(
                [(basic, 1, ["written while released", "b"], [])], "error"
            )
            assert results[0][0] == "created"

            await kernel.release()
            wire = json.loads(
                await kernel.upsert_notes_json(
                    json.dumps(
                        [
                            {
                                "note_type": "Basic",
                                "deck": "Default",
                                "fields": {"Front": "wire after release", "Back": "b"},
                            }
                        ]
                    ),
                    "error",
                    False,
                )
            )
            assert wire[0]["status"] == "created"
            await kernel.close()

        asyncio.run(flow())
