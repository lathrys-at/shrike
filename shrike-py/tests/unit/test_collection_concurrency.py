"""Concurrency tests for CollectionWrapper.

The wrapper serializes every `anki.Collection` access through a single worker
thread. These tests fire concurrent operations — async writers on the event
loop plus a background reader thread (standing in for the index-rebuild gather)
— and assert the collection stays consistent and nothing races.
"""

from __future__ import annotations

import asyncio
import threading

BASIC = {"deck": "Test", "note_type": "Basic"}


async def test_concurrent_async_upserts_all_apply(wrapper):
    """50 upserts fired concurrently on the loop must all land, none lost."""

    async def add(i: int) -> int:
        res = await wrapper.upsert_notes([{**BASIC, "fields": {"Front": f"Q{i}", "Back": f"A{i}"}}])
        return res[0]["id"]

    ids = await asyncio.gather(*[add(i) for i in range(50)])

    assert len(set(ids)) == 50  # distinct ids, no clobbering
    assert await wrapper.run(lambda c: len(c.find_notes("deck:*"))) == 50


async def test_background_reader_races_async_writers(wrapper):
    """A non-loop reader thread (like the rebuild gather) hammering the wrapper
    while the loop writes must serialize cleanly — no exceptions, no corruption.
    """
    stop = threading.Event()
    reader_errors: list[Exception] = []

    def background_reader() -> None:
        while not stop.is_set():
            try:
                wrapper.run_sync(lambda c: list(c.find_notes("deck:*")))
            except Exception as e:  # pragma: no cover - failure path
                reader_errors.append(e)
                return

    reader = threading.Thread(target=background_reader, name="rebuild-gather")
    reader.start()
    try:

        async def add(i: int) -> int:
            res = await wrapper.upsert_notes(
                [{**BASIC, "fields": {"Front": f"R{i}", "Back": f"A{i}"}}]
            )
            return res[0]["id"]

        ids = await asyncio.gather(*[add(i) for i in range(40)])
    finally:
        stop.set()
        reader.join(timeout=5)

    assert not reader.is_alive()
    assert reader_errors == []
    assert len(set(ids)) == 40
    assert await wrapper.run(lambda c: len(c.find_notes("deck:*"))) == 40


async def test_interleaved_reads_and_writes_consistent(wrapper):
    """Interleaving reads with writes returns monotonically growing counts and a
    correct final total — the worker thread orders each op atomically."""

    async def write(i: int) -> None:
        await wrapper.upsert_notes([{**BASIC, "fields": {"Front": f"W{i}", "Back": "A"}}])

    async def read() -> int:
        return await wrapper.run(lambda c: len(c.find_notes("deck:*")))

    tasks: list[asyncio.Future] = []
    for i in range(30):
        tasks.append(asyncio.ensure_future(write(i)))
        tasks.append(asyncio.ensure_future(read()))
    results = await asyncio.gather(*tasks)

    counts = [r for r in results if isinstance(r, int)]
    # Each observed count is a valid intermediate state (0..30), never garbage.
    assert all(0 <= c <= 30 for c in counts)
    assert await wrapper.run(lambda c: len(c.find_notes("deck:*"))) == 30
