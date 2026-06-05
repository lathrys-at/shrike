"""Cooperative-locking lifecycle tests (#64).

Exercises CollectionWrapper's open-on-demand / idle-release behaviour and the
acquire drift hook, plus that the default (permanent-hold) mode is unaffected.
"""

from __future__ import annotations

import asyncio

import pytest

from shrike.collection import CollectionWrapper


@pytest.fixture()
def coop(tmp_path):
    """A cooperative wrapper with a tiny idle-hold and a recording acquire hook."""
    calls: list[int] = []
    w = CollectionWrapper(
        str(tmp_path / "c.anki2"),
        cooperative=True,
        hold_seconds=0.05,
        on_acquire=lambda c: calls.append(c.mod),
    )
    w.acquire_calls = calls  # type: ignore[attr-defined]
    yield w
    w.close()


async def _wait_released(w, timeout=1.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while w.is_open and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.02)


class TestCooperativeLifecycle:
    async def test_releases_after_idle(self, coop):
        await coop.run(lambda c: c.note_count())
        assert coop.is_open  # held right after an op
        await _wait_released(coop)
        assert not coop.is_open  # released after the idle window

    async def test_reacquires_and_runs_hook(self, coop):
        await coop.run(lambda c: c.note_count())
        await _wait_released(coop)
        coop.acquire_calls.clear()

        # Next op re-opens and runs the acquire hook exactly once.
        result = await coop.run(lambda c: c.note_count())
        assert result == 0
        assert coop.is_open
        assert len(coop.acquire_calls) == 1

    async def test_data_survives_release(self, coop):
        def add(c):
            n = c.new_note(c.models.by_name("Basic"))
            n["Front"], n["Back"] = "persist", "x"
            c.add_note(n, c.decks.id("D"))
            return n.id

        nid = await coop.run(add)
        await _wait_released(coop)
        assert not coop.is_open
        found = await coop.run(lambda c: list(c.find_notes(f"nid:{nid}")))
        assert found == [nid]

    async def test_release_now_then_op_reacquires(self, coop):
        coop.release_now()
        assert not coop.is_open
        # An op immediately after an explicit release re-acquires cleanly.
        assert await coop.run(lambda c: c.note_count()) == 0
        assert coop.is_open

    def test_release_now_is_idempotent(self, coop):
        coop.release_now()
        coop.release_now()  # no error, still released
        assert not coop.is_open


class TestPermanentModeUnaffected:
    async def test_never_releases_and_no_hook(self, tmp_path):
        calls: list[int] = []
        w = CollectionWrapper(str(tmp_path / "c.anki2"), on_acquire=lambda c: calls.append(1))
        try:
            assert w.is_open
            assert not w.cooperative
            await w.run(lambda c: c.note_count())
            await asyncio.sleep(0.1)
            assert w.is_open  # permanent mode holds the lock
            assert calls == []  # hook never fires (no re-acquire)
        finally:
            w.close()
