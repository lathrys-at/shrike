"""Busy-acquire contention tests (#65): cooperative re-acquire against a held lock."""

from __future__ import annotations

import pytest
from anki.collection import Collection

from shrike.collection import CollectionBusyError, CollectionWrapper
from shrike.schemas import COLLECTION_BUSY_CODE


class TestBusyAcquire:
    async def test_reacquire_against_held_lock_raises_busy(self, tmp_path):
        path = str(tmp_path / "c.anki2")
        w = CollectionWrapper(path, cooperative=True, hold_seconds=0.05)
        try:
            await w.run(lambda c: c.note_count())  # boot-open works
            w.release_now()
            assert not w.is_open

            # Another process grabs the collection while we're released.
            other = Collection(path)
            try:
                with pytest.raises(CollectionBusyError) as exc:
                    await w.run(lambda c: c.note_count())
                # Message carries the wire code so the client can detect it.
                assert str(exc.value).startswith(f"{COLLECTION_BUSY_CODE}:")
            finally:
                other.close()

            # Once the other process releases, the next op re-acquires cleanly.
            assert await w.run(lambda c: c.note_count()) == 0
            assert w.is_open
        finally:
            w.close()

    async def test_busy_message_is_actionable(self, tmp_path):
        path = str(tmp_path / "c.anki2")
        w = CollectionWrapper(path, cooperative=True, hold_seconds=0.05)
        try:
            await w.run(lambda c: c.note_count())
            w.release_now()
            other = Collection(path)
            try:
                with pytest.raises(CollectionBusyError, match="another process"):
                    await w.run(lambda c: c.note_count())
            finally:
                other.close()
        finally:
            w.close()
