"""Busy-acquire contention tests (#65): cooperative re-acquire against a held lock.

The holder is a real SUBPROCESS (a second native core on the same file). An
in-process second handle is not a truthful stand-in: anki's collection runs
WAL and its in-process guards differ from genuine cross-process file locking,
which is the contention the busy tier exists for (Anki desktop).
"""

from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager

import pytest

from shrike.collection import CollectionBusyError, CollectionWrapper
from shrike.schemas import COLLECTION_BUSY_CODE

# Holds the collection (open + held) until stdin closes.
_HOLDER = r"""
import sys
from shrike_native import CollectionCore
core = CollectionCore(sys.argv[1])
print("HELD", flush=True)
sys.stdin.readline()
core.close()
print("RELEASED", flush=True)
"""


@contextmanager
def _held(path: str):
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None and holder.stdin is not None
    assert holder.stdout.readline().strip() == "HELD"
    try:
        yield
    finally:
        holder.stdin.close()
        assert holder.stdout.readline().strip() == "RELEASED"
        holder.wait(timeout=30)


class TestBusyAcquire:
    async def test_reacquire_against_held_lock_raises_busy(self, tmp_path):
        path = str(tmp_path / "c.anki2")
        w = CollectionWrapper(path, cooperative=True, hold_seconds=0.05)
        try:
            await w.run(lambda c: len(c.find_notes("deck:*")))  # boot-open works
            w.release_now()
            assert not w.is_open

            # Another process grabs the collection while we're released.
            with _held(path):
                with pytest.raises(CollectionBusyError) as exc:
                    await w.run(lambda c: len(c.find_notes("deck:*")))
                # Message carries the wire code so the client can detect it.
                assert str(exc.value).startswith(f"{COLLECTION_BUSY_CODE}:")

            # Once the other process releases, the next op re-acquires cleanly.
            assert await w.run(lambda c: len(c.find_notes("deck:*"))) == 0
            assert w.is_open
        finally:
            w.close()

    async def test_busy_message_is_actionable(self, tmp_path):
        path = str(tmp_path / "c.anki2")
        w = CollectionWrapper(path, cooperative=True, hold_seconds=0.05)
        try:
            await w.run(lambda c: len(c.find_notes("deck:*")))
            w.release_now()
            with _held(path), pytest.raises(CollectionBusyError, match="another process"):
                await w.run(lambda c: len(c.find_notes("deck:*")))
        finally:
            w.close()
