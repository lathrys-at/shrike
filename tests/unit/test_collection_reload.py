"""Collection-layer tests for reopen (#79): close + re-open the handle.

Exercises CollectionWrapper.reopen and the run-at-execution-time reading of
self.col that makes a post-reopen op see the new handle.
"""

from __future__ import annotations


def _add(wrapper, front, back="x"):
    def build(c):
        n = c.new_note(c.models.by_name("Basic"))
        n["Front"], n["Back"] = front, back
        c.add_note(n, c.decks.id("D"))
        return n.id

    return wrapper.run_sync(build)


class TestReopen:
    async def test_swaps_handle(self, wrapper):
        before = id(wrapper.col)
        await wrapper.reopen()
        assert id(wrapper.col) != before  # a fresh Collection object

    async def test_preserves_committed_data(self, wrapper):
        nid = _add(wrapper, "survives-reopen")
        await wrapper.reopen()
        found = await wrapper.run(lambda c: list(c.find_notes(f"nid:{nid}")))
        assert found == [nid]
        # The note is readable through the new handle.
        content = await wrapper.run(lambda c: dict(c.get_note(nid).items()))
        assert content["Front"] == "survives-reopen"

    async def test_writable_after_reopen(self, wrapper):
        await wrapper.reopen()
        nid = _add(wrapper, "added-after-reopen")
        found = await wrapper.run(lambda c: list(c.find_notes(f"nid:{nid}")))
        assert found == [nid]

    async def test_op_after_reopen_uses_new_handle(self, wrapper):
        # The old handle is closed by reopen; reading self.col at execution time
        # means this op runs against the re-opened collection, not a closed one.
        _add(wrapper, "a")
        await wrapper.reopen()
        total = await wrapper.run(lambda c: len(list(c.find_notes("deck:*"))))
        assert total == 1
