"""Collection-layer tests for reopen: close + re-open the handle.

Exercises CollectionWrapper.reopen and the run-at-execution-time reading of
self.col that makes a post-reopen op see the new handle.
"""

from __future__ import annotations

import json


def _add(wrapper, front, back="x"):
    def build(c):
        return json.loads(
            c.upsert_notes(
                json.dumps(
                    [{"note_type": "Basic", "deck": "D", "fields": {"Front": front, "Back": back}}]
                ),
                "allow",
                False,
            )
        )[0]["id"]

    return wrapper.run_sync(build)


class TestReopen:
    async def test_swaps_handle(self, wrapper):
        # The wrapper keeps ONE native core whose reopen() swaps the underlying
        # collection handle in place — pin the observable contract: ops keep
        # working and the watermark stays readable.
        before = await wrapper.col_mod()
        await wrapper.reopen()
        assert await wrapper.col_mod() >= before  # a fresh Collection object

    async def test_preserves_committed_data(self, wrapper):
        nid = _add(wrapper, "survives-reopen")
        await wrapper.reopen()
        found = await wrapper.run(lambda c: list(c.find_notes(f"nid:{nid}")))
        assert found == [nid]

        # The note is readable through the new handle.
        def read(c):
            _, names, values = c.note_field_map([nid])[0]
            return dict(zip(names, values, strict=False))

        content = await wrapper.run(read)
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
