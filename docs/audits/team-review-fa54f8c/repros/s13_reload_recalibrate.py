"""S13-1 repro (preserved by lead; rev-S13 worktree reaped).
Harness.reload() omits the secondary-floor recalibration that every other reindex
path performs (_drive_reindex:440-443, _rebuild_then_calibrate:597-601,
_drive_boot_reindex:892-895 all call _recalibrate_secondary_floors after
reindex_if_needed). After a /reload that reconciles drift (re-embeds secondary
image vectors), the cross-space image floor (#580/#576) stays stale → mis-gates.
RED at fa54f8c (control passes, reload fails). Fix: one await self._recalibrate_secondary_floors() after harness.py:943.
Run: SHRIKE_SKIP_NATIVE_STALE_CHECK=1 .venv/bin/python -m pytest <this> -q -p no:cacheprovider
"""
from __future__ import annotations

import asyncio

from shrike.harness import Harness


class _FakeKernel:
    def __init__(self):
        self.recalibrated = 0
    async def reindex_if_needed(self):
        return True
    async def calibrate_secondary_floors(self, margin):
        self.recalibrated += 1
        return []


class _FakeWrapper:
    cooperative = False
    async def reopen(self):
        return None
    async def col_mod(self):
        return 123


class _FakeRuntime:
    backend = object()


def _make_harness():
    h = Harness.__new__(Harness)
    h.kernel = _FakeKernel()
    h.wrapper = _FakeWrapper()
    h.runtime = _FakeRuntime()
    h.secondary_runtimes = []
    h.cross_space_floor_margin = 1.0
    h._bg_tasks = set()
    async def _noop_build():
        return None
    h._maybe_build_derived = _noop_build  # type: ignore[assignment]
    return h


def test_drive_boot_reindex_recalibrates():  # control — PASSES
    h = _make_harness()
    asyncio.run(h._drive_boot_reindex())
    assert h.kernel.recalibrated == 1


def test_reload_recalibrates_after_reindex():  # RED today
    h = _make_harness()
    out = asyncio.run(h.reload())
    assert out["rebuilding"] is True
    assert h.kernel.recalibrated == 1, (
        "reload() reconciled drift but never recalibrated the secondary "
        "cross-space image floor (stale floor mis-gates the image space)"
    )
