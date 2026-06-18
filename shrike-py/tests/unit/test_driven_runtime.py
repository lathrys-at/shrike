"""Unit pins for the DrivenRuntime committed-thread helper.

The end-to-end driven boot/serve/shutdown is pinned natively
(tests/native/test_teardown.py); these are the fast, native-free contract pins:
the install→start→shutdown flow spawns and joins the committed threads, and the
guard that an inactive driven mode (a reused process where the default runtime
was already pinned) makes start() a no-op instead of spawning threads that would
error with no driven queues to drive.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from shrike.platform import driven_runtime
from shrike.platform.driven_runtime import DrivenRuntime


@pytest.fixture
def fake_native(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the native module the helper calls so no real runtime is touched.

    The drive_* loops would park forever, so they are no-op stand-ins: a thread
    targeting one returns immediately, which is all the join path needs to
    observe.
    """
    native = MagicMock()
    native.drive_io = lambda: None
    native.drive_sync = lambda: None
    native.drive_compute = lambda: None
    monkeypatch.setattr(driven_runtime, "shrike_native", native)
    return native


def test_install_start_shutdown_spawns_and_joins(fake_native: MagicMock) -> None:
    fake_native.init_driven_runtime.return_value = True
    rt = DrivenRuntime(compute_threads=3)

    rt.install()
    rt.start()
    assert len(rt._threads) == 5  # 1 io + 1 sync + 3 compute
    names = {t.name for t in rt._threads}
    assert "shrike-drive-io" in names
    assert "shrike-drive-sync" in names
    assert sum(n.startswith("shrike-drive-compute-") for n in names) == 3

    rt.shutdown()
    fake_native.drive_pools_shutdown.assert_called_once()
    assert rt._threads == []


def test_start_is_a_noop_when_driven_mode_inactive(fake_native: MagicMock) -> None:
    # install() reporting False (the default runtime was already pinned) must
    # make start() spawn nothing — the threads would error with no driven queues.
    fake_native.init_driven_runtime.return_value = False
    rt = DrivenRuntime(compute_threads=2)

    rt.install()
    rt.start()
    assert rt._threads == []


def test_start_is_idempotent(fake_native: MagicMock) -> None:
    fake_native.init_driven_runtime.return_value = True
    rt = DrivenRuntime(compute_threads=2)
    rt.install()
    rt.start()
    first = list(rt._threads)
    rt.start()  # a second call must not spawn another set
    assert rt._threads == first
    rt.shutdown()


def test_compute_thread_count_is_bounded() -> None:
    n = driven_runtime._compute_thread_count()
    assert driven_runtime.MIN_COMPUTE_THREADS <= n <= driven_runtime.MAX_COMPUTE_THREADS
