"""Busy/cooperative surface (#278 step 6): release/reopen + the BUSY tier.

The cross-process case is the real #64/#65 story: a pip-core CollectionWrapper
in a SUBPROCESS holds the collection while the native core (this process)
tries to re-acquire — NativeBusyError, retryable; once the holder exits, the
reopen succeeds. (The two cores never co-manage: the holder exists to hold.)
"""

from __future__ import annotations

import subprocess
import sys

import pytest

shrike_native = pytest.importorskip("shrike_native")

from .conftest import requires_anki_core  # noqa: E402

pytestmark = requires_anki_core

# Holds the collection until stdin closes (the parent controls the window).
_HOLDER = r"""
import sys
from shrike.collection import CollectionWrapper
w = CollectionWrapper(sys.argv[1])
print("HELD", flush=True)
sys.stdin.readline()  # parent closes stdin to release
w.close()
print("RELEASED", flush=True)
"""


def test_cross_process_busy_then_reopen(tmp_path, native_core):
    work = tmp_path / "busy"
    work.mkdir()
    path = str(work / "collection.anki2")
    # native_core holds ITS OWN tmp collection; this test wants a fresh file
    # both sides reference, so open a dedicated native core on `path`.
    core = type(native_core)(path)
    try:
        basic = core.notetype_id("Basic")
        core.release()

        holder = subprocess.Popen(
            [sys.executable, "-c", _HOLDER, path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert holder.stdout is not None and holder.stdin is not None
            assert holder.stdout.readline().strip() == "HELD"

            with pytest.raises(shrike_native.NativeBusyError, match="in use by another process"):
                core.reopen()

            holder.stdin.close()  # release
            assert holder.stdout.readline().strip() == "RELEASED"
        finally:
            holder.wait(timeout=30)

        core.reopen()
        nid = core.create_note(basic, 1, ["after contention", "b"], [])
        assert isinstance(nid, int)
    finally:
        core.close()


def test_busy_error_is_retryable_type(native_core):
    """NativeBusyError is a distinct RuntimeError subclass — the facades key
    catch-and-retry on the type, never on string parsing."""
    assert issubclass(shrike_native.NativeBusyError, RuntimeError)
    assert not issubclass(shrike_native.NativeBusyError, ValueError)
