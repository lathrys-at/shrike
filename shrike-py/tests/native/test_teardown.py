"""Interpreter-teardown pins (#374 design 7; #435).

The kernel's runtime threads outlive any kernel (the process-global
runtime is never shut down); the hazard is a Python-touching task being
mid-GIL-attach while the interpreter finalizes. The clean path is
`kernel.close()` (which drains the actor), but the pins here are the UNCLEAN
paths: open a kernel, attach a Python backend, do real work, and exit
WITHOUT closing — the process must still exit cleanly (no segfault, no
hang, no abort).

The #435 lesson: "every op completed" is not enough. The bridge waker's
`call_soon_threadsafe` releases the GIL inside the call, so the loop thread
can observe the result, finish `main()`, and reach `Py_Finalize` while the
waker thread is still inside its gilstate window — on CPython 3.12/Linux
that aborted the process (`PyGILState_Release: thread state ... must be
current when releasing`). The finalization gate (`finalize_gate.rs`, armed
via atexit) drains those windows before finalization begins; these pins
cover the quiesced exit, the ops-still-in-flight exit, that same exit with
the pyo3-log bridge installed (#450 — its `Python::attach` rides the gate
too), and the gate's refusal path directly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

shrike_native = pytest.importorskip("shrike_native")

_OPEN_KERNEL = """
import asyncio
import shrike_native

class StubBackend:
    modalities = frozenset({"text"})
    def embed_texts(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    def model_fingerprint(self):
        return "stub:v1"
    def embedding_dim(self):
        return 4

async def open_kernel(collection_path, cache_dir):
    kernel = await shrike_native.async_kernel_open(collection_path, cache_dir)
    kernel.attach_embedder(shrike_native.PyEmbedder.capture(StubBackend()))
    await kernel.reindex_if_needed()
    return kernel
"""


def _run_teardown_script(tmp_path, body: str) -> None:
    """Run a teardown scenario in a fresh interpreter; it must exit rc=0."""
    script = (
        textwrap.dedent(_OPEN_KERNEL)
        + textwrap.dedent(body).format(
            collection=str(tmp_path / "collection.anki2"), cache=str(tmp_path / "cache")
        )
        + '\nprint("REACHED-EXIT")\n'
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        # Diagnosis rig for the intermittent Linux SIGABRT (#435): faulthandler
        # dumps every Python thread's stack into the captured stderr on a fatal
        # signal, and a Rust panic (if that's what aborts) carries its backtrace.
        env={**os.environ, "PYTHONFAULTHANDLER": "1", "RUST_BACKTRACE": "1"},
    )
    if proc.returncode != 0:
        # Assertion explanations can be truncated by the report; the captured
        # stderr stream is shown in full on failure — put the whole transcript
        # there so a CI flake carries its own diagnosis (#435).
        sys.stderr.write(
            f"teardown subprocess rc={proc.returncode}\n"
            f"--- subprocess stdout ---\n{proc.stdout}\n"
            f"--- subprocess stderr ---\n{proc.stderr}\n"
        )
    assert proc.returncode == 0, f"unclean teardown: rc={proc.returncode}\n{proc.stderr}"
    assert "REACHED-EXIT" in proc.stdout


def test_exit_without_close_is_clean(tmp_path) -> None:
    """The original #374 D pin: ops run to COMPLETION, exit without close()."""
    _run_teardown_script(
        tmp_path,
        """
        async def main():
            kernel = await open_kernel({collection!r}, {cache!r})
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            results = await kernel.upsert_notes(
                [(basic, 1, ["teardown pin", "no close"], [])], "error"
            )
            assert all(r[0] == "created" for r in results)
            hits = await kernel.search("teardown", 3)
            assert hits
            # Deliberately NO kernel.close(): the unclean exit path.

        asyncio.run(main())
        """,
    )


def test_exit_with_inflight_ops_is_clean(tmp_path) -> None:
    """Ops still IN FLIGHT at exit (#435's widened window).

    Un-awaited ops keep running on the kernel runtime after the loop closes
    (`spawn_op` detaches observation, never aborts the work), so their embeds
    and completion wakes race interpreter finalization head-on. The gate must
    refuse those late attach windows; the process must still exit rc=0.
    """
    _run_teardown_script(
        tmp_path,
        """
        async def main():
            kernel = await open_kernel({collection!r}, {cache!r})
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            for i in range(4):
                asyncio.ensure_future(kernel.upsert_notes(
                    [(basic, 1, [f"inflight {{i}}", "x"], [])], "allow"
                ))
                asyncio.ensure_future(kernel.search("inflight", 3))
            # Return immediately: the ops above are mid-flight on the kernel
            # runtime while asyncio.run tears the loop down and the
            # interpreter exits.

        asyncio.run(main())
        """,
    )


def test_exit_with_logging_bridge_and_inflight_ops_is_clean(tmp_path) -> None:
    """The pyo3-log attach path under the gate (#450).

    Every server process calls `init_logging()`, after which ANY native
    `log`/`tracing` emission from a kernel-runtime thread attaches the GIL
    inside pyo3-log — an attach window the #449 site-by-site pins never
    exercised (none of the other teardown scripts install the bridge). Run
    the in-flight-ops exit with the bridge installed, Python logging
    configured, and native log levels wide open: late emissions racing
    finalization must be dropped by the gated logger, never abort the exit.
    """
    _run_teardown_script(
        tmp_path,
        """
        import logging
        logging.basicConfig(level=logging.DEBUG)
        shrike_native.init_logging()

        async def main():
            kernel = await open_kernel({collection!r}, {cache!r})
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            for i in range(4):
                asyncio.ensure_future(kernel.upsert_notes(
                    [(basic, 1, [f"late log {{i}}", "x"], [])], "allow"
                ))
                asyncio.ensure_future(kernel.search("late", 3))
            # Return immediately: the ops above (and whatever they log) are
            # mid-flight on the kernel runtime while the interpreter exits.

        asyncio.run(main())
        """,
    )


def test_exit_after_gate_close_drops_late_python_work(tmp_path) -> None:
    """The gate's refusal path, exercised deterministically.

    Closing the gate (the atexit hook, called early here) must make every
    later Python-touching kernel callback refuse instead of attach: wakes are
    dropped, backend dispatches error — and the exit stays clean. This is the
    direct pin on `finalize_gate_close`; the atexit registration re-runs it
    harmlessly (idempotent).
    """
    _run_teardown_script(
        tmp_path,
        """
        async def main():
            kernel = await open_kernel({collection!r}, {cache!r})
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            # Quiesce, then close the gate — everything after this point is
            # exactly what a post-atexit world looks like.
            shrike_native.finalize_gate_close()
            # A new op's embed dispatch is refused (Unavailable) and its
            # completion wake is dropped — so it must NOT be awaited (the
            # bridge future can never resolve once wakes are refused).
            asyncio.ensure_future(kernel.upsert_notes(
                [(basic, 1, ["late", "x"], [])], "allow"
            ))

        asyncio.run(main())
        """,
    )
