"""Interpreter-teardown pins.

The kernel's runtime threads outlive any kernel (the process-global
runtime is never shut down); the hazard is a Python-touching task being
mid-GIL-attach while the interpreter finalizes. The clean path is
`kernel.close()` (which drains the actor), but the pins here are the UNCLEAN
paths: open a kernel, attach a Python backend, do real work, and exit
WITHOUT closing — the process must still exit cleanly (no segfault, no
hang, no abort).

The lesson: "every op completed" is not enough. The bridge waker's
`call_soon_threadsafe` releases the GIL inside the call, so the loop thread
can observe the result, finish `main()`, and reach `Py_Finalize` while the
waker thread is still inside its gilstate window — on CPython 3.12/Linux
that aborted the process (`PyGILState_Release: thread state ... must be
current when releasing`). The finalization gate (`finalize_gate.rs`, armed
via atexit) drains those windows before finalization begins; these pins
cover the quiesced exit, the ops-still-in-flight exit, that same exit with
the pyo3-log bridge installed (its `Python::attach` rides the gate
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
import threading
import shrike_native

class StubBackend:
    modalities = frozenset({"text"})
    def embed_texts(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    def model_fingerprint(self):
        return "stub:v1"
    def embedding_dim(self):
        return 4

def spawn_drivers():
    # The kernel runtime is harness-driven with no lazy default, so the
    # committed driver threads must be parked before any op. Honor the startup
    # barrier: io first, probe until it drives, then sync + compute. Returns the
    # threads so a test can shut down and join.
    #
    # DAEMON threads on purpose: these pins model the UNCLEAN exit (a crash /
    # interpreter teardown WITHOUT drive_pools_shutdown). A non-daemon driver
    # parks forever and would block process exit; a daemon is torn down at
    # finalization — exactly the GIL-state-abort window the finalization gate
    # guards, which is the property under test. The clean-shutdown pin still
    # joins explicitly (a daemon joins fine once the pools close).
    assert shrike_native.init_driven_runtime()
    io = threading.Thread(target=shrike_native.drive_io, name="shrike-io", daemon=True)
    io.start()
    shrike_native.runtime_probe()
    sync = threading.Thread(target=shrike_native.drive_sync, name="shrike-sync", daemon=True)
    sync.start()
    compute = [
        threading.Thread(
            target=shrike_native.drive_compute, name=f"shrike-work-{i}", daemon=True
        )
        for i in range(2)
    ]
    for t in compute:
        t.start()
    return [io, sync, *compute]

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
        # Diagnosis rig for the intermittent Linux SIGABRT: faulthandler
        # dumps every Python thread's stack into the captured stderr on a fatal
        # signal, and a Rust panic (if that's what aborts) carries its backtrace.
        env={**os.environ, "PYTHONFAULTHANDLER": "1", "RUST_BACKTRACE": "1"},
    )
    if proc.returncode != 0:
        # Assertion explanations can be truncated by the report; the captured
        # stderr stream is shown in full on failure — put the whole transcript
        # there so a CI flake carries its own diagnosis.
        sys.stderr.write(
            f"teardown subprocess rc={proc.returncode}\n"
            f"--- subprocess stdout ---\n{proc.stdout}\n"
            f"--- subprocess stderr ---\n{proc.stderr}\n"
        )
    assert proc.returncode == 0, f"unclean teardown: rc={proc.returncode}\n{proc.stderr}"
    assert "REACHED-EXIT" in proc.stdout


def test_exit_without_close_is_clean(tmp_path) -> None:
    """The original pin: ops run to COMPLETION, exit without close()."""
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
            # The write is decoupled from indexing (the ingest actor embeds +
            # indexes off the collection write), so settle before searching —
            # "ops run to COMPLETION" now includes the async index drain.
            await kernel.settle()
            hits = await kernel.search("teardown", 3)
            assert hits
            # Deliberately NO kernel.close(): the unclean exit path.

        spawn_drivers()  # park the committed threads; deliberately not joined
        asyncio.run(main())
        """,
    )


def test_exit_with_inflight_ops_is_clean(tmp_path) -> None:
    """Ops still IN FLIGHT at exit (the widened window).

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

        spawn_drivers()  # park the committed threads; deliberately not joined
        asyncio.run(main())
        """,
    )


def test_exit_with_logging_bridge_and_inflight_ops_is_clean(tmp_path) -> None:
    """The pyo3-log attach path under the gate.

    Every server process calls `init_logging()`, after which ANY native
    `log`/`tracing` emission from a kernel-runtime thread attaches the GIL
    inside pyo3-log — an attach window the site-by-site pins never
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

        spawn_drivers()  # park the committed threads; deliberately not joined
        asyncio.run(main())
        """,
    )


def test_driven_runtime_boot_serve_and_clean_shutdown(tmp_path) -> None:
    """The S5 go-live flip: a kernel on the DRIVEN current_thread runtime, with
    the harness's committed N+2 threads, boots, serves an op, and shuts down
    cleanly — joining every committed thread before the interpreter finalizes.

    This is the production server's threading model end to end: install the
    driven runtime before any op, donate one io + one sync + N compute threads
    (each GIL-released in its native drive loop), run a real upsert + search via
    the asyncio bridge (driven by the io thread, not tokio workers), then
    close() the kernel (draining the actor) and drive_pools_shutdown() so the
    loops return and join. A regression that fails to drive an op hangs (caught
    by the subprocess timeout); one that fails to close a pool hangs a join; one
    that left the threads to be torn down at finalization would risk the
    GIL-state abort the finalization gate guards — all show as rc!=0.
    """
    _run_teardown_script(
        tmp_path,
        """
        threads = spawn_drivers()

        async def main():
            kernel = await open_kernel({collection!r}, {cache!r})
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            results = await kernel.upsert_notes(
                [(basic, 1, ["driven flip", "served"], [])], "error"
            )
            assert all(r[0] == "created" for r in results), results
            hits = await kernel.search("driven", 3)
            assert hits, "the bridge served a search under the driven runtime"
            # Clean shutdown: close drains the actor (a kernel op, still driven
            # by the io thread), THEN close the pools so the committed threads
            # return.
            await kernel.close()

        asyncio.run(main())

        # The kernel is quiesced; close the pools and join every committed
        # thread before the interpreter exits.
        shrike_native.drive_pools_shutdown()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive(), f"{{t.name}} did not return after shutdown"
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

        spawn_drivers()  # park the committed threads; deliberately not joined
        asyncio.run(main())
        """,
    )
