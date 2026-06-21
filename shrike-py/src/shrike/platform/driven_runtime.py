"""The committed threads that drive the kernel's current_thread runtime.

The kernel runs a single ``current_thread`` tokio runtime and spawns no threads
of its own; the harness donates every thread it needs. The server installs the
driven runtime at boot, then this module spawns the committed ``N + 2`` threads:
one IO/timer driver, one serialized-collection thread, and ``N`` CPU-compute
workers. Each parks in a GIL-releasing native loop for the server's life, so the
asyncio loop holds the GIL only for its own callbacks and the brief FFI hops.

The threads are non-daemon: shutdown closes the kernel's pool queues
(``drive_pools_shutdown``), which lets each loop return, and then this module
joins them before the interpreter finalizes — daemon threads would risk being
torn down mid-kernel-work, the finalization-abort class the binding's finalize
gate guards against.
"""

from __future__ import annotations

import logging
import os
import threading

import shrike_native

logger = logging.getLogger("shrike.server")

# CPU-compute worker count bounds. At least two so independent engine batches
# overlap (the "N >= 2" property); capped so a many-core host doesn't commit an
# unreasonable number of parked threads (they are cheap when idle, but the
# committed-pool model wants a predictable count).
MIN_COMPUTE_THREADS = 2
MAX_COMPUTE_THREADS = 4

# How long to wait for each committed thread to return after the pools are
# closed. The kernel is quiesced before shutdown, so the join is normally
# immediate; the bound only keeps a wedged thread from hanging process exit.
JOIN_TIMEOUT_SECONDS = 5.0


def _compute_thread_count() -> int:
    cpu = os.cpu_count() or MIN_COMPUTE_THREADS
    return max(MIN_COMPUTE_THREADS, min(cpu, MAX_COMPUTE_THREADS))


class DrivenRuntime:
    """Owns the committed driver threads for the driven kernel runtime.

    ``install()`` puts the runtime in driven mode (set-once, before any kernel
    op). ``start()`` spawns the ``N + 2`` driver threads. ``shutdown()`` closes
    the pools and joins them. Built once per server process; the integration
    suite spawns a fresh subprocess per server, so the process-global runtime
    seam is never contended in-process.
    """

    def __init__(self, *, compute_threads: int | None = None) -> None:
        self._compute_threads = (
            compute_threads if compute_threads is not None else _compute_thread_count()
        )
        self._threads: list[threading.Thread] = []
        self._driven = False

    def install(self) -> None:
        """Install the driven runtime — call ONCE, before any kernel op (the
        kernel has no lazy fallback, so an op before this panics). Records whether
        the runtime is installed: in the normal server process it always is
        (nothing has touched the runtime yet). The seam is set-once, so a reused
        process where it was already installed still reports ``True`` and
        :meth:`start` is idempotent either way."""
        self._driven = bool(shrike_native.init_driven_runtime(self._compute_threads))

    def start(self) -> None:
        """Spawn the committed N + 2 driver threads, each parked in its native
        drive loop. A no-op when driven mode isn't active (an un-installed or
        already-default runtime) and idempotent (a second call after threads are
        live).

        The IO thread is started and confirmed driving (``runtime_probe``) before
        the collection/compute threads are spawned. tokio's ``current_thread``
        runtime gives ownership of the IO/timer drivers to the FIRST ``block_on``
        caller, which MUST be the IO thread; a collection/compute leaf reaching
        its own ``block_on`` first would win that ownership and leave the drivers
        advancing only while it parks in ``recv``, starving timers/IO."""
        if not self._driven or self._threads:
            return
        io = threading.Thread(target=shrike_native.drive_io, name="shrike-io")
        self._threads.append(io)
        io.start()
        # The barrier: returns once the IO thread is inside its block_on and owns
        # the drivers, so the leaves below can't claim driver ownership.
        shrike_native.runtime_probe()
        leaves = [threading.Thread(target=shrike_native.drive_collection, name="shrike-collection")]
        leaves += [
            threading.Thread(target=shrike_native.drive_compute, name=f"shrike-work-{i}")
            for i in range(self._compute_threads)
        ]
        for thread in leaves:
            self._threads.append(thread)
            thread.start()
        logger.info(
            "Driven runtime: %d committed threads (1 io, 1 collection, %d compute)",
            len(self._threads),
            self._compute_threads,
        )

    def shutdown(self) -> None:
        """Close the kernel's pool queues + trip the IO shutdown signal, then
        join every committed thread (bounded). Call AFTER kernel work has
        quiesced (the collection actor drained), so the queues close and the
        joins are immediate. Idempotent; a thread that fails to return inside
        the bound is logged and left rather than hanging exit."""
        shrike_native.drive_pools_shutdown()
        for thread in self._threads:
            thread.join(timeout=JOIN_TIMEOUT_SECONDS)
            if thread.is_alive():
                logger.warning(
                    "Driven thread %s did not return within %.0fs of shutdown",
                    thread.name,
                    JOIN_TIMEOUT_SECONDS,
                )
        self._threads = []
