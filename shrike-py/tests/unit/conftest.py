from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest

from shrike.harness.collection import CollectionWrapper


def make_notes(wrapper: CollectionWrapper, notes: list[dict]) -> list[dict]:
    """Create notes synchronously through the wrapper's worker thread.

    The shared setup helper: routes through the SAME native upsert the async
    API uses (so fixtures and tests exercise one code path), usable from sync
    and async tests alike.
    """
    return wrapper.run_sync(  # type: ignore[no-any-return]
        lambda c: json.loads(c.upsert_notes(json.dumps(notes), "error", False))
    )


@pytest.fixture()
def wrapper(tmp_path):
    """Create a CollectionWrapper backed by a fresh empty Anki collection."""
    path = str(tmp_path / "collection.anki2")
    w = CollectionWrapper(path)
    yield w
    w.close()


@pytest.fixture()
def basic_note(wrapper):
    """Create a single Basic note in the Test deck and return its ID.

    Synchronous so it can serve both sync and async tests. Routes through the
    wrapper's worker thread (the same serialized path the async API uses).
    """
    results = make_notes(
        wrapper,
        [
            {
                "deck": "Test",
                "note_type": "Basic",
                "fields": {"Front": "What is 2+2?", "Back": "4"},
                "tags": ["math", "easy"],
            }
        ],
    )
    return results[0]["id"]


# ── The unit-test kernel harness ─────────────────────────────────────────────
#
# The tool-layer unit suites drive a REAL AsyncKernel — the same kernel the
# server assembles — from a dedicated asyncio loop thread. Every kernel
# awaitable is created and awaited ON that loop (the binding's completion bridge
# binds to the running loop at call time), and the sync test body waits on a
# thread-safe future.


@pytest.fixture(scope="session")
def _driven_runtime():
    """Install + park the kernel's committed driver threads for the session.

    The kernel runs a harness-driven ``current_thread`` runtime with no lazy
    fallback, so the ``KernelHarness`` (a real ``AsyncKernel``) only makes
    progress while a driver thread drives it. Install once (the seam is set-once
    and the threads outlive any kernel, as in production) via the production
    :class:`DrivenRuntime`, and tear down at session end."""
    from shrike.platform.driven_runtime import DrivenRuntime

    runtime = DrivenRuntime()
    runtime.install()
    runtime.start()
    yield
    runtime.shutdown()


@pytest.fixture(scope="session")
def kernel_loop(_driven_runtime):
    """One asyncio loop on a dedicated daemon thread for the whole session.

    Per-test kernels are cheap (tests/native opens one per test at ~ms cost);
    the loop thread is the only shared piece, carrying no per-test state. Depends
    on ``_driven_runtime`` so the kernel runtime is installed + driven before any
    harness opens a kernel on this loop.
    """
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="kernel-loop", daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=10)
    loop.close()


class EmbedRecorder:
    """A minimal kernel-slot backend that records every embed call.

    Embeds everything to [0, 1] (orthogonal to the [1, 0] the search tests use
    as the query vector). The ``calls`` log is how the metadata-bump tests
    assert "no re-embed" — vectors untouched means no new embed call.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.0, 1.0] for _ in texts]

    def model_fingerprint(self) -> str:
        return "unit:recorder:v1"

    def embedding_dim(self) -> int:
        return 2


class KernelProxy:
    """A delegating AsyncKernel stand-in for failure injection / spying.

    ``__getattr__`` forwards to the real kernel; a test sets an instance
    attribute to shadow one op (e.g. make ``forget_notes`` raise) or calls
    :meth:`spy` to count an op's calls while still delegating. NOTE: the
    search action passes the kernel handle into native code (a PyRef), so a
    proxy can stand in only where ``search_notes`` is never exercised.
    """

    def __init__(self, kernel: Any) -> None:
        self._kernel = kernel
        self.calls: dict[str, int] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kernel, name)

    def spy(self, name: str) -> None:
        orig = getattr(self._kernel, name)
        self.calls.setdefault(name, 0)

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            self.calls[name] += 1
            return orig(*args, **kwargs)

        setattr(self, name, _wrapped)


class KernelHarness:
    """A real ``AsyncKernel`` + kernel-mode ``CollectionWrapper`` per test.

    Writes route through the kernel's maintained ops (index + derived +
    watermarks crate-side), and assertions read observable state
    (``index_status_json``, the shared engine handle) instead of mock call
    counts.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, tmp_path: Any) -> None:
        import shrike_native

        self._loop = loop
        self._native = shrike_native
        self.collection_path = str(tmp_path / "kernel-collection.anki2")
        self.cache_dir = str(tmp_path / "kernel-cache")

        async def _open() -> Any:
            return await shrike_native.async_kernel_open(self.collection_path, self.cache_dir)

        self.kernel = self.run(_open())
        self.wrapper = CollectionWrapper.over_kernel(self.kernel, self.collection_path)

    # -- plumbing ------------------------------------------------------------

    def run(self, coro: Any, timeout: float = 120.0) -> Any:
        """Run a coroutine on the kernel loop and return its result.

        Kernel awaitables must be *created* on the loop too (the bridge grabs
        the running loop at call time), so pass a coroutine whose body makes
        the kernel calls — never a pre-built kernel awaitable.
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def call_tool(self, mcp: Any, name: str, args: dict[str, Any] | None = None) -> Any:
        """Call a registered MCP tool on the kernel loop; returns the structured result."""

        async def _go() -> Any:
            _, structured = await mcp.call_tool(name, args or {})
            return structured

        return self.run(_go())

    def close(self) -> None:
        self.wrapper.close()

        async def _close() -> None:
            await self.kernel.close()

        self.run(_close())

    # -- kernel conveniences ---------------------------------------------------

    @property
    def engine(self) -> Any:
        """The kernel's Arc-shared native engine handle (the vectors it maintains)."""
        return self.kernel.engine_handle()

    def attach_embedder(
        self,
        backend: Any,
        read: Any = None,
        exists: Any = None,
        *,
        reindex: bool = True,
    ) -> None:
        """Capture *backend* into the kernel's embed slot; reindex by default
        so the index materializes and reads ``ready``."""

        async def _go() -> None:
            self.kernel.attach_embedder(self._native.PyEmbedder.capture(backend), read, exists)
            if reindex:
                await self.kernel.reindex_if_needed()

        self.run(_go())

    def upsert_notes(
        self, notes: list[dict], on_duplicate: str = "allow", dry_run: bool = False
    ) -> list[dict]:
        """Seed/edit notes through the kernel's maintained op (index + derived
        + watermarks ride along, exactly like the served upsert path)."""

        async def _go() -> Any:
            return json.loads(
                await self.kernel.upsert_notes_json(json.dumps(notes), on_duplicate, dry_run)
            )

        return self.run(_go())

    def seed_note(self, front: str, *, deck: str = "Test", back: str = "x", tags=None) -> int:
        results = self.upsert_notes(
            [
                {
                    "deck": deck,
                    "note_type": "Basic",
                    "fields": {"Front": front, "Back": back},
                    "tags": list(tags or []),
                }
            ]
        )
        assert results[0]["status"] == "created", results[0]
        return int(results[0]["id"])

    def index_status(self) -> dict[str, Any]:
        return json.loads(self.kernel.index_status_json())  # type: ignore[no-any-return]

    def col_mod(self) -> int:
        return self.run(self.wrapper.col_mod())  # type: ignore[no-any-return]

    def reindex_if_needed(self) -> bool:
        async def _go() -> bool:
            return await self.kernel.reindex_if_needed()  # type: ignore[no-any-return]

        return self.run(_go())  # type: ignore[no-any-return]

    def proxy(self) -> KernelProxy:
        return KernelProxy(self.kernel)


@pytest.fixture()
def kharness(tmp_path, kernel_loop):
    """A fresh kernel harness per test (own temp collection + cache dir)."""
    h = KernelHarness(kernel_loop, tmp_path)
    yield h
    h.close()


@pytest.fixture()
def kbasic_note(kharness):
    """A single Basic note in the Test deck, seeded through the kernel."""
    return kharness.seed_note("What is 2+2?", back="4", tags=["math", "easy"])
