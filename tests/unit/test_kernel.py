"""ShrikeKernel + Scheduler port (#275).

The behavioural gate for this refactor is the integration suite passing
unmodified (the HTTP host must be byte-identical through the wire); these tests
pin the new seam itself: the Scheduler protocol shape, the WorkerScheduler's
collection-thread routing, and the kernel verbs' core semantics against the
real wrapper fixture.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

from shrike.derived import DerivedTextStore
from shrike.embedding import EmbeddingRuntime
from shrike.index import IndexSaver, VectorIndex
from shrike.kernel import KernelConfigError, Scheduler, ShrikeKernel, WorkerScheduler


def _kernel(wrapper, tmp_path: Path) -> ShrikeKernel:
    index = VectorIndex(path=tmp_path / "index")
    derived = DerivedTextStore(path=tmp_path / "shrike.db")
    runtime = EmbeddingRuntime(index=index)
    return ShrikeKernel(
        wrapper=wrapper,
        index=index,
        saver=IndexSaver(index),
        derived=derived,
        runtime=runtime,
        scheduler=WorkerScheduler(wrapper),
    )


class TestSchedulerPort:
    def test_worker_scheduler_satisfies_protocol(self, wrapper) -> None:
        assert isinstance(WorkerScheduler(wrapper), Scheduler)

    def test_run_on_collection_runs_on_the_worker_thread(self, wrapper) -> None:
        scheduler = WorkerScheduler(wrapper)
        seen: dict[str, str] = {}

        def probe(col) -> int:
            seen["thread"] = threading.current_thread().name
            return len(col.find_notes("deck:*"))

        count = scheduler.run_on_collection(probe)
        assert isinstance(count, int)
        assert seen["thread"].startswith("shrike-collection")

    def test_spawn_compute_runs_in_background(self, wrapper) -> None:
        scheduler = WorkerScheduler(wrapper)
        done = threading.Event()
        scheduler.spawn_compute("test-compute", done.set)
        assert done.wait(timeout=5)

    def test_call_later_fires_and_cancels(self, wrapper) -> None:
        scheduler = WorkerScheduler(wrapper)
        fired = threading.Event()
        scheduler.call_later(0.01, fired.set)
        assert fired.wait(timeout=5)

        never = threading.Event()
        handle = scheduler.call_later(5.0, never.set)
        handle.cancel()
        time.sleep(0.05)
        assert not never.is_set()


class TestKernelVerbs:
    def test_status_block_shape(self, wrapper, tmp_path: Path) -> None:
        kernel = _kernel(wrapper, tmp_path)
        status = kernel.status()
        assert status["locking"] == "permanent"
        assert status["collection_held"] is True
        assert status["embedding"]["available"] is False
        assert status["index"]["state"] == "unavailable"
        assert "state" in status["derived"]

    def test_rebuild_without_embedder_is_a_config_error(self, wrapper, tmp_path: Path) -> None:
        kernel = _kernel(wrapper, tmp_path)
        try:
            kernel.rebuild_index()
        except KernelConfigError as e:
            assert "not running" in str(e)
        else:
            raise AssertionError("expected KernelConfigError")

    def test_save_index_empty(self, wrapper, tmp_path: Path) -> None:
        kernel = _kernel(wrapper, tmp_path)
        assert kernel.save_index() == {"status": "empty"}

    def test_stop_embedding_when_not_running(self, wrapper, tmp_path: Path) -> None:
        kernel = _kernel(wrapper, tmp_path)
        assert kernel.stop_embedding() == {"status": "not_running"}

    def test_reload_reports_col_mod(self, wrapper, tmp_path: Path) -> None:
        kernel = _kernel(wrapper, tmp_path)
        out = kernel.reload()
        assert out["status"] == "reloaded"
        assert isinstance(out["col_mod"], int)
        assert out["rebuilding"] is False

    def test_close_tears_down_core(self, wrapper, tmp_path: Path) -> None:
        kernel = _kernel(wrapper, tmp_path)
        kernel.runtime = MagicMock()
        kernel.close()
        kernel.runtime.stop.assert_called_once()
        assert wrapper._closed is True
