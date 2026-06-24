"""Harness multi-space fan-out and secondary-floor recalibration."""

from __future__ import annotations

import asyncio

import pytest

shrike_native = pytest.importorskip("shrike_native")

from shrike.harness.derived import DerivedTextStore, NativeDerivedEngine  # noqa: E402
from shrike.harness.engines.embedding.runtime import EmbeddingRuntime  # noqa: E402
from shrike.harness.harness import Harness  # noqa: E402


class _FakeEmbedBackend:
    """A captured embedder backend over the wire contract (`embed_texts` /
    `model_fingerprint` / `embedding_dim`) — distinct fingerprints make two
    distinct kernel embed spaces."""

    def __init__(self, fingerprint: str, dim: int = 16) -> None:
        self._fp = fingerprint
        self._dim = dim
        self.running = True

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (self._dim - 1) for _ in texts]

    def model_fingerprint(self) -> str:
        return self._fp

    def embedding_dim(self) -> int:
        return self._dim

    @property
    def modalities(self):
        return frozenset({"text"})

    def stop(self) -> None:
        self.running = False


class _FakeRuntime:
    """A minimal stand-in for EmbeddingRuntime exercising the harness fan-out
    contract (`running` / `start` / `backend` / `backend_kind` / `stop`). A
    `fail=True` runtime raises on start — proving a failing space degrades only
    itself."""

    def __init__(self, fingerprint: str, *, fail: bool = False) -> None:
        self._fp = fingerprint
        self._fail = fail
        self.backend_kind = "onnx"
        self.backend = None

    @property
    def running(self) -> bool:
        return self.backend is not None and self.backend.running

    def start(self, **_overrides):
        if self._fail:
            raise RuntimeError(f"space {self._fp} cannot start")
        self.backend = _FakeEmbedBackend(self._fp)
        return self.backend

    def stop(self) -> None:
        if self.backend is not None:
            self.backend.stop()
            self.backend = None


class TestMultiSpaceFanOut:
    def test_two_spaces_attach_and_a_failing_one_leaves_the_other_live(self, tmp_path) -> None:
        # The harness fan-out: the primary plus one secondary space both
        # attach → the kernel holds TWO embed spaces. A SECOND secondary that
        # fails to start degrades only its own space — the others stay live.
        async def flow():
            runtime = EmbeddingRuntime(model=None)
            derived = DerivedTextStore(
                path=tmp_path / "cache" / "shrike.db", engine_factory=NativeDerivedEngine
            )
            harness = await Harness.assemble(
                collection_path=str(tmp_path / "collection.anki2"),
                cache_dir=str(tmp_path / "cache"),
                runtime=runtime,
                derived=derived,
                cooperative=False,
                hold_seconds=5.0,
                media_read=None,
                media_exists=None,
                secondary_runtimes=[
                    _FakeRuntime("space:b"),
                    _FakeRuntime("space:c", fail=True),  # degrades only itself
                ],
            )
            await harness.boot(start_embedding=False)
            assert harness.kernel.embed_space_count() == 0

            # Attach the primary (real onnx-free PyEmbedder) + fan out.
            primary = _FakeEmbedBackend("space:a")
            harness._attach(primary)  # primary attaches unkeyed (N=1 path)
            assert harness.kernel.embed_space_count() == 1
            await harness._attach_secondaries({})

            # space:a (primary) + space:b (secondary) are live; space:c failed.
            assert harness.kernel.embed_space_count() == 2
            assert harness.secondary_runtimes[0].running  # b started
            assert not harness.secondary_runtimes[1].running  # c failed

            # A second fan-out is idempotent for an already-running secondary.
            await harness._attach_secondaries({})
            assert harness.kernel.embed_space_count() == 2

            # stop_embedding clears every space and stops the secondaries.
            harness.kernel.detach_embedder()
            for rt in harness.secondary_runtimes:
                rt.stop()
            assert harness.kernel.embed_space_count() == 0

            await harness.close()

        asyncio.run(flow())


class _RecalibKernel:
    """A minimal kernel stand-in counting secondary-floor recalibrations and
    forcing a reconcile (reindex_if_needed -> True)."""

    def __init__(self) -> None:
        self.recalibrated = 0

    async def reindex_if_needed(self) -> bool:
        return True

    async def settle(self) -> None:
        return None

    async def calibrate_secondary_floors(self, margin: float) -> list[tuple[str, float | None]]:
        self.recalibrated += 1
        return []


class _RecalibWrapper:
    cooperative = False

    async def reopen(self) -> None:
        return None

    async def col_mod(self) -> int:
        return 123


class _RecalibRuntime:
    backend = object()


def _make_recalib_harness() -> Harness:
    h = Harness.__new__(Harness)
    h.kernel = _RecalibKernel()  # type: ignore[assignment]
    h.wrapper = _RecalibWrapper()  # type: ignore[assignment]
    h.runtime = _RecalibRuntime()  # type: ignore[assignment]
    h.secondary_runtimes = []
    h.cross_space_floor_margin = 1.0
    h._bg_tasks = set()
    h._ready = asyncio.Event()
    h._generation = 0

    async def _noop_build() -> None:
        return None

    h._maybe_build_derived = _noop_build  # type: ignore[assignment]
    return h


class TestReloadRecalibratesSecondaryFloors:
    """reload() must recalibrate the secondary cross-space image floor
    after a reconcile, like every other reindex path
    (_drive_reindex/_rebuild_then_calibrate/_drive_boot_reindex). Otherwise the
    floor stays computed against pre-reload vectors and mis-gates the
    image space until the next boot/rebuild/cooperative-reacquire. N>=2 only."""

    def test_drive_boot_reindex_recalibrates(self) -> None:  # control
        h = _make_recalib_harness()
        asyncio.run(h._drive_boot_reindex())
        assert h.kernel.recalibrated == 1

    def test_reload_recalibrates_after_reindex(self) -> None:
        h = _make_recalib_harness()
        out = asyncio.run(h.reload())
        assert out["rebuilding"] is True
        assert h.kernel.recalibrated == 1, (
            "reload() reconciled drift but never recalibrated the secondary "
            "cross-space image floor (stale floor mis-gates the image space)"
        )
