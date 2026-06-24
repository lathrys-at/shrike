"""The facade-readiness boot window: serving present derived rows while BUILDING."""

from __future__ import annotations

import asyncio

import pytest

shrike_native = pytest.importorskip("shrike_native")

from shrike.harness.derived import DerivedTextStore, NativeDerivedEngine  # noqa: E402
from shrike.harness.engines.embedding.runtime import EmbeddingRuntime  # noqa: E402
from shrike.harness.harness import Harness  # noqa: E402

from .conftest import _StubAsr  # noqa: E402


class TestFacadeReadinessBootWindow:
    """The facade-readiness race: a search during the boot rebuild window.

    `boot()` spawns the host ``DerivedTextStore`` rebuild fire-and-forget
    (``_maybe_build_derived`` -> ``_spawn_bg(_rebuild_derived())`` — never
    awaited), so the facade sits at ``state=BUILDING`` until that un-awaited
    task's ``await kernel.rebuild_derived()`` continuation runs
    ``settle_external_build`` -> ``READY``. While ``BUILDING``,
    ``DerivedTextStore.search_substring`` must not short-circuit to ``None``
    (gated on ``available == _state==READY``) when the recognition rows it would
    return are already present in shrike.db, written kernel-side by
    ``recognition_sweep`` — otherwise a read descheduled past the un-awaited
    settle returns ``None``.

    This pins the PRODUCT invariant: the boot-rebuild ``BUILDING`` window MUST
    NOT silently drop already-present derived rows. The same gap means a real
    ``search_notes`` substring/fuzzy query during the boot build-window
    transiently misses OCR/ASR rows (a silent field-fall-back).

    Deterministic: the ``BUILDING`` window is entered with the real
    ``claim_external_build()`` primitive ``_rebuild_derived`` itself uses,
    instead of racing for it under load.
    """

    def test_search_substring_serves_present_rows_during_building(self, tmp_path) -> None:
        import hashlib

        class _TokenHash:
            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                out = []
                for t in texts:
                    v = [0.0] * 64
                    for tok in t.lower().split():
                        h = int(hashlib.blake2b(tok.encode(), digest_size=2).hexdigest(), 16)
                        v[h % 64] += 1.0
                    n = sum(x * x for x in v) ** 0.5 or 1.0
                    out.append([x / n for x in v])
                return out

            def model_fingerprint(self) -> str:
                return "tok:v1"

            def embedding_dim(self) -> int:
                return 64

        async def flow():
            media = {"lecture.mp3": b"mitochondria are the powerhouse of the cell"}
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
                media_read=media.get,
                media_exists=lambda name: name in media,
            )
            await harness.boot(start_embedding=False)
            harness.kernel.attach_embedder(shrike_native.PyEmbedder.capture(_TokenHash()))
            await harness.kernel.reindex_if_needed()

            await harness.wrapper.upsert_notes(
                [
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {"Front": "Listen [sound:lecture.mp3]", "Back": "b"},
                    }
                ]
            )
            harness.attach_recognizer(_StubAsr(), "asr")
            report = await harness.recognition_sweep(batch_size=4)
            assert report["total_stored"] == 1, report

            # Settle the boot rebuild so we ISOLATE the BUILDING gate (not a boot
            # race) and establish the precondition: when READY, the kernel-written
            # ASR row is genuinely present and findable through the facade.
            await harness.settle_background()
            assert harness.derived.available, "precondition: facade READY after settle"
            assert harness.derived.search_substring("powerhouse of the cell", limit=5), (
                "precondition: the ASR row is present and findable when the facade is READY"
            )

            # Re-enter the exact state the un-awaited boot rebuild produces under
            # load: facade BUILDING, with the row already in shrike.db.
            assert harness.derived.claim_external_build(), "entered the BUILDING window"
            try:
                rows = harness.derived.search_substring("powerhouse of the cell", limit=5)
                # THE DEFECT: present rows are silently dropped while BUILDING.
                assert rows, (
                    "facade must serve already-present recognition rows during the boot "
                    "BUILDING window, not silently field-fall-back to None (#650)"
                )
            finally:
                harness.derived.settle_external_build(harness.derived.col_mod or 0)
                await harness.close()

        asyncio.run(flow())
