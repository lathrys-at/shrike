"""The debounced trigram-DF refresh wired into the kernel write paths.

The fuzzy prune ranks on a materialized ``trigram_df`` snapshot that is built at
rebuild and otherwise left behind by incremental writes. The kernel closes that
gap by poking a re-arming debounce (``DerivedSnapshotRefresher``) on every
derived change — a field write/removal off ``process_batch`` and a recognition
write off ``store_recognition`` — so the snapshot re-materializes once the write
stream settles.

These tests exercise that wiring through the assembled ``AsyncKernel``: a write
lands a derived change, the snapshot still lags after the ingest queue drains
(``settle`` does NOT cover the separate refresh job), and the novel/removed
trigram appears/disappears in ``trigram_df`` once the debounce fires. The
observable is the snapshot CONTENT, read directly from the sidecar — the fuzzy
*query* is recall-safe regardless of the snapshot (the prune keeps absent/DF-0
trigrams, so a novel-trigram note surfaces whether or not the snapshot has caught
up — pinned by ``shrike_derived``'s
``fuzzy_finds_a_match_via_just_written_trigrams_under_a_stale_snapshot``). The
snapshot lag is a RANKING drift, not a recall flip, so only the snapshot content —
not query recall — can witness the refresh.

The settle is the snapshot itself: poll ``trigram_df`` until the trigram crosses,
unbounded — a dropped poke never crosses and Bazel's per-test timeout catches the
hang, never a wall-clock budget that a starved scheduler could lose.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

shrike_native = pytest.importorskip("shrike_native")

pytestmark = pytest.mark.skipif(
    not hasattr(shrike_native, "async_kernel_open"),
    reason="anki-core build required (scripts/build-native.sh)",
)


class _Backend:
    """Deterministic unit vectors + the EmbedderBackend metadata surface."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            b = hashlib.blake2b(text.encode(), digest_size=1).digest()[0] / 255.0
            n = (b * b + 1.0) ** 0.5
            out.append([b / n, 1.0 / n, 0.0, 0.0])
        return out

    def model_fingerprint(self) -> str:
        return "test-backend:v1"

    def embedding_dim(self) -> int:
        return 4


class _StubRecognizer:
    """The RecognizerBackend wire contract: blocking recognize() returning
    (text, confidence, segments_json) per item."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def recognize(self, items: list[bytes]) -> list[tuple[str, float, str]]:
        self.calls.append(len(items))
        out = []
        for data in items:
            text = data.decode("utf-8", errors="replace")
            segments = json.dumps(
                [{"text": text, "confidence": 0.92, "bbox": [0.0, 0.0, 1.0, 0.2]}]
            )
            out.append((text, 0.92, segments))
        return out

    def model_fingerprint(self) -> str:
        return "stub:v1"


class _DfSnapshot:
    """A read-only observer of the kernel's live ``trigram_df`` snapshot.

    ONE persistent read-only connection for the whole observation — the kernel's
    write connection owns the WAL ``-shm``, and a reader that re-opens per poll
    re-maps it under the writer, which surfaces transient ``SQLITE_PROTOCOL``
    ("locking protocol"). A single read-only connection (``mode=ro``: never
    creates the ``-shm`` or takes a write lock) shares the writer's ``-shm`` and
    just reads committed WAL frames. A read landing exactly on a refresh's commit
    can still take a transient lock error, so the read retries it out behind the
    busy timeout — the same wait budget the store's own pooled readers use."""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, timeout=5.0, check_same_thread=False
        )
        self._conn.execute("PRAGMA busy_timeout=5000")

    # A read landing on the refresh's commit takes one of these transient errors;
    # they clear on retry. Anything else (e.g. "no such table" from a schema
    # regression) is a real bug and must fail fast, not retry into a timeout.
    _TRANSIENT_LOCK = ("locking protocol", "database is locked", "database table is locked")

    def terms(self) -> dict[str, int]:
        """The current ``trigram_df``, retrying ONLY a transient lock error rather
        than failing the read — a snapshot poll must witness the eventual state,
        not a momentary contention with the refresh's commit. A non-lock
        ``OperationalError`` re-raises so a real regression fails fast and legibly
        instead of polling into a Bazel timeout."""
        while True:
            try:
                return dict(self._conn.execute("SELECT term, df FROM trigram_df").fetchall())
            except sqlite3.OperationalError as err:
                if not any(msg in str(err).lower() for msg in self._TRANSIENT_LOCK):
                    raise
                time.sleep(0.01)

    def has(self, term: str) -> bool:
        return term in self.terms()

    async def settle_until(self, crossed: Callable[[dict[str, int]], bool]) -> dict[str, int]:
        """Poll ``trigram_df`` until ``crossed`` holds — the structural "the
        refresh landed" event, not a wall-clock budget. Unbounded: a refresh that
        never fires (a dropped poke) hangs and Bazel's per-test timeout catches the
        regression, so pass/fail can't hinge on a budget-vs-load race."""
        terms = self.terms()
        while not crossed(terms):
            await asyncio.sleep(0.02)
            terms = self.terms()
        return terms

    def close(self) -> None:
        self._conn.close()


def _snapshot(tmp_path: Path) -> _DfSnapshot:
    from shrike.harness import cache_layout

    db_path = cache_layout.derived_db_path(
        str(tmp_path / "cache"), str(tmp_path / "collection.anki2")
    )
    return _DfSnapshot(db_path)


async def _open(tmp_path: Path) -> Any:
    kernel = await shrike_native.async_kernel_open(
        str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
    )
    kernel.attach_embedder(shrike_native.PyEmbedder.capture(_Backend()))
    return kernel


class TestDfRefreshWiring:
    """``process_batch``/``store_recognition`` poke the debounced refresh on a
    derived change, and the snapshot re-materializes once the stream settles."""

    def test_incremental_field_write_lands_a_novel_trigram(self, tmp_path: Path) -> None:
        """An incremental ``upsert_notes`` carrying a trigram the snapshot doesn't
        know appears in ``trigram_df`` only after the debounce — and is still
        ABSENT after ``settle`` drains the ingest queue, proving the refresh (not
        the ingest, not a rebuild) is what materializes it."""

        async def flow() -> None:
            kernel = await _open(tmp_path)
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            # Seed + rebuild_derived so trigram_df is materialized over a snapshot
            # that predates the novel write below.
            await kernel.upsert_notes([(basic, 1, ["alpha beta gamma", "delta"], [])], "error")
            await kernel.rebuild_derived()
            df = _snapshot(tmp_path)
            assert not df.has("zqw"), "novel trigram absent from the seeded snapshot"

            # Incremental write (NOT a rebuild) of a note carrying the novel "zqw".
            await kernel.upsert_notes([(basic, 1, ["zqwxyv", "back"], [])], "error")
            await kernel.settle()
            assert not df.has("zqw"), (
                "the ingest queue drained but the snapshot still lags — settle does "
                "not cover the separate refresh job"
            )

            terms = await df.settle_until(lambda terms: "zqw" in terms)
            assert terms["zqw"] == 1, "the refresh materialized the just-written trigram"
            df.close()
            await kernel.close()

        asyncio.run(flow())

    def test_rearming_write_stream_lands_the_whole_batch(self, tmp_path: Path) -> None:
        """A spaced stream of incremental writes re-arms the debounce; once the
        stream goes quiet a single materialization reflects EVERY novel trigram —
        the re-arming batch window, observed by the whole set crossing together.

        The observable is the post-quiet snapshot, not the refresh COUNT. The
        coalescing is structural: one ``Maintenance::request`` job re-arms by
        construction (covered by the maintenance-primitive tests), so the exact
        poke count is an implementation detail the coalescing makes irrelevant —
        one refresh lands whether the wiring pokes once or N times. What a kernel
        test can witness, and what a dropped poke would break, is that the whole
        settled batch is reflected — so that is what this asserts."""

        async def flow() -> None:
            kernel = await _open(tmp_path)
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            await kernel.upsert_notes([(basic, 1, ["seed text only", "back"], [])], "error")
            await kernel.rebuild_derived()
            df = _snapshot(tmp_path)

            novel = ["xqv", "wkj", "vbz", "qpf"]
            for gram in novel:
                await kernel.upsert_notes([(basic, 1, [f"{gram}mn one", "back"], [])], "error")
                await kernel.settle()
            # The whole batch's trigrams cross together once the stream settles —
            # the re-arming debounce collapsed the stream into a materialization
            # that sees every write, not a per-write snapshot mid-stream.
            terms = await df.settle_until(lambda terms: all(g in terms for g in novel))
            assert all(terms[g] == 1 for g in novel), "every re-armed write landed"
            df.close()
            await kernel.close()

        asyncio.run(flow())

    def test_removal_only_batch_pokes_the_refresh(self, tmp_path: Path) -> None:
        """A removal-only ``delete_notes`` is a derived change too
        (``!removals.is_empty()``), so it pokes the refresh: the removed note's
        trigram drops out of ``trigram_df`` after the debounce."""

        async def flow() -> None:
            kernel = await _open(tmp_path)
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            results = await kernel.upsert_notes(
                [
                    (basic, 1, ["wbq removable note", "back"], []),
                    (basic, 1, ["keepme stable text", "back"], []),
                ],
                "error",
            )
            removable = results[0][1]
            await kernel.rebuild_derived()
            df = _snapshot(tmp_path)
            assert df.has("wbq"), "the removable note's trigram is in the snapshot"

            deleted = json.loads(await kernel.delete_notes([removable]))
            assert deleted == {"deleted": [removable], "not_found": []}
            await kernel.settle()
            assert df.has("wbq"), "the snapshot still carries it before the refresh"

            terms = await df.settle_until(lambda terms: "wbq" not in terms)
            assert "kee" in terms, (
                "the surviving note's trigrams stay — only the removed note dropped"
            )
            df.close()
            await kernel.close()

        asyncio.run(flow())

    def test_recognition_only_write_pokes_the_refresh(self, tmp_path: Path) -> None:
        """A recognition sweep writes derived rows off ``store_recognition`` and
        never advances col.mod, so neither the field-write tail nor a drift
        rebuild would refresh the snapshot — the recognition-tail poke is what
        lands the OCR text's novel trigram in ``trigram_df``."""

        async def flow() -> None:
            kernel = await _open(tmp_path)
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            await kernel.upsert_notes(
                [(basic, 1, ['See <img src="cyc.png">', "back"], [])], "error"
            )
            await kernel.rebuild_derived()
            df = _snapshot(tmp_path)
            assert not df.has("jkq"), "the OCR-only trigram is not in the field snapshot"

            media = {"cyc.png": b"jkq ocr recognized phrase"}
            recognizer = _StubRecognizer()
            kernel.attach_recognizer(
                shrike_native.Recognizer.capture(recognizer),
                media.get,
                lambda name: name in media,
            )
            report = json.loads(await kernel.recognize_pending(10))
            assert report["status"] == "ran"
            assert report["stored"] == 1
            await kernel.settle()
            assert not df.has("jkq"), "the snapshot lags the recognition write"

            terms = await df.settle_until(lambda terms: "jkq" in terms)
            assert terms["jkq"] == 1, "the recognition-tail poke materialized the OCR trigram"
            df.close()
            await kernel.close()

        asyncio.run(flow())

    def test_removal_re_derives_a_shared_trigram_count(self, tmp_path: Path) -> None:
        """The refresh RE-DERIVES counts, not just inserts new keys. Two notes share
        a trigram (df=2); removing one must drop its count to 1 — the trigram stays
        PRESENT, only its document frequency falls. This needs the full
        DELETE-and-reINSERT-FROM-idx_vocab the refresh does; a refresh that merely
        added missing keys would leave the stale df=2. The count is load-bearing: the
        fuzzy prune sorts the rarest trigrams first BY df, so a stale count mis-orders
        the rarest set."""

        async def flow() -> None:
            kernel = await _open(tmp_path)
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            results = await kernel.upsert_notes(
                [
                    (basic, 1, ["zqwone alpha", "back"], []),
                    (basic, 1, ["zqwtwo beta", "back"], []),
                ],
                "error",
            )
            removable = results[0][1]
            await kernel.rebuild_derived()
            df = _snapshot(tmp_path)
            assert df.terms()["zqw"] == 2, "both notes contribute the shared trigram"

            deleted = json.loads(await kernel.delete_notes([removable]))
            assert deleted == {"deleted": [removable], "not_found": []}
            await kernel.settle()
            assert df.terms()["zqw"] == 2, "the count still lags before the refresh re-derives it"

            terms = await df.settle_until(lambda terms: terms.get("zqw") == 1)
            assert terms["zqw"] == 1, "the refresh re-derived the count down, not just kept the key"
            df.close()
            await kernel.close()

        asyncio.run(flow())
