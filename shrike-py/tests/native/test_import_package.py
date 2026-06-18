"""Import an .apkg into the collection — the kernel op + the DRIFT-REBUILD path.

The load-bearing requirement: an import bumps col.mod, so the index MUST
reconcile the imported notes afterward (the boot/reload drift path), and the
op must NOT advance the watermark prematurely (which would suppress the drift
signal). These tests build a REAL .apkg via the anki pip test oracle, import it
through the kernel op, and assert both the per-bucket summary AND that the
index reconciled the imported notes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

shrike_native = pytest.importorskip("shrike_native")

pytestmark = pytest.mark.skipif(
    not hasattr(shrike_native.AsyncKernel, "import_package"),
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


def _build_apkg(tmp_path, fronts: list[str]) -> str:
    """Create a real .apkg via the anki pip test oracle: a temp collection with
    one Basic note per `fronts`, exported as an Anki package. Returns its path."""
    import anki.import_export_pb2 as ie
    from anki.collection import Collection

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    col = Collection(str(src_dir / "collection.anki2"))
    try:
        basic = col.models.by_name("Basic")
        deck_id = col.decks.id("Imported")
        for front in fronts:
            note = col.new_note(basic)
            note["Front"] = front
            note["Back"] = f"answer to {front}"
            col.add_note(note, deck_id)
        out_path = str(tmp_path / "deck.apkg")
        # A modern .apkg (legacy=False); limit=None = whole collection.
        options = ie.ExportAnkiPackageOptions(
            with_scheduling=False, with_deck_configs=False, with_media=True, legacy=False
        )
        col.export_anki_package(out_path=out_path, options=options, limit=None)
    finally:
        col.close()
    return out_path


async def _open_kernel_with_index(tmp_path, backend):
    """An empty collection with the embedder attached and an (empty) index
    built — the steady state an import then drifts."""
    kernel = await shrike_native.async_kernel_open(
        str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
    )
    kernel.attach_embedder(shrike_native.PyEmbedder.capture(backend))
    # Materialize the empty index so its watermark is current (no drift yet).
    await kernel.reindex_if_needed()
    assert not await kernel.reindex_if_needed(), "freshly built index must not drift"
    return kernel


def _lexical_hits(tmp_path, query: str) -> list:
    """Substring/lexical hits from the per-collection derived (FTS5) store — a
    fresh engine on the same namespaced shrike.db the kernel's import rebuild
    committed to. This is the surface an import's derived rebuild must populate,
    rather than leaving empty until an unrelated drift trigger."""
    from shrike.harness import cache_layout

    db_path = cache_layout.derived_db_path(
        str(tmp_path / "cache"), str(tmp_path / "collection.anki2")
    )
    engine = shrike_native.DerivedTextEngine(db_path, 2)
    try:
        return engine.search_substring(query, 10)
    finally:
        engine.close()


class TestImportDriftRebuild:
    def test_import_reconciles_the_imported_notes(self, tmp_path) -> None:
        """The core import path: import → BOTH derived caches reconcile the new
        notes (vector index AND the FTS5 lexical store), and the watermark is
        left current (no spurious drift afterward)."""

        async def flow():
            apkg = _build_apkg(tmp_path, ["alpha", "beta", "gamma"])
            kernel = await _open_kernel_with_index(tmp_path, _Backend())
            engine = kernel.engine_handle()
            assert engine.size() == 0  # empty before import

            summary_json, reindexed = await kernel.import_package(
                apkg, "if_newer", "if_newer", False, False
            )
            summary = json.loads(summary_json)

            # The three imported notes are reported NEW.
            assert summary["new"] == 3
            assert summary["found_notes"] == 3
            assert summary["updated"] == 0
            # The drift path fired and reconciled the VECTOR index.
            assert reindexed is True, "import must trigger an index reconcile"
            assert engine.size() == 3, "the imported notes must be in the index"
            # Watermark current: the reconcile advanced it, so no further drift.
            assert not await kernel.reindex_if_needed(), (
                "after the import reconcile the index must be current "
                "(the op must not leave stale drift, nor suppress it prematurely)"
            )
            await kernel.close()

            # And the DERIVED (FTS5) store was rebuilt too — an imported note is
            # findable by LEXICAL/substring search WITHOUT any intervening
            # reload (reindex_if_needed only touches the vector index; import
            # must also drive rebuild_derived).
            assert _lexical_hits(tmp_path, "alpha"), (
                "imported note must be lexically findable right after import — "
                "the derived store must be rebuilt by the import op, not left "
                "stale until an unrelated drift trigger"
            )
            assert _lexical_hits(tmp_path, "gamma")

        asyncio.run(flow())

    def test_import_without_embedder_still_imports(self, tmp_path) -> None:
        """No embedder attached: the import still mutates the collection; the
        vector reconcile is a no-op (reindexed False), but the DERIVED store is
        still rebuilt — lexical search finds the imported notes (no embedding
        required for the lexical surface)."""

        async def flow():
            apkg = _build_apkg(tmp_path, ["one", "two"])
            kernel = await shrike_native.async_kernel_open(
                str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
            )
            summary_json, reindexed = await kernel.import_package(
                apkg, "if_newer", "if_newer", False, False
            )
            summary = json.loads(summary_json)
            assert summary["new"] == 2
            assert reindexed is False  # no embedder → nothing to reindex
            # The notes really landed in the collection.
            ids = kernel.core_handle().find_notes("")
            assert len(ids) == 2
            await kernel.close()

            # Lexical search works WITHOUT any embedder (the derived rebuild ran
            # regardless of the vector index).
            assert _lexical_hits(tmp_path, "one"), "imported note must be lexically findable"

        asyncio.run(flow())

    def test_reimport_same_package_updates_not_duplicates(self, tmp_path) -> None:
        """Re-importing the same package (same GUIDs) does not duplicate: the
        notes are recognized as same-GUID (the conflict path runs)."""

        async def flow():
            apkg = _build_apkg(tmp_path, ["x", "y"])
            kernel = await _open_kernel_with_index(tmp_path, _Backend())
            first_json, _ = await kernel.import_package(apkg, "if_newer", "if_newer", False, False)
            assert json.loads(first_json)["new"] == 2
            engine = kernel.engine_handle()
            assert engine.size() == 2

            # Re-import the identical package: no new notes (same GUIDs), and
            # the index does not grow.
            second_json, _ = await kernel.import_package(apkg, "if_newer", "if_newer", False, False)
            second = json.loads(second_json)
            assert second["new"] == 0, "same-GUID notes must not re-add"
            assert engine.size() == 2, "re-import must not duplicate index entries"
            await kernel.close()

        asyncio.run(flow())


class TestImportInput:
    def test_bad_update_condition_is_input_error(self, tmp_path) -> None:
        async def flow():
            kernel = await shrike_native.async_kernel_open(
                str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
            )
            with pytest.raises(shrike_native.NativeInputError, match="if_newer"):
                await kernel.import_package("/x.apkg", "bogus", "if_newer", False, False)
            await kernel.close()

        asyncio.run(flow())

    def test_missing_package_errors(self, tmp_path) -> None:
        async def flow():
            kernel = await shrike_native.async_kernel_open(
                str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
            )
            # A missing/unreadable package surfaces anki's import error.
            with pytest.raises(shrike_native.NativeInternalError):
                await kernel.import_package(
                    str(tmp_path / "nope.apkg"), "if_newer", "if_newer", False, False
                )
            await kernel.close()

        asyncio.run(flow())
