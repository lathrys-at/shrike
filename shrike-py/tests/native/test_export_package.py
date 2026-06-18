"""Export to an Anki package — the kernel export op.

Drives ``AsyncKernel.export_package`` against a real collection: a whole-
collection ``.apkg`` round-trips (the package is a valid zip carrying the
collection db), a deck-scoped export limits the notes, ``.colpkg`` is whole-
collection-only (a scope is rejected), and a bad format/scope is a clean input
error. The path-safety GATE is host-side (tested in tests/unit/test_pathsafety
and the action tests); here the kernel trusts a caller-gated out_path.
"""

from __future__ import annotations

import asyncio
import json
import zipfile

import pytest

shrike_native = pytest.importorskip("shrike_native")

pytestmark = pytest.mark.skipif(
    not hasattr(shrike_native, "async_kernel_open"),
    reason="anki-core build required (scripts/build-native.sh)",
)


async def _kernel(tmp_path):
    return await shrike_native.async_kernel_open(
        str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
    )


def _apkg_names(path) -> set[str]:
    """The archive member names inside a .apkg/.colpkg (it's a zip)."""
    with zipfile.ZipFile(path) as z:
        return set(z.namelist())


def _has_collection_db(names: set[str]) -> bool:
    # The modern exporter writes collection.anki21b (zstd); legacy writes
    # collection.anki2/.anki21. Accept any so the assertion isn't format-brittle.
    return any(n.startswith("collection.anki2") for n in names)


class TestApkgExport:
    def test_whole_collection_apkg_round_trips(self, tmp_path) -> None:
        async def flow():
            kernel = await _kernel(tmp_path)
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            await kernel.upsert_notes(
                [
                    (basic, 1, ["alpha front", "alpha back"], []),
                    (basic, 1, ["beta front", "beta back"], []),
                ],
                "allow",
            )
            out = tmp_path / "whole.apkg"
            result = json.loads(await kernel.export_package(str(out), "apkg", "whole"))
            assert result["note_count"] == 2
            assert result["out_path"] == str(out)
            assert out.is_file() and out.stat().st_size > 0
            # A valid zip carrying the collection db (+ the media map).
            names = _apkg_names(out)
            assert _has_collection_db(names), names
            assert "media" in names  # the media manifest is always present
            await kernel.close()

        asyncio.run(flow())

    def test_deck_scoped_export_limits_notes(self, tmp_path) -> None:
        async def flow():
            kernel = await _kernel(tmp_path)
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            # A second deck; note its id for the scoped notes.
            decks = json.loads(await kernel.upsert_decks(json.dumps([{"name": "Spanish"}])))
            spanish_id = decks[0]["id"]
            await kernel.upsert_notes(
                [
                    (basic, 1, ["in default", "x"], []),
                    (basic, spanish_id, ["hola", "hello"], []),
                    (basic, spanish_id, ["gato", "cat"], []),
                ],
                "allow",
            )
            # Scope to the Spanish deck by name (the deck-ref convention).
            out = tmp_path / "spanish.apkg"
            result = json.loads(await kernel.export_package(str(out), "apkg", "deck", "Spanish"))
            # Only the two Spanish notes — the Default note is excluded.
            assert result["note_count"] == 2, result
            assert out.is_file()
            await kernel.close()

        asyncio.run(flow())

    def test_note_scoped_export(self, tmp_path) -> None:
        async def flow():
            kernel = await _kernel(tmp_path)
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            res = await kernel.upsert_notes(
                [
                    (basic, 1, ["one", "1"], []),
                    (basic, 1, ["two", "2"], []),
                    (basic, 1, ["three", "3"], []),
                ],
                "allow",
            )
            ids = [r[1] for r in res if r[0] == "created"]
            out = tmp_path / "two-notes.apkg"
            result = json.loads(
                await kernel.export_package(str(out), "apkg", "notes", None, ids[:2])
            )
            assert result["note_count"] == 2
            await kernel.close()

        asyncio.run(flow())


class TestColpkgExport:
    def test_whole_collection_colpkg(self, tmp_path) -> None:
        async def flow():
            kernel = await _kernel(tmp_path)
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            await kernel.upsert_notes([(basic, 1, ["only", "note"], [])], "allow")
            out = tmp_path / "backup.colpkg"
            result = json.loads(await kernel.export_package(str(out), "colpkg", "whole"))
            assert result["note_count"] == 1  # the whole collection's note total
            assert out.is_file()
            assert _has_collection_db(_apkg_names(out))
            await kernel.close()

        asyncio.run(flow())

    def test_colpkg_rejects_a_scope(self, tmp_path) -> None:
        async def flow():
            kernel = await _kernel(tmp_path)
            out = tmp_path / "scoped.colpkg"
            # A .colpkg is whole-collection only — a deck scope is an input error.
            with pytest.raises(Exception) as ei:
                await kernel.export_package(str(out), "colpkg", "deck", "Default")
            assert "whole-collection" in str(ei.value)
            assert not out.exists()  # nothing written on the rejected request
            await kernel.close()

        asyncio.run(flow())


class TestInputErrors:
    def test_unknown_format_is_input_error(self, tmp_path) -> None:
        async def flow():
            kernel = await _kernel(tmp_path)
            with pytest.raises(Exception, match="format must be apkg/colpkg"):
                await kernel.export_package(str(tmp_path / "x.zip"), "zip", "whole")
            await kernel.close()

        asyncio.run(flow())

    def test_unknown_deck_is_input_error(self, tmp_path) -> None:
        async def flow():
            kernel = await _kernel(tmp_path)
            out = tmp_path / "ghost.apkg"
            with pytest.raises(Exception, match="no deck"):
                await kernel.export_package(str(out), "apkg", "deck", "#999999")
            assert not out.exists()
            await kernel.close()

        asyncio.run(flow())

    def test_empty_note_scope_is_input_error(self, tmp_path) -> None:
        async def flow():
            kernel = await _kernel(tmp_path)
            with pytest.raises(Exception, match="at least one note id"):
                await kernel.export_package(str(tmp_path / "x.apkg"), "apkg", "notes", None, [])
            await kernel.close()

        asyncio.run(flow())
