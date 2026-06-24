"""The export_package action: delivery modes + the path-safety gate.

Drives the action directly off ``build_actions`` (the registry the MCP/HTTP
adapters bind) with a real kernel, a real ExportStore, and a temp export root —
exercising the two deliveries (download url / server-local path), the
purely-local + root gate on output_path, and the input-error cases. The
kernel-level export round-trip + symlink safety are pinned in
tests/native/test_export_package.py and the shrike-collection Rust tests.
"""

from __future__ import annotations

import os

import pytest

from shrike.api.actions import ActionContext, ToolInputError, build_actions
from shrike.server.export_store import ExportStore


def _export_action(kharness, **ctx_kwargs):
    """The export_package impl, wrapped so a direct call needn't pass the
    keyword-only `note_ids` (FactMCP fills its default_factory; a bare Python
    call must supply it). Tests call ``export(...)`` as the MCP layer would."""
    ctx = ActionContext(wrapper=kharness.wrapper, kernel=kharness.kernel, **ctx_kwargs)
    impl = {a.name: a.impl for a in build_actions(ctx)}["export_package"]

    async def export(**kwargs):
        kwargs.setdefault("note_ids", [])
        return await impl(**kwargs)

    return export


def _seed(kharness) -> None:
    kharness.seed_note("alpha", deck="Default")
    kharness.seed_note("beta", deck="Default")


class TestDownloadUrlDelivery:
    def test_default_returns_a_download_url(self, kharness, tmp_path) -> None:
        _seed(kharness)
        store = ExportStore(str(tmp_path / "cache"))
        export = _export_action(
            kharness, export_store=store, media_base_url="http://127.0.0.1:8372"
        )

        result = kharness.run(export())  # whole-collection .apkg, no output_path
        assert result.delivery == "url"
        assert result.note_count == 2
        assert result.format == "apkg"
        assert result.url.startswith("http://127.0.0.1:8372/export/")
        assert result.bytes > 0
        # The token claims a real on-disk temp file (the GET route serves it once).
        token = result.url.rsplit("/", 1)[1]
        path = store.claim(token)
        assert path is not None and os.path.isfile(path)

    def test_url_unavailable_without_store_or_base_url(self, kharness) -> None:
        # No export_store / no base url (direct library use, no HTTP server):
        # the download-url delivery can't be produced → a clean input error.
        export = _export_action(kharness)
        with pytest.raises(ToolInputError, match="download-url export is unavailable"):
            kharness.run(export())


class TestServerLocalPathDelivery:
    def test_output_path_in_root_on_purely_local_server(self, kharness, tmp_path) -> None:
        _seed(kharness)
        root = tmp_path / "exports"
        root.mkdir()
        export = _export_action(
            kharness,
            export_path_roots=[str(root)],
            server_purely_local=True,
        )

        out = root / "deck.apkg"
        result = kharness.run(export(output_path=str(out)))
        assert result.delivery == "path"
        assert result.path == str(out)
        assert result.note_count == 2
        assert out.is_file() and out.stat().st_size > 0

    def test_output_path_outside_root_is_rejected(self, kharness, tmp_path) -> None:
        root = tmp_path / "exports"
        root.mkdir()
        export = _export_action(kharness, export_path_roots=[str(root)], server_purely_local=True)
        out = tmp_path / "elsewhere.apkg"  # outside the root
        with pytest.raises(ToolInputError, match="output_path is not permitted"):
            kharness.run(export(output_path=str(out)))
        assert not out.exists()

    def test_output_path_rejected_when_not_purely_local(self, kharness, tmp_path) -> None:
        root = tmp_path / "exports"
        root.mkdir()
        # Root configured, but the server is NOT purely-local → output_path off.
        export = _export_action(kharness, export_path_roots=[str(root)], server_purely_local=False)
        out = root / "deck.apkg"
        with pytest.raises(ToolInputError, match="output_path is not permitted"):
            kharness.run(export(output_path=str(out)))
        assert not out.exists()

    def test_output_path_rejected_with_no_roots(self, kharness, tmp_path) -> None:
        # Purely-local but no --export-path-root → output_path stays off.
        export = _export_action(kharness, server_purely_local=True)
        out = tmp_path / "deck.apkg"
        with pytest.raises(ToolInputError, match="output_path is not permitted"):
            kharness.run(export(output_path=str(out)))


class TestValidation:
    def test_deck_and_note_ids_are_mutually_exclusive(self, kharness, tmp_path) -> None:
        export = _export_action(
            kharness, export_store=ExportStore(str(tmp_path / "c")), media_base_url="http://x"
        )
        with pytest.raises(ToolInputError, match="at most one of `deck` or `note_ids`"):
            kharness.run(export(deck="Default", note_ids=[1]))

    def test_colpkg_rejects_a_scope(self, kharness, tmp_path) -> None:
        export = _export_action(
            kharness, export_store=ExportStore(str(tmp_path / "c")), media_base_url="http://x"
        )
        with pytest.raises(ToolInputError, match="cannot be scoped"):
            kharness.run(export(format="colpkg", deck="Default"))

    def test_bad_format_is_rejected(self, kharness, tmp_path) -> None:
        export = _export_action(
            kharness, export_store=ExportStore(str(tmp_path / "c")), media_base_url="http://x"
        )
        with pytest.raises(ToolInputError, match="must be 'apkg' or 'colpkg'"):
            kharness.run(export(format="zip"))
