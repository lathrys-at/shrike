from __future__ import annotations

import base64

import pytest
import shrike_native
from pydantic import ValidationError

from shrike.schemas import (
    CollectionPruneResponse,
    FetchMediaResponse,
    StoreMediaItem,
    StoreMediaResponse,
)
from tests.unit.conftest import make_notes

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n-fake-image-bytes").decode("ascii")


class TestStoreMedia:
    async def test_store_data(self, wrapper):
        results = await wrapper.store_media(
            [{"data": PNG, "filename": "cell.png"}], allow_private_fetch=False
        )
        # validates against the wire schema
        resp = StoreMediaResponse.model_validate({"results": results})
        assert resp.results[0].status == "stored"
        assert resp.results[0].filename == "cell.png"
        assert resp.results[0].mime == "image/png"
        assert resp.results[0].deduped is False

    async def test_bad_base64_is_per_item_error(self, wrapper):
        results = await wrapper.store_media(
            [
                {"data": PNG, "filename": "ok.png"},
                {"data": "!!not-base64!!", "filename": "bad.png"},
            ],
            allow_private_fetch=False,
        )
        assert [r["status"] for r in results] == ["stored", "error"]
        assert results[1]["index"] == 1
        assert results[1]["filename"] == "bad.png"

    async def test_identical_content_dedupes(self, wrapper):
        await wrapper.store_media([{"data": PNG, "filename": "a.png"}], allow_private_fetch=False)
        again = await wrapper.store_media(
            [{"data": PNG, "filename": "a.png"}], allow_private_fetch=False
        )
        assert again[0]["deduped"] is True
        assert again[0]["filename"] == "a.png"

    async def test_collision_renames(self, wrapper):
        await wrapper.store_media([{"data": PNG, "filename": "a.png"}], allow_private_fetch=False)
        other = base64.b64encode(b"totally different bytes").decode("ascii")
        clash = await wrapper.store_media(
            [{"data": other, "filename": "a.png"}], allow_private_fetch=False
        )
        assert clash[0]["status"] == "stored"
        assert clash[0]["filename"] != "a.png"
        assert clash[0]["deduped"] is False

    async def test_store_url_derives_name_and_extension(self, wrapper, monkeypatch):
        def fake_fetch(url, allow_private=False):
            assert allow_private is False
            return b"downloaded-bytes", "image/png"

        monkeypatch.setattr(shrike_native, "fetch_media_url", fake_fetch)
        # no filename and a URL path without an extension -> derived from Content-Type
        results = await wrapper.store_media(
            [{"url": "https://example.com/asset"}], allow_private_fetch=False
        )
        assert results[0]["status"] == "stored"
        assert results[0]["filename"].endswith(".png")

    async def test_store_url_failure_is_per_item_error(self, wrapper, monkeypatch):
        def boom(url, allow_private=False):
            raise ValueError("refusing to fetch from non-public address 10.0.0.1")

        monkeypatch.setattr(shrike_native, "fetch_media_url", boom)
        results = await wrapper.store_media(
            [{"url": "http://10.0.0.1/x.png"}], allow_private_fetch=False
        )
        assert results[0]["status"] == "error"
        assert "non-public" in results[0]["error"]


class TestStoreMediaServerPath:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"path": "/x.png"},  # path alone is valid
            {"url": "http://h/x.png"},
            {"data": PNG, "filename": "x.png"},
        ],
    )
    def test_single_source_accepted(self, kwargs):
        StoreMediaItem(**kwargs)  # no raise

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"data": PNG, "filename": "x.png", "path": "/x.png"},  # two sources
            {"url": "http://h/x", "path": "/x.png"},
            {},  # zero sources
        ],
    )
    def test_multi_or_zero_source_rejected(self, kwargs):
        with pytest.raises(ValidationError, match="exactly one"):
            StoreMediaItem(**kwargs)

    async def test_path_refused_when_no_roots_even_if_otherwise_local(self, wrapper, tmp_path):
        # Off by default (#170): empty server_path_roots → `path` refused outright.
        src = tmp_path / "local.png"
        src.write_bytes(b"x")
        results = await wrapper.store_media([{"path": str(src)}], allow_private_fetch=False)
        assert results[0]["status"] == "error"
        assert "not enabled" in results[0]["error"]
        assert (await wrapper.list_media(pattern="local.png", limit=None))["count"] == 0

    async def test_path_stored_when_within_root(self, wrapper, tmp_path):
        root = tmp_path / "media-root"
        root.mkdir()
        src = root / "local.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\nserver-local")
        results = await wrapper.store_media(
            [{"path": str(src)}], allow_private_fetch=False, server_path_roots=[str(root)]
        )
        resp = StoreMediaResponse.model_validate({"results": results})
        assert resp.results[0].status == "stored"
        assert resp.results[0].filename == "local.png"  # name from the path basename
        assert resp.results[0].mime == "image/png"
        assert resp.results[0].size_bytes == len(b"\x89PNG\r\n\x1a\nserver-local")
        assert (await wrapper.list_media(pattern="local.png", limit=None))["count"] == 1

    async def test_multiple_roots_are_a_disjunction(self, wrapper, tmp_path):
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        (root_a / "x.png").write_bytes(b"a")
        (root_b / "y.png").write_bytes(b"b")
        outside = tmp_path / "z.png"
        outside.write_bytes(b"z")
        roots = [str(root_a), str(root_b)]

        async def status(p):
            r = await wrapper.store_media(
                [{"path": str(p)}], allow_private_fetch=False, server_path_roots=roots
            )
            return r[0]["status"]

        assert await status(root_a / "x.png") == "stored"  # under root A
        assert await status(root_b / "y.png") == "stored"  # under root B
        assert await status(outside) == "error"  # under neither

    async def test_missing_path_within_root_is_per_item_error(self, wrapper, tmp_path):
        root = tmp_path / "media-root"
        root.mkdir()
        results = await wrapper.store_media(
            [{"path": str(root / "nope.png")}],
            allow_private_fetch=False,
            server_path_roots=[str(root)],
        )
        assert results[0]["status"] == "error"
        assert "not found" in results[0]["error"]

    async def test_root_confines_against_traversal_and_symlink_escape(self, wrapper, tmp_path):
        import os

        root = tmp_path / "allowed"
        root.mkdir()
        inside = root / "ok.png"
        inside.write_bytes(b"in")
        outside = tmp_path / "secret.png"
        outside.write_bytes(b"out")
        escape = root / "escape.png"  # symlink inside root → outside (add_file follows it)
        os.symlink(outside, escape)
        traversal = root / ".." / "secret.png"  # `..` escape

        async def store(p):
            r = await wrapper.store_media(
                [{"path": str(p)}], allow_private_fetch=False, server_path_roots=[str(root)]
            )
            return r[0]

        assert (await store(inside))["status"] == "stored"
        for bad in (outside, escape, traversal):
            res = await store(bad)
            assert res["status"] == "error" and "outside the configured media root" in res["error"]


class TestFetchMedia:
    async def test_found_reports_path_never_bytes(self, wrapper):
        # fetch never returns bytes — only where they live (path; url added by the
        # tool layer). The model/CLI fetches from there.
        await wrapper.store_media([{"data": PNG, "filename": "a.png"}], allow_private_fetch=False)
        results = await wrapper.fetch_media(["a.png", "nope.png"])
        resp = FetchMediaResponse.model_validate({"results": results})
        assert resp.results[0].status == "found"
        assert "data" not in results[0]
        assert resp.results[0].path.endswith("a.png")
        assert resp.results[0].size_bytes > 0
        assert resp.results[1].status == "missing"

    async def test_path_traversal_is_missing(self, wrapper):
        results = await wrapper.fetch_media(["../../etc/passwd"])
        assert results[0]["status"] == "missing"


class TestListMedia:
    async def test_list_and_glob(self, wrapper):
        await wrapper.store_media(
            [
                {"data": PNG, "filename": "a.png"},
                {"data": PNG, "filename": "b.jpg"},
            ],
            allow_private_fetch=False,
        )
        allm = await wrapper.list_media(pattern=None, limit=None)
        assert allm["count"] == 2
        pngs = await wrapper.list_media(pattern="*.png", limit=None)
        assert [f["filename"] for f in pngs["files"]] == ["a.png"]

    async def test_limit_caps_files_not_count(self, wrapper):
        await wrapper.store_media(
            [{"data": PNG, "filename": f"f{i}.png"} for i in range(3)],
            allow_private_fetch=False,
        )
        listed = await wrapper.list_media(pattern=None, limit=2)
        assert listed["count"] == 3
        assert len(listed["files"]) == 2


class TestDeleteMedia:
    async def test_trash_and_not_found(self, wrapper):
        await wrapper.store_media([{"data": PNG, "filename": "a.png"}], allow_private_fetch=False)
        result = await wrapper.delete_media(["a.png", "ghost.png"])
        assert result["deleted"] == ["a.png"]
        assert result["not_found"] == ["ghost.png"]
        assert (await wrapper.list_media(pattern="a.png", limit=None))["count"] == 0


class TestMediaCheck:
    async def test_unused_and_missing(self, wrapper):
        # an unreferenced file -> unused; a note referencing an absent file -> missing
        await wrapper.store_media(
            [{"data": PNG, "filename": "orphan.png"}], allow_private_fetch=False
        )
        make_notes(
            wrapper,
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": '<img src="ghost.png">', "Back": "x"},
                }
            ],
        )
        check = await wrapper.media_check()
        assert "orphan.png" in check["unused"]
        assert "ghost.png" in check["missing"]
        assert check["missing_media_notes"]


class TestPruneUnusedMedia:
    async def test_dry_run_then_apply(self, wrapper):
        await wrapper.store_media(
            [{"data": PNG, "filename": "orphan.png"}], allow_private_fetch=False
        )
        preview, removed = await wrapper.prune(
            unused_tags=False,
            empty_notes=False,
            empty_cards=False,
            unused_media=True,
            dry_run=True,
        )
        resp = CollectionPruneResponse.model_validate(preview)
        assert resp.unused_media is not None
        assert "orphan.png" in resp.unused_media.files
        assert resp.unused_tags is None  # not requested
        assert removed == []  # media removal isn't an index concern
        assert (await wrapper.list_media(pattern="orphan.png", limit=None))["count"] == 1

        applied, _ = await wrapper.prune(
            unused_tags=False,
            empty_notes=False,
            empty_cards=False,
            unused_media=True,
            dry_run=False,
        )
        assert applied["unused_media"]["removed"] == 1
        assert (await wrapper.list_media(pattern="orphan.png", limit=None))["count"] == 0
