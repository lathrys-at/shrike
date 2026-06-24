from __future__ import annotations

import base64

import pytest
import shrike_native
from pydantic import ValidationError

from shrike.schemas import (
    StoreMediaItem,
    StoreMediaResponse,
)

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
        # Off by default: empty server_path_roots → `path` refused outright.
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
