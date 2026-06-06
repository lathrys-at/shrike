from __future__ import annotations

import base64

import pytest

from shrike import collection as collection_mod
from shrike.collection import _check_public_address, _safe_media_name
from shrike.schemas import CollectionPruneResponse, FetchMediaResponse, StoreMediaResponse

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
        def fake_fetch(url, *, allow_private, **kwargs):
            assert allow_private is False
            return b"downloaded-bytes", "image/png"

        monkeypatch.setattr(collection_mod, "_fetch_media_url", fake_fetch)
        # no filename and a URL path without an extension -> derived from Content-Type
        results = await wrapper.store_media(
            [{"url": "https://example.com/asset"}], allow_private_fetch=False
        )
        assert results[0]["status"] == "stored"
        assert results[0]["filename"].endswith(".png")

    async def test_store_url_failure_is_per_item_error(self, wrapper, monkeypatch):
        def boom(url, *, allow_private, **kwargs):
            raise ValueError("refusing to fetch from non-public address 10.0.0.1")

        monkeypatch.setattr(collection_mod, "_fetch_media_url", boom)
        results = await wrapper.store_media(
            [{"url": "http://10.0.0.1/x.png"}], allow_private_fetch=False
        )
        assert results[0]["status"] == "error"
        assert "non-public" in results[0]["error"]


class TestFetchMedia:
    async def test_inline_when_opted_in(self, wrapper):
        await wrapper.store_media([{"data": PNG, "filename": "a.png"}], allow_private_fetch=False)
        results = await wrapper.fetch_media(["a.png", "nope.png"], max_inline_bytes=8 * 1024 * 1024)
        resp = FetchMediaResponse.model_validate({"results": results})
        assert resp.results[0].status == "inline"
        assert base64.b64decode(resp.results[0].data) == base64.b64decode(PNG)
        assert resp.results[1].status == "missing"

    async def test_default_returns_link_not_base64(self, wrapper):
        # max_inline_bytes=0 (the tool default) never inlines — base64 is opt-in.
        await wrapper.store_media([{"data": PNG, "filename": "a.png"}], allow_private_fetch=False)
        results = await wrapper.fetch_media(["a.png"], max_inline_bytes=0)
        assert results[0]["status"] == "link"
        assert "data" not in results[0]
        assert results[0]["path"].endswith("a.png")
        assert results[0]["size_bytes"] > 0

    async def test_path_traversal_is_missing(self, wrapper):
        results = await wrapper.fetch_media(["../../etc/passwd"], max_inline_bytes=8 * 1024 * 1024)
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
        wrapper.run_sync(
            lambda _c: wrapper._upsert_notes(
                [
                    {
                        "deck": "Test",
                        "note_type": "Basic",
                        "fields": {"Front": '<img src="ghost.png">', "Back": "x"},
                    }
                ]
            )
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


class TestSsrfGuard:
    @pytest.mark.parametrize("host", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1"])
    def test_private_addresses_blocked(self, host):
        with pytest.raises(ValueError, match="non-public address"):
            _check_public_address(host)

    def test_public_address_allowed(self):
        _check_public_address("8.8.8.8")  # numeric literal: no DNS, no network

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("../../etc/passwd", "passwd"), ("a/b/c.png", "c.png"), ("..", ""), ("", "")],
    )
    def test_safe_media_name(self, raw, expected):
        assert _safe_media_name(raw) == expected
