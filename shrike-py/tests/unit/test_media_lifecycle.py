from __future__ import annotations

import base64

from shrike.schemas import (
    CollectionPruneResponse,
    FetchMediaResponse,
)
from tests.unit.conftest import make_notes

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n-fake-image-bytes").decode("ascii")


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
