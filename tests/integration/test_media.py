"""Media integration tests — store/fetch/list/delete + collection check/prune.

Media is collection-wide and *not* covered by the shared-collection reset
tracker, so these use the dedicated `isolated_*` fixtures (a fresh, un-reset
collection per test) rather than the shared server.
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.integration

RAW = b"\x89PNG\r\n\x1a\n-fake-image-bytes"
PNG = base64.b64encode(RAW).decode("ascii")


class TestMediaTools:
    def test_store_then_fetch_url_serves_bytes(self, isolated_mcp):
        stored = isolated_mcp("store_media", {"items": [{"data": PNG, "filename": "cell.png"}]})
        assert stored["results"][0]["status"] == "stored"
        assert stored["results"][0]["filename"] == "cell.png"

        # fetch never returns bytes — it reports `found` with a url to GET them.
        fetched = isolated_mcp("fetch_media", {"filenames": ["cell.png"]})
        result = fetched["results"][0]
        assert result["status"] == "found"
        assert "data" not in result
        assert result["mime"] == "image/png"
        assert result["url"] and result["url"].endswith("/media/cell.png")

        # The url serves the actual bytes (the base64-free retrieval path).
        resp = httpx.get(result["url"])
        assert resp.status_code == 200
        assert resp.content == RAW

    def test_client_read_media_downloads_bytes(self, isolated_server):
        from shrike.client import ShrikeClient

        with ShrikeClient(isolated_server.url, autostart=False) as client:
            client.store_media([{"data": PNG, "filename": "cell.png"}])
            assert client.read_media("cell.png") == RAW

    def test_media_endpoint_404s_for_missing_and_traversal(self, isolated_server):
        base = isolated_server.url.rsplit("/", 1)[0]
        assert httpx.get(f"{base}/media/does-not-exist.png").status_code == 404
        # Percent-encoded so httpx doesn't normalize the `..` client-side (which
        # would never reach the route) — this actually exercises _safe_media_name's
        # traversal guard on the server.
        resp = httpx.get(f"{base}/media/..%2F..%2Fsecret.txt", follow_redirects=False)
        assert resp.status_code == 404

    def test_store_bad_base64_is_per_item_error(self, isolated_mcp):
        out = isolated_mcp(
            "store_media",
            {"items": [{"data": PNG, "filename": "ok.png"}, {"data": "!!", "filename": "bad.png"}]},
        )
        assert [r["status"] for r in out["results"]] == ["stored", "error"]

    def test_list_and_glob(self, isolated_mcp):
        isolated_mcp(
            "store_media",
            {"items": [{"data": PNG, "filename": "a.png"}, {"data": PNG, "filename": "b.jpg"}]},
        )
        listing = isolated_mcp("list_media", {})
        assert listing["media_dir"]
        assert listing["count"] >= 2
        pngs = isolated_mcp("list_media", {"pattern": "*.png"})
        assert "a.png" in [f["filename"] for f in pngs["files"]]
        assert "b.jpg" not in [f["filename"] for f in pngs["files"]]

    def test_delete(self, isolated_mcp):
        isolated_mcp("store_media", {"items": [{"data": PNG, "filename": "gone.png"}]})
        out = isolated_mcp("delete_media", {"filenames": ["gone.png", "never.png"]})
        assert out["deleted"] == ["gone.png"]
        assert out["not_found"] == ["never.png"]
        assert isolated_mcp("list_media", {"pattern": "gone.png"})["count"] == 0

    def test_check_then_prune_unused(self, isolated_mcp):
        isolated_mcp("store_media", {"items": [{"data": PNG, "filename": "orphan.png"}]})
        check = isolated_mcp("collection_check", {})
        assert "orphan.png" in check["unused"]

        preview = isolated_mcp("collection_prune", {"unused_media": True, "dry_run": True})
        assert "orphan.png" in preview["unused_media"]["files"]
        assert isolated_mcp("list_media", {"pattern": "orphan.png"})["count"] == 1

        applied = isolated_mcp("collection_prune", {"unused_media": True, "dry_run": False})
        assert applied["unused_media"]["removed"] >= 1
        assert isolated_mcp("list_media", {"pattern": "orphan.png"})["count"] == 0


class TestMediaCLI:
    def test_store_list_fetch_delete(self, isolated_runner, tmp_path):
        src = tmp_path / "pic.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\nhello-cli")

        store = isolated_runner.invoke(["media", "store", str(src)])
        assert store.exit_code == 0, store.output
        assert "Stored" in store.output and "pic.png" in store.output

        listing = isolated_runner.json(["media", "list"])
        assert "pic.png" in [f["filename"] for f in listing["files"]]

        dest = tmp_path / "out.png"
        fetch = isolated_runner.invoke(["media", "fetch", "pic.png", "-o", str(dest)])
        assert fetch.exit_code == 0, fetch.output
        assert dest.read_bytes() == b"\x89PNG\r\n\x1a\nhello-cli"

        delete = isolated_runner.invoke(["media", "delete", "pic.png", "--yes"])
        assert delete.exit_code == 0, delete.output
        assert isolated_runner.json(["media", "list", "pic.png"])["count"] == 0

    def test_collection_check_reports_unused(self, isolated_runner, tmp_path):
        src = tmp_path / "orphan.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\norphan")
        isolated_runner.invoke(["media", "store", str(src)])

        data = isolated_runner.json(["collection", "check"])
        assert "orphan.png" in data["unused"]
