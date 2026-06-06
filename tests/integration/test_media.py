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

    def test_media_endpoint_404s_for_missing(self, isolated_server):
        base = isolated_server.url.rsplit("/", 1)[0]
        assert httpx.get(f"{base}/media/does-not-exist.png").status_code == 404

    def test_media_endpoint_cannot_escape_media_dir(self, isolated_server):
        # Plant a secret one level above the media dir; a traversal request must
        # not reach it. Percent-encoded so httpx doesn't normalize `..` away
        # client-side (it would never hit the route), so this exercises the
        # server-side _safe_media_name guard for real — stronger than a bare 404,
        # which a plain not-found would also give.
        import os

        base = isolated_server.url.rsplit("/", 1)[0]
        media_dir = os.path.splitext(isolated_server.collection_path)[0] + ".media"
        secret = os.path.join(os.path.dirname(media_dir), "escape_target.txt")
        with open(secret, "w") as fh:
            fh.write("TOP-SECRET-OUTSIDE-MEDIA")

        resp = httpx.get(f"{base}/media/..%2Fescape_target.txt", follow_redirects=False)
        assert resp.status_code == 404
        assert "TOP-SECRET-OUTSIDE-MEDIA" not in resp.text

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


class TestRedirectRealHttpx:
    """The end-to-end proof the unit fakes can't give: with real httpx, a 302 is
    NOT auto-followed past the guard. Catches a follow_redirects=True regression."""

    def test_real_httpx_does_not_autofollow_past_guard(self, monkeypatch):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from shrike import collection as collection_mod
        from shrike.collection import _fetch_media_url

        hits: list[str] = []

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_GET(self):  # noqa: N802 (stdlib API)
                hits.append(self.path)
                if self.path == "/start":
                    self.send_response(302)
                    self.send_header("Location", "/internal")
                    self.end_headers()
                elif self.path == "/internal":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"SECRET")
                else:
                    self.send_response(404)
                    self.end_headers()

        srv = HTTPServer(("127.0.0.1", 0), _Handler)
        port = srv.server_port
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            # Allow the first hop (return its vetted IP), refuse the second — so a
            # correct (no-autofollow) loop re-enters and raises before /internal.
            calls = {"n": 0}

            def fake_resolve(host: str) -> str:
                calls["n"] += 1
                if calls["n"] >= 2:
                    raise ValueError(f"refusing to fetch from non-public address (host '{host}')")
                return host  # loopback literal — pin straight back to it

            monkeypatch.setattr(collection_mod, "_resolve_public_ip", fake_resolve)
            with pytest.raises(ValueError, match="non-public address"):
                _fetch_media_url(f"http://127.0.0.1:{port}/start", allow_private=False)
            assert "/start" in hits
            assert "/internal" not in hits  # decisive: never fetched
        finally:
            srv.shutdown()


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
