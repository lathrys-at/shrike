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


class TestServerLocalPath:
    """store_media `path` source (#164/#170): off by default; enabled only by an
    explicit --media-path-root on a purely-local daemon, and confined to that root."""

    def test_path_off_by_default(self, isolated_mcp, tmp_path):
        # The isolated server is purely-local but sets NO --media-path-root → off.
        src = tmp_path / "local.png"
        src.write_bytes(RAW)
        out = isolated_mcp("store_media", {"items": [{"path": str(src)}]})
        assert out["results"][0]["status"] == "error"
        assert "not enabled" in out["results"][0]["error"]

    def test_two_roots_disjunction_and_fetchable(self, server_factory, tmp_path):
        from .conftest import MCPClient

        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        (root_a / "in.png").write_bytes(RAW)
        (root_b / "in2.png").write_bytes(RAW)
        outside = tmp_path / "out.png"
        outside.write_bytes(RAW)

        srv = server_factory(
            "rooted",
            extra_args=["--media-path-root", str(root_a), "--media-path-root", str(root_b)],
        )
        mcp = MCPClient(srv.url)
        # A path under either root stores (disjunction).
        for f in ("in.png", "in2.png"):
            src = (root_a if f == "in.png" else root_b) / f
            assert mcp("store_media", {"items": [{"path": str(src)}]})["results"][0]["status"] == (
                "stored"
            )
        # Fetchable back.
        fetched = mcp("fetch_media", {"filenames": ["in.png"]})["results"][0]
        assert fetched["status"] == "found"
        assert httpx.get(fetched["url"]).content == RAW
        # Outside both roots is refused.
        out = mcp("store_media", {"items": [{"path": str(outside)}]})["results"][0]
        assert out["status"] == "error" and "outside the configured media root" in out["error"]

    def test_root_set_but_not_purely_local_stays_disabled(self, server_factory, tmp_path):
        from .conftest import MCPClient

        # --no-dns-rebinding-protection (proxy/tailnet signal) → not purely-local,
        # so even an in-root path is refused (the two gates compose).
        root = tmp_path / "allowed"
        root.mkdir()
        inside = root / "in.png"
        inside.write_bytes(RAW)
        srv = server_factory(
            "rooted-exposed",
            extra_args=["--media-path-root", str(root), "--no-dns-rebinding-protection"],
        )
        out = MCPClient(srv.url)("store_media", {"items": [{"path": str(inside)}]})["results"][0]
        assert out["status"] == "error" and "not enabled" in out["error"]

    def test_filesystem_root_fails_startup(self, tmp_path):
        # Per-element validation rejects `/` at startup (fail fast), even though the
        # bind is otherwise fine — a degenerate root would re-open everything.
        import subprocess
        import sys

        for d in ("logs", "state", "cache"):
            (tmp_path / d).mkdir()
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "shrike.server",
                "--collection",
                str(tmp_path / "c.anki2"),
                "--media-path-root",
                "/",
                "--foreground",
                "--log-dir",
                str(tmp_path / "logs"),
                "--state-dir",
                str(tmp_path / "state"),
                "--cache-dir",
                str(tmp_path / "cache"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 1
        assert "filesystem root" in (proc.stdout + proc.stderr)


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

    def test_store_server_path_disabled_by_default(self, isolated_runner, tmp_path):
        # The CLI sends a {path} item; the default daemon has no --media-path-root,
        # so the server refuses it (off by default, #170). Exercises the CLI
        # plumbing + the default-off behavior end-to-end.
        src = tmp_path / "on-server.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\nserver-side")
        result = isolated_runner.invoke(["media", "store", "--server-path", str(src)])
        assert result.exit_code == 1
        assert "not enabled" in result.output


# NOTE (#278 cutover): the live-httpx redirect-guard and real-TLS SNI tests
# retired with the Python fetch implementation. The native fetch's per-hop
# re-vetting + IP pinning are covered by media_fetch.rs's unit tests and the
# loopback-server cases in tests/native/test_media_url_fetch.py; a LIVE
# TLS-SNI validation test has no native equivalent yet (the Rust resolver
# can't be monkeypatched) — flagged as a residual for the security review.
