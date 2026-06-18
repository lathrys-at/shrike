"""Media integration tests — store/fetch/list/delete + collection check/prune.

Media is collection-wide and *not* covered by the shared-collection reset
tracker, so these run against a dedicated, un-reset collection — but they
don't need isolation from EACH OTHER (#477): one MODULE-scoped server
replaces the ~12 per-test boots that dominated this file's runtime. Every
test uses its own filenames and pattern-scoped assertions, so shared media
state never leaks into an assert (the prune test trashes whatever unused
media earlier tests left — by design, nothing reads another test's files).
Under xdist each worker gets its own module server, so cross-worker
interference is structural, not conventional. Only the tests needing
special startup args (`server_factory`) keep dedicated boots.
"""

from __future__ import annotations

import base64

import httpx
import pytest

from tests.integration.conftest import CLIRunner, MCPClient, ServerInfo, _write_cli_config

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def media_server(server_factory) -> ServerInfo:
    """One dedicated, un-reset collection shared by the whole module."""
    return server_factory("media-module")


@pytest.fixture(scope="module")
def media_mcp(media_server: ServerInfo) -> MCPClient:
    return MCPClient(media_server.url)


@pytest.fixture(scope="module")
def media_runner(media_server: ServerInfo, tmp_path_factory: pytest.TempPathFactory) -> CLIRunner:
    return CLIRunner(media_server.url, str(_write_cli_config(media_server, tmp_path_factory)))


RAW = b"\x89PNG\r\n\x1a\n-fake-image-bytes"
PNG = base64.b64encode(RAW).decode("ascii")


class TestMediaTools:
    def test_store_then_fetch_url_serves_bytes(self, media_mcp):
        stored = media_mcp("store_media", {"items": [{"data": PNG, "filename": "cell.png"}]})
        assert stored["results"][0]["status"] == "stored"
        assert stored["results"][0]["filename"] == "cell.png"

        # fetch never returns bytes — it reports `found` with a url to GET them.
        fetched = media_mcp("fetch_media", {"filenames": ["cell.png"]})
        result = fetched["results"][0]
        assert result["status"] == "found"
        assert "data" not in result
        assert result["mime"] == "image/png"
        assert result["url"] and result["url"].endswith("/media/cell.png")

        # The url serves the actual bytes (the base64-free retrieval path).
        resp = httpx.get(result["url"])
        assert resp.status_code == 200
        assert resp.content == RAW

    def test_client_read_media_downloads_bytes(self, media_server):
        from shrike.client import ShrikeClient

        with ShrikeClient(media_server.url, autostart=False) as client:
            client.store_media([{"data": PNG, "filename": "cell.png"}])
            assert client.read_media("cell.png") == RAW

    def test_media_endpoint_404s_for_missing(self, media_server):
        base = media_server.url.rsplit("/", 1)[0]
        assert httpx.get(f"{base}/media/does-not-exist.png").status_code == 404

    def test_media_endpoint_cannot_escape_media_dir(self, media_server):
        # Plant a secret one level above the media dir; a traversal request must
        # not reach it. Percent-encoded so httpx doesn't normalize `..` away
        # client-side (it would never hit the route), so this exercises the
        # server-side _safe_media_name guard for real — stronger than a bare 404,
        # which a plain not-found would also give.
        import os

        base = media_server.url.rsplit("/", 1)[0]
        media_dir = os.path.splitext(media_server.collection_path)[0] + ".media"
        secret = os.path.join(os.path.dirname(media_dir), "escape_target.txt")
        with open(secret, "w") as fh:
            fh.write("TOP-SECRET-OUTSIDE-MEDIA")

        resp = httpx.get(f"{base}/media/..%2Fescape_target.txt", follow_redirects=False)
        assert resp.status_code == 404
        assert "TOP-SECRET-OUTSIDE-MEDIA" not in resp.text

    def test_store_bad_base64_is_per_item_error(self, media_mcp):
        out = media_mcp(
            "store_media",
            {"items": [{"data": PNG, "filename": "ok.png"}, {"data": "!!", "filename": "bad.png"}]},
        )
        assert [r["status"] for r in out["results"]] == ["stored", "error"]

    def test_list_and_glob(self, media_mcp):
        media_mcp(
            "store_media",
            {"items": [{"data": PNG, "filename": "a.png"}, {"data": PNG, "filename": "b.jpg"}]},
        )
        listing = media_mcp("list_media", {})
        assert listing["media_dir"]
        assert listing["count"] >= 2
        pngs = media_mcp("list_media", {"pattern": "*.png"})
        assert "a.png" in [f["filename"] for f in pngs["files"]]
        assert "b.jpg" not in [f["filename"] for f in pngs["files"]]

    def test_delete(self, media_mcp):
        media_mcp("store_media", {"items": [{"data": PNG, "filename": "gone.png"}]})
        out = media_mcp("delete_media", {"filenames": ["gone.png", "never.png"]})
        assert out["deleted"] == ["gone.png"]
        assert out["not_found"] == ["never.png"]
        assert media_mcp("list_media", {"pattern": "gone.png"})["count"] == 0

    def test_check_then_prune_unused(self, media_mcp):
        media_mcp("store_media", {"items": [{"data": PNG, "filename": "orphan.png"}]})
        check = media_mcp("collection_check", {})
        assert "orphan.png" in check["unused"]

        preview = media_mcp("collection_prune", {"unused_media": True, "dry_run": True})
        assert "orphan.png" in preview["unused_media"]["files"]
        assert media_mcp("list_media", {"pattern": "orphan.png"})["count"] == 1

        applied = media_mcp("collection_prune", {"unused_media": True, "dry_run": False})
        assert applied["unused_media"]["removed"] >= 1
        assert media_mcp("list_media", {"pattern": "orphan.png"})["count"] == 0


class TestServerLocalPath:
    """store_media `path` source (#164/#170): off by default; enabled only by an
    explicit --media-path-root on a purely-local daemon, and confined to that root."""

    def test_path_off_by_default(self, media_mcp, tmp_path):
        # The module server is purely-local but sets NO --media-path-root → off.
        src = tmp_path / "local.png"
        src.write_bytes(RAW)
        out = media_mcp("store_media", {"items": [{"path": str(src)}]})
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
    def test_store_list_fetch_delete(self, media_runner, tmp_path):
        src = tmp_path / "pic.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\nhello-cli")

        store = media_runner.invoke(["collection", "media", "store", str(src)])
        assert store.exit_code == 0, store.output
        assert "Stored" in store.output and "pic.png" in store.output

        listing = media_runner.json(["collection", "media", "list"])
        assert "pic.png" in [f["filename"] for f in listing["files"]]

        dest = tmp_path / "out.png"
        fetch = media_runner.invoke(["collection", "media", "fetch", "pic.png", "-o", str(dest)])
        assert fetch.exit_code == 0, fetch.output
        assert dest.read_bytes() == b"\x89PNG\r\n\x1a\nhello-cli"

        delete = media_runner.invoke(["collection", "media", "delete", "pic.png", "--yes"])
        assert delete.exit_code == 0, delete.output
        assert media_runner.json(["collection", "media", "list", "pic.png"])["count"] == 0

    def test_collection_check_reports_unused(self, media_runner, tmp_path):
        src = tmp_path / "orphan.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\norphan")
        media_runner.invoke(["collection", "media", "store", str(src)])

        data = media_runner.json(["collection", "check"])
        assert "orphan.png" in data["unused"]

    def test_store_server_path_disabled_by_default(self, media_runner, tmp_path):
        # The CLI sends a {path} item; the default daemon has no --media-path-root,
        # so the server refuses it (off by default, #170). Exercises the CLI
        # plumbing + the default-off behavior end-to-end.
        src = tmp_path / "on-server.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\nserver-side")
        result = media_runner.invoke(["collection", "media", "store", "--server-path", str(src)])
        assert result.exit_code == 1
        assert "not enabled" in result.output


# NOTE (#278 cutover): the live-httpx redirect-guard and real-TLS SNI tests
# retired with the Python fetch implementation. The native fetch's per-hop
# re-vetting + IP pinning are covered by shrike-media's unit tests (#711) and the
# loopback-server cases in tests/native/test_media_url_fetch.py; a LIVE
# TLS-SNI validation test has no native equivalent yet (the Rust resolver
# can't be monkeypatched) — flagged as a residual for the security review.
