"""The privileged control-plane routes over HTTP.

The control routes (/status, /index/rebuild, /index/save, /embedding/start,
/embedding/stop, /reload, /shutdown) live on the always-local control listener
(a UDS on POSIX, loopback TCP on Windows), reached via ``control_request``,
which resolves the channel from the server's server.json — the same discovery
the CLI/client use.

Every route that MUTATES or TEARS DOWN server state (/index/rebuild, /reload,
/shutdown) runs against ``isolated_server`` — a dedicated server per test — so
the session-shared ``server`` (reused by the rest of the suite) is never
disturbed. The read-only /status and the no-op /index/save (empty index) ride
the shared ``server``.

The harness here has NO embedder configured, so these assertions are on each
route's own contract (status code, response shape, post-condition) — never on
search recall, which would be vacuous without a populated derived/vector store.
"""

from __future__ import annotations

import subprocess
import time

import httpx
import pytest

from .conftest import ServerInfo

pytestmark = pytest.mark.integration


def _base_url(server: ServerInfo) -> str:
    return server.url.rsplit("/", 1)[0]


class TestStatusRoute:
    """The full /status diagnostics block — the control-plane superset of the
    data plane's minimal /health."""

    def test_status_shape_and_index_state(self, server: ServerInfo) -> None:
        resp = server.control_request("GET", "/status", timeout=10.0)
        assert resp.status_code == 200
        body = resp.json()
        # The control-plane diagnostics /health deliberately withholds.
        assert body["running"] is True
        assert body["pid"] > 0
        assert "wire_protocol_version" in body
        assert "collection" in body
        assert "log_dir" in body
        # The core status block (folded in from harness.status()).
        assert "index" in body
        # No embedder configured here, so the index is `unavailable`; the other
        # states are the populated/in-flight lifecycle (pinned so the field stays
        # a known enum, not free text).
        assert body["index"]["state"] in {
            "unavailable",
            "empty",
            "building",
            "ready",
            "errored",
        }
        assert "derived" in body
        assert body["locking"] in {"cooperative", "permanent"}
        assert "collection_held" in body

    def test_status_reports_uptime(self, server: ServerInfo) -> None:
        # `started` is set at boot, so a freshly-booted server reports an uptime
        # in the seconds bucket ("Ns"). Pinned so the uptime branch stays wired.
        body = server.control_request("GET", "/status", timeout=10.0).json()
        assert "uptime" in body
        assert body["uptime"].endswith(("s", "m"))

    def test_single_collection_omits_collections_rows(self, server: ServerInfo) -> None:
        # The per-collection `collections` array is emitted only when the manager
        # knows of more than the boot collection; a single-collection daemon's
        # payload carries no such key.
        body = server.control_request("GET", "/status", timeout=10.0).json()
        assert "collections" not in body

    def test_status_embedding_unconfigured(self, server: ServerInfo) -> None:
        # With no embedder configured the embedding block reports unavailable /
        # not_configured and the index reports the unavailable state — both are
        # always present in the payload (pinned so the fields stay wired).
        body = server.control_request("GET", "/status", timeout=10.0).json()
        assert "url" in body
        assert body["embedding"]["available"] is False
        assert body["embedding"]["state"] == "not_configured"
        assert body["index"]["state"] == "unavailable"


class TestIndexSaveRoute:
    """POST /index/save flushes the index now. With no embedder the index never
    materializes (ndim is None), so the route reports the clean `empty` status —
    the no-op flush contract, distinct from a real `saved`."""

    def test_save_empty_index_reports_empty(self, server: ServerInfo) -> None:
        resp = server.control_request("POST", "/index/save", timeout=10.0)
        assert resp.status_code == 200
        assert resp.json() == {"status": "empty"}


class TestIndexRebuildRoute:
    """POST /index/rebuild. With no embedder the harness can't rebuild, so the
    caller-actionable config error surfaces as a 400 — the documented refusal,
    not a 500. Uses isolated_server: rebuild touches index state."""

    def test_rebuild_without_embedder_is_400(self, isolated_server: ServerInfo) -> None:
        resp = isolated_server.control_request("POST", "/index/rebuild", timeout=10.0)
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert "not running" in body["error"].lower()
        # The refusal is clean: the server still serves afterward.
        assert isolated_server.control_request("GET", "/status", timeout=10.0).status_code == 200


class TestEmbeddingStopRoute:
    """POST /embedding/stop. With nothing running it reports `not_running` — the
    idempotent no-op contract — rather than erroring. Uses isolated_server: the
    route mutates embedding posture."""

    def test_stop_when_not_running_reports_not_running(self, isolated_server: ServerInfo) -> None:
        resp = isolated_server.control_request("POST", "/embedding/stop", timeout=10.0)
        assert resp.status_code == 200
        assert resp.json() == {"status": "not_running"}


class TestEmbeddingStartRoute:
    """POST /embedding/start. With no model configured every body variant lands on
    the caller-actionable `no model` 400 (KernelConfigError), never a 500. Uses
    isolated_server: a successful start would mutate the embedding posture (here
    it can't, but the route is in the control/destructive family)."""

    def test_start_without_model_is_400(self, isolated_server: ServerInfo) -> None:
        resp = isolated_server.control_request("POST", "/embedding/start", timeout=15.0)
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        # The local (UDS) control plane forwards overrides, so this is the config
        # error, NOT the exec-override refusal.
        assert "Execution-shaping parameters" not in body["error"]
        assert isolated_server.control_request("GET", "/status", timeout=10.0).status_code == 200


class TestReloadRoute:
    """POST /reload closes and re-opens the collection without a full shutdown.
    Uses isolated_server: reopen mutates the running collection's generation."""

    def test_reload_reopens_and_keeps_serving(self, isolated_server: ServerInfo) -> None:
        resp = isolated_server.control_request("POST", "/reload", timeout=15.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "reloaded"
        # `col_mod` is the post-reopen collection modification stamp; `rebuilding`
        # is False with no embedder (no reindex to drive).
        assert isinstance(body["col_mod"], int)
        assert body["rebuilding"] is False
        # The collection is live after the reopen: a follow-up status succeeds and
        # the data plane still answers /health.
        assert isolated_server.control_request("GET", "/status", timeout=10.0).status_code == 200
        assert httpx.get(f"{_base_url(isolated_server)}/health", timeout=10.0).status_code == 200


class TestShutdownRoute:
    """POST /shutdown drains both listeners gracefully (uvicorn should_exit), then
    the process exits. Uses isolated_server: this terminates the server."""

    def test_shutdown_acks_then_process_exits(self, isolated_server: ServerInfo) -> None:
        resp = isolated_server.control_request("POST", "/shutdown", timeout=10.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["pid"] > 0
        # The graceful path: uvicorn drains in-flight responses, then serve()
        # returns and the process exits. Block on the subprocess handle until it
        # exits (bounded — a /shutdown that never exits is a real bug, not a slow
        # boot); the wait returns the instant the process is reaped.
        try:
            returncode = isolated_server.proc.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            pytest.fail("server did not exit after /shutdown")
        assert returncode == 0


class TestMediaPathSafety:
    """The data-plane /media route reduces the filename to a basename inside the
    media dir. A name that sanitizes to nothing (a bare traversal token) is
    refused with a 404 before any file read — the empty-safe-name guard."""

    def test_traversal_only_name_is_404(self, server: ServerInfo) -> None:
        # '..' (URL-encoded so the router forwards it as a path param rather than
        # normalizing it away) sanitizes to an empty basename → 404, never a read
        # outside the media dir.
        resp = httpx.get(f"{_base_url(server)}/media/%2e%2e", timeout=10.0)
        assert resp.status_code == 404


class TestCooperativeLocking:
    """End-to-end cooperative locking on an isolated server."""

    def test_status_reports_and_idle_release(self, server_factory):
        from pathlib import Path

        from shrike.client import ShrikeClient

        def wait_for_release(client: ShrikeClient, deadline: float) -> bool:
            while client.status().collection_held and time.time() < deadline:
                time.sleep(0.2)
            return not client.status().collection_held

        info = server_factory(
            "coop", extra_args=["--cooperative-lock", "--lock-hold-seconds", "2.0"]
        )
        client = ShrikeClient(info.url, autostart=False, state_dir=Path(info.state_dir))

        st = client.status()
        assert st.locking == "cooperative"
        # Boot release may not have landed yet — poll with a short deadline.
        assert wait_for_release(client, time.time() + 8)

        # An MCP op re-acquires the collection.
        client.query("deck:*")
        assert client.status().collection_held is True

        # After the idle window it releases again (poll, timing-tolerant).
        assert wait_for_release(client, time.time() + 8)

        # Re-acquire still works after release — including a WRITE through
        # the kernel ops (the reopen-on-demand is kernel-side; a read alone
        # would not exercise it).
        created = client.upsert_notes(
            [
                {
                    "note_type": "Basic",
                    "deck": "Coop",
                    "fields": {"Front": "written after idle release", "Back": "b"},
                }
            ]
        )
        assert created.results[0].status == "created"
        assert client.query("deck:Coop").total == 1

    def test_default_server_is_permanent(self, server):
        from pathlib import Path

        from shrike.client import ShrikeClient

        st = ShrikeClient(server.url, autostart=False, state_dir=Path(server.state_dir)).status()
        assert st.locking == "permanent"
        assert st.collection_held is True


class TestSigtermShutdown:
    """SIGTERM tears down cleanly: uvicorn's handlers drain the server, then
    the post-serve teardown flushes/closes and releases the lock."""

    def test_sigterm_drains_and_tears_down(self, server_factory):
        import os
        import pathlib
        import signal as _signal

        info = server_factory("sigterm")
        pid = info.control_request("GET", "/status", timeout=5.0).json()["pid"]
        health_url = info.url.rsplit("/", 1)[0] + "/health"

        os.kill(pid, _signal.SIGTERM)
        # Liveness via the data-plane endpoint, not os.kill(pid, 0): the spawned
        # server stays a zombie under the test runner until reaped, so signal-0
        # keeps succeeding after exit. SIGTERM drains both listeners.
        for _ in range(75):
            try:
                httpx.get(health_url, timeout=1.0)
            except httpx.HTTPError:
                break
            time.sleep(0.2)
        else:
            pytest.fail("server still serving after SIGTERM")

        logs = ""
        for _ in range(50):
            logs = "".join(p.read_text() for p in pathlib.Path(info.log_dir).glob("*.log"))
            if "Shutdown complete" in logs:
                break
            time.sleep(0.2)
        assert "Shutdown complete" in logs
