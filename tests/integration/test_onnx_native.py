"""End-to-end test of the native ONNX backend (#270) through the real server.

The conformance coverage lives in test_backend_conformance.py (the onnx-*
cases); this boots a server with ``--embedding-backend onnx-rs`` — the retired
dual-engine kind, kept as an accepted alias of ``onnx`` since the #278 cutover —
then indexes and searches, proving the facade, the kind plumbing (incl. the
alias), and the native engine compose over the wire.
"""

from __future__ import annotations

import time

import httpx
import pytest

from tests.integration.conftest import (
    MCPClient,
    ServerInfo,
    requires_onnxruntime,
    requires_shrike_native,
)

pytestmark = [pytest.mark.integration, pytest.mark.embedding]


@requires_onnxruntime
@requires_shrike_native
class TestOnnxRsServer:
    @pytest.fixture(scope="class")
    def srv(self, server_factory, onnx_model) -> ServerInfo:
        server = server_factory(
            "onnx-rs",
            embedding_model=str(onnx_model),
            extra_args=["--embedding-backend", "onnx-rs"],
        )
        base = server.url.rsplit("/", 1)[0]
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = httpx.get(f"{base}/status", timeout=5.0).json()
            if status["embedding"]["available"]:
                return server
            time.sleep(0.05)
        pytest.fail("onnx-rs embedding service did not become available")

    def test_health_reports_native_backend(self, srv: ServerInfo) -> None:
        base = srv.url.rsplit("/", 1)[0]
        emb = httpx.get(f"{base}/status", timeout=5.0).json()["embedding"]
        # The onnx-rs alias normalizes to the canonical kind in status.
        assert emb["backend"] == "onnx"
        assert emb["active_providers"]

    def test_index_and_search(self, srv: ServerInfo) -> None:
        base = srv.url.rsplit("/", 1)[0]
        mcp = MCPClient(srv.url)
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Basic",
                        "fields": {"Front": f"What is {topic}?", "Back": desc},
                        "tags": ["onnx-rs"],
                    }
                    for topic, desc in [
                        ("a derivative", "the instantaneous rate of change of a function"),
                        ("momentum", "the product of an object's mass and velocity"),
                        ("a mitochondrion", "an organelle that produces ATP"),
                    ]
                ]
            },
        )
        assert all(r["status"] == "created" for r in result["results"])

        httpx.post(f"{base}/index/rebuild", timeout=60.0)
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            idx = httpx.get(f"{base}/status", timeout=5.0).json()["index"]
            if idx.get("state") == "ready" and idx.get("size", 0) >= 3:
                break
            time.sleep(0.05)
        else:
            pytest.fail("index did not become ready")

        # The native engine's fingerprint namespace shows up as the index model_id.
        assert idx["model_id"].startswith("onnx-rs:")

        res = mcp(
            "search_notes",
            {"queries": ["calculus rate of change"], "top_k": 3, "threshold": 0.0},
        )
        matches = res["results"][0]["matches"]
        assert matches
        assert "derivative" in matches[0]["content"]["Front"]
