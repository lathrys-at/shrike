"""The fused native pipeline end to end (#274).

ONE fully-native server (native onnx backend + native index) exercising
upsert→index→search over the wire. The standalone ``fused_*`` parity layer
retired with the crate that carried it (#443): the fused embed→index
composition lives inside the kernel now, exercised end to end by
``TestFullyNativeServer`` below, and the raw per-side handle surfaces are
pinned in ``tests/native/test_index_engine_binding.py``. The server boots
via the retired ``onnx-rs`` alias, which IS the alias-normalization test
(absorbed from test_onnx_native.py, #441 — the two files booted near-identical
servers for the same seam).
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
class TestFullyNativeServer:
    @pytest.fixture(scope="class")
    def srv(self, server_factory, onnx_model) -> ServerInfo:
        # Index/derived/compute/backends are all native unconditionally since
        # the #278 cutover — every server is native end to end. Booted via the
        # retired `onnx-rs` kind (the accepted alias of `onnx`): the boot
        # succeeding + status normalizing it is the alias contract.
        server = server_factory(
            "fully-native",
            embedding_model=str(onnx_model),
            extra_args=["--embedding-backend", "onnx-rs"],
        )
        base = server.url.rsplit("/", 1)[0]
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if httpx.get(f"{base}/status", timeout=5.0).json()["embedding"]["available"]:
                return server
            time.sleep(0.05)
        pytest.fail("fully-native embedding service did not become available")

    def test_status_normalizes_alias_and_reports_providers(self, srv: ServerInfo) -> None:
        # From test_onnx_native.py (#441): the onnx-rs alias normalizes to the
        # canonical kind in /status, and the loaded ort providers surface.
        base = srv.url.rsplit("/", 1)[0]
        emb = httpx.get(f"{base}/status", timeout=5.0).json()["embedding"]
        assert emb["backend"] == "onnx"
        assert emb["active_providers"]

    def test_upsert_index_search_round_trip(self, srv: ServerInfo) -> None:
        base = srv.url.rsplit("/", 1)[0]
        mcp = MCPClient(srv.url)
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Basic",
                        "fields": {"Front": f"What is {t}?", "Back": d},
                        "tags": ["fused"],
                    }
                    for t, d in [
                        ("an integral", "the accumulation of quantities over an interval"),
                        ("kinetic energy", "the energy of motion"),
                        ("ATP synthase", "an enzyme that synthesizes ATP"),
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

        # The native engine's fingerprint namespace shows up as the index
        # model_id (from test_onnx_native.py, #441).
        assert idx["model_id"].startswith("onnx-rs:")

        res = mcp(
            "search_notes",
            {"queries": ["calculus accumulation"], "limit": 3, "threshold": 0.0},
        )
        matches = res["results"][0]["matches"]
        assert matches
        assert "integral" in matches[0]["content"]["Front"]
        # Provenance (#182) flows through the native fusion identically.
        assert any(p["signal"] == "text" for p in matches[0]["provenance"])

    def test_exact_tier_survives_native_fusion(self, srv: ServerInfo) -> None:
        mcp = MCPClient(srv.url)
        res = mcp(
            "search_notes",
            {"queries": ["ATP synthase"], "limit": 3, "threshold": 0.0},
        )
        matches = res["results"][0]["matches"]
        assert matches
        top = matches[0]
        assert top["substring"] is not None  # the literal hit tiers first
        assert any(p["signal"] == "exact" for p in top["provenance"])
