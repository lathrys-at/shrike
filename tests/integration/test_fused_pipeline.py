"""The fused native pipeline end to end (#274).

Two layers: a direct VectorIndex parity check — the fused embed→add/search
calls (vectors never crossing the FFI) return the same results as the per-side
native paths — and a fully-native server (onnx-rs backend + native index +
native compute) exercising upsert→index→search over the wire.
"""

from __future__ import annotations

import time
from pathlib import Path

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
class TestFusedIndexParity:
    def test_fused_add_and_search_match_unfused(
        self, tmp_path: Path, onnx_model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from shrike.embedding_onnx import OnnxBackend
        from shrike.index import NoteEmbedInput, VectorIndex

        texts = {
            1: "the mitochondria is the powerhouse of the cell",
            2: "momentum is mass times velocity",
            3: "a derivative is the instantaneous rate of change",
        }
        inputs = [NoteEmbedInput(nid, t) for nid, t in texts.items()]

        backend = OnnxBackend(model=str(onnx_model), native=True)
        backend.start()
        try:
            monkeypatch.setenv("SHRIKE_NATIVE_INDEX", "1")
            fused_idx = VectorIndex(tmp_path / "fused", backend=backend)
            assert fused_idx._fused_text_handles() is not None  # the fused path is live
            fused_idx.rebuild(inputs, col_mod=1, model_id="m")

            monkeypatch.delenv("SHRIKE_NATIVE_INDEX")
            plain_idx = VectorIndex(tmp_path / "plain", backend=backend)
            assert plain_idx._fused_text_handles() is None
            plain_idx.rebuild(inputs, col_mod=1, model_id="m")

            for query in ("rate of change calculus", "physics of motion"):
                fused_hits = fused_idx.search([query], top_k=3)[0]
                plain_hits = plain_idx.search([query], top_k=3)[0]
                assert [h["note_id"] for h in fused_hits] == [h["note_id"] for h in plain_hits]
                for f, p in zip(fused_hits, plain_hits, strict=True):
                    assert f["distance"] == pytest.approx(p["distance"], abs=1e-5)
        finally:
            backend.stop()


@requires_onnxruntime
@requires_shrike_native
class TestFullyNativeServer:
    @pytest.fixture(scope="class")
    def srv(self, server_factory, onnx_model, request: pytest.FixtureRequest) -> ServerInfo:
        # The spawned server inherits the env: all three native bake flags on.
        mp = pytest.MonkeyPatch()
        request.addfinalizer(mp.undo)
        mp.setenv("SHRIKE_NATIVE_INDEX", "1")
        mp.setenv("SHRIKE_NATIVE_COMPUTE", "1")
        mp.setenv("SHRIKE_NATIVE_DERIVED", "1")
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
            time.sleep(0.5)
        pytest.fail("fully-native embedding service did not become available")

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
            time.sleep(0.5)
        else:
            pytest.fail("index did not become ready")

        res = mcp(
            "search_notes",
            {"queries": ["calculus accumulation"], "top_k": 3, "threshold": 0.0},
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
            {"queries": ["ATP synthase"], "top_k": 3, "threshold": 0.0},
        )
        matches = res["results"][0]["matches"]
        assert matches
        top = matches[0]
        assert top["substring"] is not None  # the literal hit tiers first
        assert any(p["signal"] == "exact" for p in top["provenance"])
