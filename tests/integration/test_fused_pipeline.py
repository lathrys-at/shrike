"""The fused native pipeline end to end (#274).

Two layers: a direct parity check on the raw native handles — the fused
embed→add/search calls (vectors never crossing the FFI) return the same
results as the per-side native paths — and ONE fully-native server (native
onnx backend + native index + native compute) exercising upsert→index→search
over the wire. (The facade-shaped variant retired with #355; shrike-compute's
fused_* survive as the standalone embed→index composition.) The server boots
via the retired ``onnx-rs`` alias, which IS the alias-normalization test
(absorbed from test_onnx_native.py, #441 — the two files booted near-identical
servers for the same seam).
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
    def test_fused_add_and_search_match_unfused(self, onnx_model: Path) -> None:
        import shrike_native

        from shrike.embedding_onnx import OnnxBackend

        texts = {
            1: "the mitochondria is the powerhouse of the cell",
            2: "momentum is mass times velocity",
            3: "a derivative is the instantaneous rate of change",
        }
        ids = list(texts)
        bodies = list(texts.values())

        backend = OnnxBackend(model=str(onnx_model))
        backend.start()
        try:
            native = backend._native_engine  # the shrike_native.OnnxTextEmbedder handle
            assert native is not None

            # Fused: embed→add composes inside one GIL-released native call.
            fused_engine = shrike_native.NativeIndexEngine(["text", "image"])
            chunk = backend._effective_batch(len(bodies))
            shrike_native.fused_add_text(native, fused_engine, "text", ids, bodies, chunk)

            # Per-side: the vectors cross the FFI — must build the identical index.
            plain_engine = shrike_native.NativeIndexEngine(["text", "image"])
            plain_engine.add("text", ids, backend.embed_texts(bodies))

            for query in ("rate of change calculus", "physics of motion"):
                fused_ids, fused_dists = shrike_native.fused_search_text(
                    native, fused_engine, [query], 3, ["text"]
                )[0]["text"]
                plain_ids, plain_dists = plain_engine.search_by_modality(
                    backend.embed_texts([query]), 3, ["text"]
                )[0]["text"]
                assert list(fused_ids) == list(plain_ids)
                for f, p in zip(fused_dists, plain_dists, strict=True):
                    assert f == pytest.approx(p, abs=1e-5)
        finally:
            backend.stop()


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
