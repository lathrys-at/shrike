"""Cross-backend parity tests — the minimal embedding subset run end to end.

A parameterized fixture spins up a populated, indexed server per param; each test
runs once per param:

- ``llama`` — llama-server + the GGUF MiniLM (384-dim). Skips without llama-server.
- ``onnx`` — the ONNX backend + the same MiniLM (384-dim ONNX). Skips without the
  'onnx' extra. Same vector space as ``llama``, so they share the dimension.
- ``onnx-roberta`` — the ONNX backend + DistilRoBERTa (768-dim, BPE). A *different*
  model and vector space — its job is to prove a second, architecturally-different
  real export loads, indexes, and ranks end to end ("any ONNX dir"); it carries its
  own expected dimension, no cross-model comparison.

The expected dimension is carried in the param so the dimension assertion stays
honest across the two models. Heavier llama-specific behaviour (lifecycle, CLI,
neighbor edge cases) stays in test_semantic.py / test_embedding.py.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

from tests.integration.conftest import (
    MCPClient,
    ServerInfo,
    requires_llama_server,
    requires_onnxruntime,
)

pytestmark = [pytest.mark.integration, pytest.mark.embedding]

# A compact collection — 3 concepts × 4 cards — so both backends index quickly.
_CONCEPTS: list[dict[str, Any]] = [
    {
        "deck": "Biology",
        "tag": "cell-biology",
        "cards": [
            ("What is a mitochondrion?", "An organelle that produces ATP"),
            ("What is ATP synthase?", "Enzyme that synthesizes ATP using a proton gradient"),
            ("What is oxidative phosphorylation?", "ATP production via electron transport"),
            ("What is the citric acid cycle?", "A metabolic pathway in the matrix"),
        ],
    },
    {
        "deck": "Physics",
        "tag": "mechanics",
        "cards": [
            ("What is momentum?", "The product of an object's mass and velocity"),
            ("What is kinetic energy?", "Energy of motion: KE = 0.5 * m * v^2"),
            ("What is acceleration?", "The rate of change of velocity over time"),
            ("What is Newton's first law?", "An object at rest stays at rest unless acted on"),
        ],
    },
    {
        "deck": "Mathematics",
        "tag": "calculus",
        "cards": [
            ("What is a derivative?", "The instantaneous rate of change of a function"),
            ("What is an integral?", "The accumulation of quantities over an interval"),
            ("What is a limit?", "The value a function approaches as input approaches a point"),
            ("What is the chain rule?", "d/dx[f(g(x))] = f'(g(x)) * g'(x)"),
        ],
    },
]
_TOTAL_NOTES = sum(len(c["cards"]) for c in _CONCEPTS)


def _base_url(server: ServerInfo) -> str:
    return server.url.rsplit("/", 1)[0]


def _wait_for_index_ready(server: ServerInfo, timeout: float = 60.0) -> dict:
    base = _base_url(server)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        idx = httpx.get(f"{base}/status", timeout=5.0).json().get("index", {})
        if idx.get("state") == "ready" and idx.get("size", 0) > 0:
            return idx
        time.sleep(0.05)
    raise TimeoutError("Index did not become ready")


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(("llama", 384), marks=requires_llama_server, id="llama"),
        pytest.param(("onnx", 384), marks=requires_onnxruntime, id="onnx"),
        # No onnx-roberta server param: "a second ONNX export loads and ranks
        # end-to-end" exercises only model-agnostic plumbing; the roberta
        # lineage's own contracts (768-dim, no-PAD tokenizer, batch lock) are
        # pinned at backend level in conformance + test_onnx_models.
    ],
)
def backend_server(request: pytest.FixtureRequest, server_factory) -> tuple[ServerInfo, str, int]:
    """A populated, indexed server for one (backend, expected-dim) param."""
    backend, ndim = request.param
    if backend == "onnx":
        model = request.getfixturevalue("onnx_model")
        srv = server_factory(
            "backend-onnx",
            embedding_model=str(model),
            extra_args=["--embedding-backend", "onnx"],
        )
    else:
        model = request.getfixturevalue("embedding_model")
        srv = server_factory("backend-llama", embedding_model=str(model))

    base = _base_url(srv)
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if httpx.get(f"{base}/status", timeout=5.0).json()["embedding"]["available"]:
            break
        time.sleep(0.05)
    else:
        pytest.skip(f"{backend} embedding service did not become available")

    mcp = MCPClient(srv.url)
    notes = [
        {
            "deck": concept["deck"],
            "note_type": "Basic",
            "fields": {"Front": front, "Back": back},
            "tags": [concept["tag"]],
        }
        for concept in _CONCEPTS
        for front, back in concept["cards"]
    ]
    result = mcp("upsert_notes", {"notes": notes})
    created = sum(1 for r in result["results"] if r["status"] == "created")
    assert created == _TOTAL_NOTES, f"expected {_TOTAL_NOTES} created, got {created}"

    httpx.post(f"{base}/index/rebuild", timeout=60.0)
    _wait_for_index_ready(srv)
    return srv, backend, ndim


class TestBackendParity:
    def test_index_dimension(self, backend_server: tuple[ServerInfo, str, int]) -> None:
        srv, _, ndim = backend_server
        idx = _wait_for_index_ready(srv)
        assert idx["size"] >= _TOTAL_NOTES
        assert idx["ndim"] == ndim

    def test_model_id_namespaced_by_backend(
        self, backend_server: tuple[ServerInfo, str, int]
    ) -> None:
        # The fingerprint prefix is what keeps a backend's vectors from ever colliding
        # with another's for the "same" model (onnx-rs: vs meta:/file:).
        srv, backend, _ = backend_server
        idx = httpx.get(f"{_base_url(srv)}/status", timeout=5.0).json()["index"]
        model_id = idx.get("model_id", "")
        if backend.startswith("onnx"):
            assert model_id.startswith("onnx-rs:")
        else:
            assert model_id.startswith(("meta:", "file:"))

    def test_semantic_ranking(self, backend_server: tuple[ServerInfo, str, int]) -> None:
        srv, _, _ = backend_server
        _wait_for_index_ready(srv)
        mcp = MCPClient(srv.url)
        # threshold=0 so a borderline absolute score doesn't drop the right answer
        # (the small models here score just under the default 0.5); this is a
        # ranking check, not an absolute-score check.
        result = mcp(
            "search_notes",
            {"queries": ["derivative calculus rate of change"], "limit": 5, "threshold": 0.0},
        )
        matches = result["results"][0]["matches"]
        assert matches
        top_tags = {t for m in matches for t in m.get("tags", [])}
        assert "calculus" in top_tags
