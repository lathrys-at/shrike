"""Cross-backend parity tests — the minimal embedding subset run against BOTH
the llama-server and ONNX backends (#172).

A single parameterized fixture spins up a populated, indexed server per backend;
each test then runs once per backend. The llama param skips without llama-server,
the onnx param without the 'onnx' extra — so a machine with only one backend still
exercises that one. The model is the same all-MiniLM-L6-v2 (384-dim) in two
runtimes, so the semantic assertions hold for both.

The heavier, llama-specific behaviour (lifecycle, CLI, neighbor edge cases) stays
in test_semantic.py / test_embedding.py; this module is deliberately small.
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
_NDIM = 384  # all-MiniLM-L6-v2, both GGUF and ONNX


def _base_url(server: ServerInfo) -> str:
    return server.url.rsplit("/", 1)[0]


def _wait_for_index_ready(server: ServerInfo, timeout: float = 60.0) -> dict:
    base = _base_url(server)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        idx = httpx.get(f"{base}/status", timeout=5.0).json().get("index", {})
        if idx.get("state") == "ready" and idx.get("size", 0) > 0:
            return idx
        time.sleep(0.5)
    raise TimeoutError("Index did not become ready")


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("llama", marks=requires_llama_server),
        pytest.param("onnx", marks=requires_onnxruntime),
    ],
)
def backend_server(request: pytest.FixtureRequest, server_factory) -> tuple[ServerInfo, str]:
    """A populated, indexed server for one embedding backend (param: llama|onnx)."""
    backend = request.param
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
        time.sleep(0.5)
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
    return srv, backend


class TestBackendParity:
    def test_embedding_available(self, backend_server: tuple[ServerInfo, str]) -> None:
        srv, _ = backend_server
        status = httpx.get(f"{_base_url(srv)}/status", timeout=5.0).json()
        assert status["embedding"]["available"] is True

    def test_index_dimension(self, backend_server: tuple[ServerInfo, str]) -> None:
        srv, _ = backend_server
        idx = _wait_for_index_ready(srv)
        assert idx["size"] >= _TOTAL_NOTES
        assert idx["ndim"] == _NDIM

    def test_model_id_namespaced_by_backend(self, backend_server: tuple[ServerInfo, str]) -> None:
        # The fingerprint prefix is what keeps the two backends' vectors from ever
        # colliding for the "same" model (onnx: vs meta:/file:).
        srv, backend = backend_server
        idx = httpx.get(f"{_base_url(srv)}/status", timeout=5.0).json()["index"]
        model_id = idx.get("model_id", "")
        if backend == "onnx":
            assert model_id.startswith("onnx:")
        else:
            assert model_id.startswith(("meta:", "file:"))

    def test_semantic_ranking(self, backend_server: tuple[ServerInfo, str]) -> None:
        srv, _ = backend_server
        _wait_for_index_ready(srv)
        mcp = MCPClient(srv.url)
        # threshold=0 so a borderline absolute score doesn't drop the right answer
        # (the small models here score just under the default 0.5); this is a
        # ranking check, not an absolute-score check.
        result = mcp(
            "search_notes",
            {"queries": ["derivative calculus rate of change"], "top_k": 5, "threshold": 0.0},
        )
        matches = result["results"][0]["matches"]
        assert matches
        top_tags = {t for m in matches for t in m.get("tags", [])}
        assert "calculus" in top_tags

    def test_upsert_returns_neighbors(self, backend_server: tuple[ServerInfo, str]) -> None:
        srv, _ = backend_server
        _wait_for_index_ready(srv)
        mcp = MCPClient(srv.url)
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Biology",
                        "note_type": "Basic",
                        "fields": {
                            "Front": "What is cellular respiration?",
                            "Back": "How cells convert glucose into ATP",
                        },
                        "tags": ["cell-biology"],
                    }
                ],
                "neighbor_threshold": 0.0,
            },
        )
        r = result["results"][0]
        assert r["status"] == "created"
        assert r["neighbors"]
        assert 0 < r["neighbors"][0]["score"] <= 1.0
        mcp("delete_notes", {"ids": [r["id"]]})
