"""Integration tests for the embedding service.

These tests require llama-server on PATH and download a small (~20MB)
GGUF model on first run. They are skipped automatically when
llama-server is not available.
"""

from __future__ import annotations

import httpx
import pytest

from tests.integration.conftest import requires_llama_server

pytestmark = [pytest.mark.integration, pytest.mark.embedding, requires_llama_server]


class TestEmbeddingHealth:
    """Verify the embedding service starts and reports health."""

    def test_status_reports_embedding_available(self, embedding_server):
        status_url = embedding_server.url.rsplit("/", 1)[0] + "/status"
        resp = httpx.get(status_url, timeout=5.0)
        assert resp.status_code == 200
        body = resp.json()
        assert "embedding" in body
        assert body["embedding"]["available"] is True

    def test_status_embedding_has_pid(self, embedding_server):
        status_url = embedding_server.url.rsplit("/", 1)[0] + "/status"
        body = httpx.get(status_url, timeout=5.0).json()
        assert body["embedding"]["pid"] is not None
        assert isinstance(body["embedding"]["pid"], int)

    def test_status_embedding_has_url(self, embedding_server):
        status_url = embedding_server.url.rsplit("/", 1)[0] + "/status"
        body = httpx.get(status_url, timeout=5.0).json()
        assert body["embedding"]["url"] == f"http://127.0.0.1:{embedding_server.embedding_port}"

    def test_status_embedding_has_model(self, embedding_server):
        status_url = embedding_server.url.rsplit("/", 1)[0] + "/status"
        body = httpx.get(status_url, timeout=5.0).json()
        assert body["embedding"]["model"].endswith(".gguf")

    def test_llama_server_health_endpoint(self, embedding_server):
        resp = httpx.get(f"{embedding_server.embedding_url}/health", timeout=5.0)
        assert resp.status_code == 200


class TestEmbeddings:
    """Verify actual embedding computation via the running service."""

    def test_single_text(self, embedding_server):
        resp = httpx.post(
            f"{embedding_server.embedding_url}/v1/embeddings",
            json={"input": "hello world"},
            timeout=30.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 1
        vec = data["data"][0]["embedding"]
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_batch_texts(self, embedding_server):
        texts = ["the cat sat on the mat", "dogs are loyal animals", "quantum physics"]
        resp = httpx.post(
            f"{embedding_server.embedding_url}/v1/embeddings",
            json={"input": texts},
            timeout=30.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 3
        dims = len(data["data"][0]["embedding"])
        assert all(len(item["embedding"]) == dims for item in data["data"])

    def test_similar_texts_have_close_vectors(self, embedding_server):
        texts = [
            "the weather is sunny today",
            "it is a bright and sunny day",
            "quantum chromodynamics describes strong nuclear force",
        ]
        resp = httpx.post(
            f"{embedding_server.embedding_url}/v1/embeddings",
            json={"input": texts},
            timeout=30.0,
        )
        data = resp.json()
        vecs = [item["embedding"] for item in data["data"]]

        def cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b)

        sim_close = cosine_sim(vecs[0], vecs[1])
        sim_far = cosine_sim(vecs[0], vecs[2])
        assert sim_close > sim_far


class TestEmbeddingServiceViaShrike:
    """Test the EmbeddingService.embed() method through a running server."""

    def test_embed_method(self, embedding_server):
        from shrike.embedding import EmbeddingService

        svc = EmbeddingService.__new__(EmbeddingService)
        svc._base_url = embedding_server.embedding_url
        svc._model = "test"
        svc._model_name = None
        svc._process = type("FakeProc", (), {"poll": lambda self: None})()

        result = svc.embed(["hello", "world"])
        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)
        assert all(isinstance(v[0], float) for v in result)

    def test_embed_returns_consistent_dimensions(self, embedding_server):
        from shrike.embedding import EmbeddingService

        svc = EmbeddingService.__new__(EmbeddingService)
        svc._base_url = embedding_server.embedding_url
        svc._model = "test"
        svc._model_name = None
        svc._process = type("FakeProc", (), {"poll": lambda self: None})()

        r1 = svc.embed(["a single sentence"])
        r2 = svc.embed(["another sentence", "and one more", "three total"])
        assert len(r1[0]) == len(r2[0])
        assert len(r1[0]) == len(r2[1])
