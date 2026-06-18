"""Unit tests for the resilient test-model download helper."""

from __future__ import annotations

import httpx
import pytest

from tests.integration import model_cache as mc


def _resp(status: int, content: bytes = b"data", headers: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status, content=content, headers=headers or {}, request=httpx.Request("GET", "http://x")
    )


def test_retries_transient_then_succeeds(tmp_path, monkeypatch):
    seq = iter([_resp(429, headers={"retry-after": "1"}), _resp(200, b"MODEL")])
    monkeypatch.setattr(mc.httpx, "get", lambda *a, **k: next(seq))
    slept: list[float] = []

    out = mc.download_with_retry("http://x", tmp_path / "m.gguf", sleep=slept.append)

    assert out.read_bytes() == b"MODEL"
    assert slept == [1.0]  # honored Retry-After, backed off once


def test_raises_after_persistent_429(tmp_path, monkeypatch):
    monkeypatch.setattr(mc.httpx, "get", lambda *a, **k: _resp(429))
    with pytest.raises(httpx.HTTPStatusError):
        mc.download_with_retry("http://x", tmp_path / "m.gguf", attempts=3, sleep=lambda _: None)


def test_non_retryable_status_raises_immediately(tmp_path, monkeypatch):
    calls: list[int] = []

    def fake_get(*a, **k):
        calls.append(1)
        return _resp(404)

    monkeypatch.setattr(mc.httpx, "get", fake_get)
    with pytest.raises(httpx.HTTPStatusError):
        mc.download_with_retry("http://x", tmp_path / "m.gguf", attempts=5, sleep=lambda _: None)
    assert len(calls) == 1  # 404 is not retried


def test_transport_error_is_retried(tmp_path, monkeypatch):
    calls: list[int] = []

    def fake_get(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("boom")
        return _resp(200, b"OK")

    monkeypatch.setattr(mc.httpx, "get", fake_get)
    out = mc.download_with_retry("http://x", tmp_path / "m.gguf", sleep=lambda _: None)
    assert out.read_bytes() == b"OK"
    assert len(calls) == 2


def test_cached_model_path_prefers_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SHRIKE_TEST_MODEL_DIR", str(tmp_path / "stable"))
    p = mc.cached_model_path("x.gguf", tmp_path / "fallback")
    assert p == tmp_path / "stable" / "x.gguf"
    assert p.parent.is_dir()


def test_cached_model_path_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv("SHRIKE_TEST_MODEL_DIR", raising=False)
    p = mc.cached_model_path("x.gguf", tmp_path / "fallback")
    assert p == tmp_path / "fallback" / "x.gguf"


def test_backoff_honors_retry_after():
    assert mc._backoff_delay(0, _resp(429, headers={"retry-after": "3"})) == 3.0


def test_backoff_exponential_without_header():
    assert mc._backoff_delay(2, None) == 4.0
    assert mc._backoff_delay(0, None) == 1.0
