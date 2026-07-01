"""Prometheus registry shape and the appended kernel exposition block."""

from __future__ import annotations

import shrike_native
from prometheus_client.parser import text_string_to_metric_families

from shrike.observability.metrics import Metrics


def _samples(text: str) -> dict[str, float]:
    return {
        sample.name + str(sorted(sample.labels.items())): sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
    }


def test_action_and_http_metrics_use_bounded_labels() -> None:
    registry = Metrics()
    registry.observe_action("search_notes", "mcp", "ok", 0.01)
    registry.observe_http("data", "GET", "/media/{filename:path}", 404, 0.02)

    body, content_type = registry.render()
    text = body.decode()
    assert content_type.startswith("text/plain")
    assert (
        'shrike_action_requests_total{action="search_notes",result="ok",transport="mcp"} 1.0'
        in text
    )
    assert 'route="/media/{filename:path}"' in text
    assert "a-secret-filename" not in text


def test_kernel_block_is_appended(monkeypatch) -> None:
    # The kernel's Prometheus exporter renders its own block; /metrics appends it
    # after the Python registry. Disjoint name prefixes keep the concatenation a
    # single valid exposition.
    block = (
        "# HELP shrike_runtime_pool_workers Live workers driving a pool.\n"
        "# TYPE shrike_runtime_pool_workers gauge\n"
        'shrike_runtime_pool_workers{pool="compute"} 4\n'
    )
    monkeypatch.setattr(shrike_native, "render_prometheus", lambda: block)

    body, _ = Metrics().render()
    text = body.decode()
    samples = _samples(text)
    # Both halves are present and parse as one exposition: the appended kernel
    # block and the Python registry's own families (HELP emitted via auto_describe).
    assert samples["shrike_runtime_pool_workers[('pool', 'compute')]"] == 4
    assert "shrike_action_requests" in text


def test_render_tolerates_absent_kernel_block(monkeypatch) -> None:
    # A compute-only build (no anki-core) or a render failure must not break
    # /metrics — the Python registry still renders on its own.
    def _boom() -> str:
        raise RuntimeError("no kernel exporter")

    monkeypatch.setattr(shrike_native, "render_prometheus", _boom)
    body, _ = Metrics().render()
    assert isinstance(body, bytes)
    assert b"shrike_action_requests" in body


def test_index_state_is_one_hot() -> None:
    registry = Metrics()
    registry.update_index("vector", "building", 42)
    samples = _samples(registry.render()[0].decode())
    assert samples["shrike_index_entries[('collection', 'default'), ('index', 'vector')]"] == 42

    def _state_key(state: str) -> str:
        return (
            "shrike_index_state[('collection', 'default'), "
            f"('index', 'vector'), ('state', '{state}')]"
        )

    assert samples[_state_key("building")] == 1
    assert samples[_state_key("ready")] == 0


def test_recognition_running_is_per_collection() -> None:
    registry = Metrics()
    registry.recognition_running.labels("default").set(1)
    samples = _samples(registry.render()[0].decode())
    assert samples["shrike_recognition_sweep_running[('collection', 'default')]"] == 1
