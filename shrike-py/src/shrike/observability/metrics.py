"""Process-local Prometheus metrics.

The registry deliberately contains only Shrike metrics.  In particular it does
not install the prometheus-client process/GC collectors: this surface describes
the daemon's behaviour, is cheap to scrape, and has bounded, privacy-safe
labels.  All durations use seconds so the instruments map directly to OTel.
"""

from __future__ import annotations

import contextlib

import shrike_native
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

_DURATION_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
_INDEX_STATES = ("ready", "building", "unavailable", "error")


class Metrics:
    """The daemon's single process-local metrics registry."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.action_requests = Counter(
            "shrike_action_requests",
            "Action requests by action, transport, and result.",
            ("action", "transport", "result"),
            registry=self.registry,
        )
        self.action_duration = Histogram(
            "shrike_action_request_duration_seconds",
            "Action request latency.",
            ("action", "transport", "result"),
            buckets=_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.http_requests = Counter(
            "shrike_http_requests",
            "HTTP requests by plane, method, normalized route, and status.",
            ("plane", "method", "route", "status_code"),
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "shrike_http_request_duration_seconds",
            "HTTP request latency.",
            ("plane", "method", "route", "status_code"),
            buckets=_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.index_size = Gauge(
            "shrike_index_entries",
            "Entries in a Shrike index.",
            ("collection", "index"),
            registry=self.registry,
        )
        self.index_state = Gauge(
            "shrike_index_state",
            "One-hot index state.",
            ("collection", "index", "state"),
            registry=self.registry,
        )
        self.index_operations = Counter(
            "shrike_index_operations",
            "Index maintenance operations.",
            ("collection", "index", "operation", "result"),
            registry=self.registry,
        )
        self.index_operation_duration = Histogram(
            "shrike_index_operation_duration_seconds",
            "Index maintenance latency.",
            ("collection", "index", "operation", "result"),
            buckets=_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.collection_size = Gauge(
            "shrike_collection_notes",
            "Last observed collection note count.",
            ("collection",),
            registry=self.registry,
        )
        self.lock_attempts = Counter(
            "shrike_collection_lock_attempts",
            "Cooperative collection lock attempts.",
            ("result",),
            registry=self.registry,
        )
        self.lock_wait = Histogram(
            "shrike_collection_lock_wait_seconds",
            "Time spent acquiring the cooperative collection lock.",
            ("result",),
            buckets=_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.lock_held = Gauge(
            "shrike_collection_lock_held",
            "Whether Shrike currently holds the collection lock.",
            ("collection",),
            registry=self.registry,
        )
        self.recognition_sweeps = Counter(
            "shrike_recognition_sweeps",
            "Recognition sweeps.",
            ("result",),
            registry=self.registry,
        )
        self.recognition_duration = Histogram(
            "shrike_recognition_sweep_duration_seconds",
            "Recognition sweep latency.",
            ("result",),
            buckets=_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.recognition_items = Counter(
            "shrike_recognition_items",
            "Items stored by recognition sweeps.",
            registry=self.registry,
        )
        self.recognition_running = Gauge(
            "shrike_recognition_sweep_running",
            "Whether a recognition sweep is running, per collection.",
            ("collection",),
            registry=self.registry,
        )

    def observe_action(self, action: str, transport: str, result: str, seconds: float) -> None:
        labels = (action, transport, result)
        self.action_requests.labels(*labels).inc()
        self.action_duration.labels(*labels).observe(seconds)

    def observe_http(
        self, plane: str, method: str, route: str, status_code: int, seconds: float
    ) -> None:
        labels = (plane, method, route, str(status_code))
        self.http_requests.labels(*labels).inc()
        self.http_duration.labels(*labels).observe(seconds)

    def update_index(
        self, index: str, state: str, size: int, *, collection: str = "default"
    ) -> None:
        self.index_size.labels(collection, index).set(size)
        for candidate in _INDEX_STATES:
            self.index_state.labels(collection, index, candidate).set(candidate == state)

    def render(self) -> tuple[bytes, str]:
        # The Python registry, then the kernel's Prometheus block appended. The two
        # halves carry disjoint name prefixes (shrike_runtime_*/shrike_embedding_*/
        # shrike_index_saver_* are kernel-owned), so the concatenation is a single
        # valid exposition. A compute-only build (no anki-core) has no kernel block.
        body = generate_latest(self.registry)
        with contextlib.suppress(Exception):
            native = shrike_native.render_prometheus()
            if native:
                native_bytes = native.encode("utf-8")
                if not body.endswith(b"\n"):
                    body += b"\n"
                body += native_bytes
        return body, CONTENT_TYPE_LATEST


metrics = Metrics()
