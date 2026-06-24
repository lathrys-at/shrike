"""Tool-layer tests for delete_notes index maintenance.

These run against a REAL AsyncKernel (the unit harness in conftest.py): writes
route through the kernel's maintained ops, the index is the kernel's own engine,
and assertions read observable state instead of facade mocks.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP

from shrike.api.tools import register_tools
from shrike.harness.harness import KernelIndexView

# One embedder for notes AND queries: each distinct string maps to its own axis,
# so two distinct strings embed to orthogonal one-hot unit vectors (cosine 0).
# The axis map resets per test (the autouse fixture below), so the dimension
# comfortably covers any single test's distinct strings.
_EMBED_DIM = 256
_axes: dict[str, int] = {}


def _axis(text: str) -> int:
    """A stable, collision-free axis index for ``text`` within one test."""
    return _axes.setdefault(text, len(_axes))


def _onehot(text: str) -> list[float]:
    v = [0.0] * _EMBED_DIM
    v[_axis(text) % _EMBED_DIM] = 1.0
    return v


@pytest.fixture(autouse=True)
def _reset_axes():
    _axes.clear()
    yield
    _axes.clear()


class _NoteBackend:
    """The kernel's embedder: a string embeds to a one-hot unit vector at its
    own axis, so two distinct strings are orthogonal (cosine 0) — a seeded note
    is semantically invisible to any other-text query until a test plants a
    scripted vector for it."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_onehot(t) for t in texts]

    def model_fingerprint(self) -> str:
        return "unit:onehot:v1"

    def embedding_dim(self) -> int:
        return _EMBED_DIM


class _StatsView(KernelIndexView):
    """KernelIndexView with an injectable activation-stats override: the
    kernel calibrates stats from real data; the gate tests script them."""

    stats_override: dict[str, dict[str, float]] | None = None

    @property
    def activation_stats(self) -> dict[str, dict[str, float]]:
        if self.stats_override is not None:
            return self.stats_override
        return super().activation_stats


@pytest.fixture()
def sem_view(kharness):
    """A real kernel index (one-hot embedder attached, materialized) behind a
    KernelIndexView. The kernel embeds queries in-core now; the view's runtime
    only needs a non-None backend so availability reads ``ready``."""
    kharness.attach_embedder(_NoteBackend())
    return _StatsView(kharness.kernel, SimpleNamespace(backend=object()))


@pytest.fixture()
def mcp_sem(kharness, sem_view):
    mcp = FastMCP("test")
    register_tools(mcp, kharness.wrapper, index=sem_view, kernel=kharness.kernel)
    return mcp


class TestDeleteIndexUpdate:
    def test_removes_from_index(self, kharness, mcp_sem, kbasic_note):
        assert kharness.engine.contains(kbasic_note)
        result = kharness.call_tool(mcp_sem, "delete_notes", {"ids": [kbasic_note]})
        assert kbasic_note in result["deleted"]
        assert not kharness.engine.contains(kbasic_note)

    def test_index_failure_doesnt_fail_delete(self, kharness, kbasic_note):
        kharness.attach_embedder(_NoteBackend())
        proxy = kharness.proxy()

        def boom(_ids):
            raise RuntimeError("index broken")

        proxy.forget_notes = boom
        mcp = FastMCP("test")
        register_tools(mcp, kharness.wrapper, index=None, kernel=proxy)
        result = kharness.call_tool(mcp, "delete_notes", {"ids": [kbasic_note]})
        assert kbasic_note in result["deleted"]

    def test_no_forget_call_on_not_found(self, kharness):
        proxy = kharness.proxy()
        proxy.spy("forget_notes")
        mcp = FastMCP("test")
        register_tools(mcp, kharness.wrapper, index=None, kernel=proxy)
        kharness.call_tool(mcp, "delete_notes", {"ids": [9999999999999]})
        assert proxy.calls["forget_notes"] == 0

    def test_updates_col_mod(self, kharness, mcp_sem, kbasic_note):
        kharness.call_tool(mcp_sem, "delete_notes", {"ids": [kbasic_note]})
        # The watermark advanced with the delete: no drift on the next check.
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False
