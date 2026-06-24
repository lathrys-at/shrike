"""Tool-layer tests for upsert_notes: index maintenance + the tool's
duplicate/dry-run policy.

These run against a REAL AsyncKernel (the unit harness in conftest.py): writes
route through the kernel's maintained ops, the index is the kernel's own engine,
and assertions read observable state instead of facade mocks.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP

from shrike.api.tools import register_tools
from shrike.harness.harness import KernelIndexView

BASIC_NOTE = {
    "deck": "Test",
    "note_type": "Basic",
    "fields": {"Front": "Q", "Back": "A"},
}


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


def _upsert(kharness, mcp: FastMCP, notes: list[dict], **extra: Any) -> dict[str, Any]:
    # Duplicates are often intentional in these tests (similarity setups), so
    # allow them unless a test opts into a different policy.
    extra.setdefault("on_duplicate", "allow")
    return kharness.call_tool(mcp, "upsert_notes", {"notes": notes, **extra})


class TestUpsertIndexUpdate:
    def test_adds_to_index(self, kharness, mcp_sem):
        result = _upsert(kharness, mcp_sem, [BASIC_NOTE])
        nid = result["results"][0]["id"]
        assert kharness.engine.contains(nid), "the kernel op indexed the new note"

    def test_updates_col_mod_after_upsert(self, kharness, mcp_sem):
        _upsert(kharness, mcp_sem, [BASIC_NOTE])
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False


class TestUpsertPolicyTool:
    """The upsert_notes *tool* defaults (error-on-duplicate); dry_run echoes."""

    def test_tool_default_errors_on_duplicate(self, kharness, mcp_sem):
        # Call the tool directly (not the _upsert helper, which forces allow) so
        # the registered default on_duplicate="error" is what's exercised.
        first = kharness.call_tool(mcp_sem, "upsert_notes", {"notes": [BASIC_NOTE]})
        assert first["results"][0]["status"] == "created"

        second = kharness.call_tool(mcp_sem, "upsert_notes", {"notes": [BASIC_NOTE]})
        assert second["results"][0]["status"] == "error"
        assert second["results"][0]["reason"] == "duplicate"

    def test_dry_run_echoed_and_skips_index(self, kharness, mcp_sem):
        size_before = kharness.engine.size()
        result = kharness.call_tool(
            mcp_sem, "upsert_notes", {"notes": [BASIC_NOTE], "dry_run": True}
        )
        assert result["dry_run"] is True
        assert result["results"][0] == {"status": "ok", "index": 0, "action": "create"}
        # No write, so the index is never touched on a dry run.
        assert kharness.engine.size() == size_before
