"""Tool-layer tests for search_notes retrieval modes + the dedup-stats sampler.

These run against a REAL AsyncKernel (the unit harness in conftest.py): writes
route through the kernel's maintained ops, the index is the kernel's own engine,
and assertions read observable state instead of facade mocks.

The vector-planting scheme: ONE embedder (attached to the kernel) embeds every
text — note AND query — to a one-hot unit vector at the text's own axis, so two
distinct strings are orthogonal (cosine 0). A seeded note is therefore
semantically invisible to any differently-worded query until a test *plants* a
scripted vector for it directly in the shared engine.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP

from shrike.api.tools import register_tools
from shrike.harness.harness import KernelIndexView
from tests.unit.conftest import KernelHarness

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


def _plant(
    kharness: KernelHarness,
    query: str,
    items: list[tuple[int, float]],
    modality: str = "text",
) -> None:
    """Override each note's vector so the kernel ranks it at exactly cosine
    ``1 - dist`` against ``query``'s embedding (a one-hot at ``query``'s axis).
    The planted vector replaces the note's own kernel vector for ``modality``."""
    qaxis = _axis(query)
    alt = (qaxis + 1) % _EMBED_DIM
    keys = [nid for nid, _ in items]
    vecs: list[list[float]] = []
    for _, dist in items:
        c = max(min(1.0 - dist, 1.0), -1.0)
        v = [0.0] * _EMBED_DIM
        v[qaxis] = c
        v[alt] = math.sqrt(max(0.0, 1.0 - c * c))
        vecs.append(v)
    kharness.engine.add(modality, keys, vecs)


def _embed_text(kharness: KernelHarness, nid: int) -> str:
    """The exact text the kernel embeds for an id anchor — so an id-anchored
    search plants its neighbours against the same axis the anchor lands on."""

    async def _go() -> str:
        return (await kharness.wrapper.note_texts_for_embedding([nid]))[0]

    return kharness.run(_go())


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


class TestSearchMode:
    """The retrieval-mode contract: mode='lexical' runs only the
    no-embedding signals and reports partial; the min-query gate keeps typing
    fragments from burning embedding calls; `version` echoes verbatim."""

    def test_lexical_mode_skips_semantic_and_reports_partial(self, kharness, mcp_sem):
        planted = kharness.seed_note("sem only", back="A")
        _plant(kharness, "qry", [(planted, 0.05)])
        result = kharness.call_tool(
            mcp_sem, "search_notes", {"queries": ["qry"], "mode": "lexical", "version": 7}
        )
        assert result["completeness"] == "partial"
        assert result["version"] == 7
        # The semantically-planted note does not surface in lexical mode.
        ids = [m["id"] for m in result["results"][0]["matches"]]
        assert planted not in ids

    def test_fused_mode_reports_full_and_finds_semantic(self, kharness, mcp_sem):
        planted = kharness.seed_note("sem hit", back="A")
        _plant(kharness, "qry", [(planted, 0.05)])
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["qry"]})
        assert result["completeness"] == "full"
        assert result["version"] is None
        assert planted in [m["id"] for m in result["results"][0]["matches"]]

    def test_single_query_lexical_routes_to_the_dedicated_routine(self, kharness, sem_view):
        # Lock the fast-path routing: ONLY (mode=lexical, exactly one query, no id
        # anchors) takes the dedicated kernel routine; every other shape takes the
        # general fused path. Parity makes the results identical either way, so a
        # spy on which kernel method the action calls is the only thing that pins it.
        kharness.seed_note("mitochondria powerhouse", back="energy")
        anchor = kharness.seed_note("anchor note", back="A")

        class _SpyKernel:
            def __init__(self, real):
                self._real = real
                self.calls: list[str] = []

            def search_lexical_single(self, *a, **k):
                self.calls.append("search_lexical_single")
                return self._real.search_lexical_single(*a, **k)

            def search_fused(self, *a, **k):
                self.calls.append("search_fused")
                return self._real.search_fused(*a, **k)

            def __getattr__(self, name):
                return getattr(self._real, name)

        cases = [
            ({"queries": ["mito"], "mode": "lexical"}, "search_lexical_single"),
            ({"queries": ["mito", "chond"], "mode": "lexical"}, "search_fused"),
            ({"queries": ["mito"], "mode": "fused"}, "search_fused"),
            ({"queries": ["mito"], "ids": [anchor], "mode": "lexical"}, "search_fused"),
        ]
        for params, expected in cases:
            spy = _SpyKernel(kharness.kernel)
            mcp = FastMCP("test")
            register_tools(mcp, kharness.wrapper, index=sem_view, kernel=spy)
            kharness.call_tool(mcp, "search_notes", params)
            assert spy.calls == [expected], f"{params} -> {spy.calls}, expected {expected}"

    def test_min_query_gate_skips_semantic_but_is_final(self, kharness, mcp_sem):
        planted = kharness.seed_note("ab gate", back="A")
        _plant(kharness, "ab", [(planted, 0.05)])
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["ab"]})
        # Final for this query (a client must not poll for more) + advisory.
        assert result["completeness"] == "full"
        assert "skipped" in result["message"]
        # The literal substring still matches (the cheap signals ran).
        assert planted in [m["id"] for m in result["results"][0]["matches"]]

    def test_id_anchors_are_never_gated(self, kharness, mcp_sem):
        a = kharness.seed_note("anchor", back="A")
        b = kharness.seed_note("neighbor", back="A")
        _plant(kharness, _embed_text(kharness, a), [(b, 0.10)])
        result = kharness.call_tool(mcp_sem, "search_notes", {"ids": [a]})
        assert result["completeness"] == "full"
        assert b in [m["id"] for m in result["results"][0]["matches"]]


class TestDedupStats:
    """The calibration feedstock: one best-semantic-match sample per search
    query group, recorded from the search path (never the activation gate)."""

    def test_recorder_buckets_and_no_match(self):
        from shrike.harness.harness import DedupStatsRecorder

        rec = DedupStatsRecorder()
        assert rec.snapshot() is None  # empty → absent from /status

        rec.record(0.62)
        rec.record(0.97)
        rec.record(1.0)  # clamps into the last bucket
        rec.record(None)
        snap = rec.snapshot()
        assert snap["samples"] == 4
        assert snap["no_match"] == 1
        assert snap["buckets"][12] == 1  # 0.62 → [0.60, 0.65)
        assert snap["buckets"][19] == 2  # 0.97 and the clamped 1.0

    @pytest.fixture()
    def stats(self):
        from shrike.harness.harness import DedupStatsRecorder

        return DedupStatsRecorder()

    @pytest.fixture()
    def mcp_dedup_stats(self, kharness, sem_view, stats):
        mcp = FastMCP("test")
        register_tools(
            mcp, kharness.wrapper, index=sem_view, dedup_stats=stats, kernel=kharness.kernel
        )
        return mcp

    def test_search_records_a_sample(self, kharness, mcp_dedup_stats, stats):
        # The sampler rides the search path now (#848): one best-semantic-match
        # sample per query group, the same scores the dropped neighbor self-
        # search produced.
        existing = kharness.seed_note("anchor card")
        _plant(kharness, "zz new card zz", [(existing, 0.1)])  # best match 0.9
        kharness.call_tool(mcp_dedup_stats, "search_notes", {"queries": ["zz new card zz"]})
        snap = stats.snapshot()
        assert snap["samples"] == 1
        assert snap["buckets"][18] == 1  # 0.9 → [0.90, 0.95)
