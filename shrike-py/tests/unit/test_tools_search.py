"""Tests for search_notes, upsert neighbor attachment, and delete index updates.

These run against a REAL AsyncKernel (the unit harness in conftest.py): writes
route through the kernel's maintained ops, the index is the kernel's own engine,
and assertions read observable state instead of facade mocks.

The vector-planting scheme: ONE embedder (attached to the kernel) embeds every
text — note AND query — to a one-hot unit vector at the text's own axis, so two
distinct strings are orthogonal (cosine 0). A seeded note is therefore
semantically invisible to any differently-worded query (cosine 0, below the 0.5
default threshold) until a test *plants* a scripted vector for it directly in the
shared engine. ``_plant(query, [(id, dist)])`` overrides a note's vector with one
at cosine ``1 - dist`` against ``query``'s embedding, so the search clusters keep
their scripted distances while the genuine kernel-mode path — which now embeds the
query in-core (``kernel.search_fused``) — runs end to end.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP

from shrike.api.tools import register_tools
from shrike.harness.harness import KernelIndexView
from shrike.harness.index import IndexState
from tests.unit.conftest import KernelHarness

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


class _StubIndex:
    """Duck-typed index stub for the state-machine paths (building/error/
    unavailable) — the search action reads only state/progress/availability
    on those branches, and a real kernel can't be parked in them."""

    engine = None
    size = 0
    activation_stats: dict[str, dict[str, float]] = {}

    def __init__(self, state: IndexState, progress: tuple[int, int] = (0, 0)) -> None:
        self.state = state
        self.build_progress = progress
        self.available = state == IndexState.READY

    def embed_queries(self, texts: list[str]) -> None:
        return None

    def search(self, texts: list[str], top_k: int = 10) -> list[list[dict[str, Any]]]:
        return [[] for _ in texts]


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


@pytest.fixture()
def mcp_no_index(kharness):
    mcp = FastMCP("test")
    register_tools(mcp, kharness.wrapper, index=None, kernel=kharness.kernel)
    return mcp


def _mcp_with_stub(kharness, stub: _StubIndex) -> FastMCP:
    mcp = FastMCP("test")
    register_tools(mcp, kharness.wrapper, index=stub, kernel=kharness.kernel)
    return mcp


def _upsert(kharness, mcp: FastMCP, notes: list[dict], **extra: Any) -> dict[str, Any]:
    # Duplicates are often intentional in these tests (similarity setups), so
    # allow them unless a test opts into a different policy.
    extra.setdefault("on_duplicate", "allow")
    return kharness.call_tool(mcp, "upsert_notes", {"notes": notes, **extra})


class TestSearchNotesStates:
    def test_unavailable_still_runs_exact(self, kharness):
        # Semantic down, but substring matching needs no index: the call still
        # runs (no literal "test" in a fresh collection → empty group) and notes
        # that semantic ranking was skipped.
        mcp = _mcp_with_stub(kharness, _StubIndex(IndexState.UNAVAILABLE))
        result = kharness.call_tool(mcp, "search_notes", {"queries": ["test"]})
        assert "exact text matches" in result["message"]
        assert all(not g["matches"] for g in result["results"])

    def test_building_returns_progress(self, kharness):
        mcp = _mcp_with_stub(kharness, _StubIndex(IndexState.BUILDING, progress=(50, 100)))
        result = kharness.call_tool(mcp, "search_notes", {"queries": ["test"]})
        assert "50/100" in result["message"]

    def test_error_returns_message(self, kharness):
        mcp = _mcp_with_stub(kharness, _StubIndex(IndexState.ERROR))
        result = kharness.call_tool(mcp, "search_notes", {"queries": ["test"]})
        assert "error" in result["message"]

    def test_no_index_still_runs_exact(self, kharness, mcp_no_index):
        result = kharness.call_tool(mcp_no_index, "search_notes", {"queries": ["test"]})
        assert "exact text matches" in result["message"]
        assert all(not g["matches"] for g in result["results"])

    def test_ids_only_with_no_index_returns_message(self, kharness, mcp_no_index):
        # An id anchor has no literal text to match, so with no index there's
        # nothing to do — message, no results.
        nid = kharness.seed_note("Q", back="A")
        result = kharness.call_tool(mcp_no_index, "search_notes", {"ids": [nid]})
        assert result["results"] == []
        assert "unavailable" in result["message"]

    def test_requires_queries_or_ids(self, kharness, mcp_no_index):
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match="queries or ids"):
            kharness.call_tool(mcp_no_index, "search_notes", {})


class _GatedBackend(_NoteBackend):
    """A kernel-slot backend whose embed BLOCKS until released — so a write's
    embed maintenance stays in flight, holding the kernel un-settled, while a
    concurrent search reads the freshness probe."""

    def __init__(self) -> None:
        import threading

        self.release = threading.Event()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.release.wait(timeout=30.0)
        return super().embed_texts(texts)


class TestSearchNotesFreshness:
    """The stale-read advisory: the action threads through the read-time `stale`
    verdict the native search computes (the col_mod + settled bracket around the
    read). A settled search is fresh; a search run while a write drains is stale.
    The native bracket itself is proven end-to-end in test_async_kernel."""

    def test_settled_search_is_not_stale(self, kharness, mcp_sem):
        # The harness settles after the seed, so the kernel is quiescent.
        kharness.seed_note("Electron transport chain")
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["transport"]})
        assert result["stale"] is False

    def test_search_while_write_draining_is_stale(self, kharness, mcp_no_index):
        # A gated embedder parks the upsert's embed maintenance, so the kernel
        # stays un-settled; the native read's bracket sees it and the action
        # threads the resulting `stale` through. (The query has text, so the read
        # runs even with no index — the bracket is on the read either way.)
        gated = _GatedBackend()
        # No reindex at attach: the collection is empty, and a reindex would
        # itself park in the gated embed. The first embed to park is the upsert's.
        kharness.attach_embedder(gated, reindex=False)

        async def _upsert_without_settle() -> None:
            await kharness.kernel.upsert_notes_json(
                json.dumps(
                    [{"deck": "Test", "note_type": "Basic", "fields": {"Front": "Q", "Back": "A"}}]
                ),
                "allow",
                False,
            )

        kharness.run(_upsert_without_settle())  # returns once the write commits
        try:
            assert kharness.kernel.is_settled() is False, "the parked embed keeps it un-settled"
            result = kharness.call_tool_no_settle(
                mcp_no_index, "search_notes", {"queries": ["anything"]}
            )
            assert result["stale"] is True
        finally:
            # Release the embed and drain, so teardown's close() doesn't hang.
            gated.release.set()
            kharness.settle()

        # Once drained, a fresh search is no longer stale.
        result = kharness.call_tool(mcp_no_index, "search_notes", {"queries": ["anything"]})
        assert result["stale"] is False


class TestSearchNotesResults:
    def test_text_query(self, kharness, mcp_sem, kbasic_note):
        _plant(kharness, "math question", [(kbasic_note, 0.1)])
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["math question"]})
        assert len(result["results"]) == 1
        assert result["results"][0]["source"] == "math question"
        matches = result["results"][0]["matches"]
        assert len(matches) == 1
        assert matches[0]["id"] == kbasic_note
        assert matches[0]["score"] == 0.9

    def test_id_query(self, kharness, mcp_sem, kbasic_note):
        other = kharness.seed_note("Q", back="A")
        _plant(kharness, _embed_text(kharness, kbasic_note), [(other, 0.2)])
        result = kharness.call_tool(mcp_sem, "search_notes", {"ids": [kbasic_note]})
        assert len(result["results"]) == 1
        assert result["results"][0]["source"] == f"note #{kbasic_note}"
        # The anchor embeds its own note text; the planted neighbour (at cosine
        # 0.8 against that text) surfaces — exercising the id-anchor embed path.
        assert other in [m["id"] for m in result["results"][0]["matches"]]

    def test_exclude_ids(self, kharness, mcp_sem, kbasic_note):
        other = kharness.seed_note("Q", back="A")
        _plant(kharness, "test", [(kbasic_note, 0.05), (other, 0.2)])
        result = kharness.call_tool(
            mcp_sem, "search_notes", {"queries": ["test"], "exclude_ids": [kbasic_note]}
        )
        matches = result["results"][0]["matches"]
        assert all(m["id"] != kbasic_note for m in matches)

    def test_deck_filter(self, kharness, mcp_sem, kbasic_note):
        _plant(kharness, "test", [(kbasic_note, 0.1)])
        result = kharness.call_tool(
            mcp_sem, "search_notes", {"queries": ["test"], "deck": "Nonexistent"}
        )
        assert result["results"][0]["matches"] == []


class TestUnifiedSearch:
    """Each query is matched by semantics AND exact substring, folded together."""

    def test_exact_match_without_semantic(self, kharness, mcp_sem):
        # Nothing planted → no semantic hits; the literal path alone surfaces it.
        kharness.seed_note("Electron transport chain")
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["transport"]})
        m = result["results"][0]["matches"]
        assert len(m) == 1
        assert m[0]["score"] is None
        assert m[0]["substring"]["matched_fields"] == ["Front"]

    def test_both_score_and_substring(self, kharness, mcp_sem):
        nid = kharness.seed_note("Electron transport chain")
        _plant(kharness, "transport", [(nid, 0.1)])  # score 0.9
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["transport"]})
        m = result["results"][0]["matches"][0]
        assert m["score"] == 0.9
        assert m["substring"] is not None

    def test_threshold_does_not_drop_exact(self, kharness, mcp_sem):
        nid = kharness.seed_note("unique phrase here")
        # Semantic score 0.01 is below threshold → not attached; exact still includes it.
        _plant(kharness, "unique phrase", [(nid, 0.99)])
        result = kharness.call_tool(
            mcp_sem, "search_notes", {"queries": ["unique phrase"], "threshold": 0.5}
        )
        m = result["results"][0]["matches"]
        assert len(m) == 1
        assert m[0]["score"] is None
        assert m[0]["substring"] is not None

    def test_exact_first_ordering(self, kharness, mcp_sem):
        exact_nid = kharness.seed_note("alpha beta gamma")
        sem_only = kharness.seed_note("unrelated content")
        # semantic ranks the unrelated note with a high score; exact match has none
        _plant(kharness, "beta gamma", [(sem_only, 0.05)])  # 0.95
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["beta gamma"]})
        ids = [x["id"] for x in result["results"][0]["matches"]]
        assert ids[0] == exact_nid  # literal hit ranks first despite no score

    def test_literal_hit_missed_by_prefilter_still_floats(self, kharness, mcp_sem):
        # The exact tier follows the `substring` annotation, not pre-filter
        # membership — a literal hit that reaches note_data only through the SEMANTIC ranking
        # (no derived store here, and a deck scope below would be the other route) still gets
        # the annotation recompute and floats. The query contains a '*' so Anki's wildcard
        # pre-filter can't literally match it, but the field text does contain it.
        literal = kharness.seed_note("alpha *beta* gamma")
        sem_only = kharness.seed_note("unrelated content")
        _plant(kharness, "*beta* gamma", [(sem_only, 0.05), (literal, 0.20)])
        result = kharness.call_tool(
            mcp_sem, "search_notes", {"queries": ["*beta* gamma"], "threshold": 0.5}
        )
        m = result["results"][0]["matches"]
        assert [x["id"] for x in m][0] == literal  # floats above the stronger-semantic non-literal
        assert m[0]["substring"] is not None

    def test_tags_filter(self, kharness, mcp_sem, kbasic_note):
        _plant(kharness, "test", [(kbasic_note, 0.1)])
        result = kharness.call_tool(
            mcp_sem, "search_notes", {"queries": ["test"], "tags": ["nonexistent-tag"]}
        )
        assert result["results"][0]["matches"] == []

    def test_result_includes_content(self, kharness, mcp_sem, kbasic_note):
        _plant(kharness, "test", [(kbasic_note, 0.1)])
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["test"]})
        match = result["results"][0]["matches"][0]
        assert "content" in match
        assert match["content"]["Front"] == "What is 2+2?"

    def test_limit_out_of_range_rejected(self, kharness, mcp_no_index):
        """limit is schema-constrained (ge=0, le=50); above-max is rejected."""
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError):
            kharness.call_tool(mcp_no_index, "search_notes", {"queries": ["test"], "limit": 51})

    def test_limit_zero_accepted(self, kharness, mcp_no_index):
        """limit=0 means "return all" — it is a valid value, not rejected."""
        # No index attached, so this returns no semantic results but must not raise
        # on the bound (it would have under the old ge=1).
        res = kharness.call_tool(mcp_no_index, "search_notes", {"queries": ["test"], "limit": 0})
        assert "results" in res or res.get("message")

    def test_too_many_queries_rejected(self, kharness, mcp_no_index):
        """queries is capped at 50 (schema max_length) to bound embedding load."""
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError):
            kharness.call_tool(
                mcp_no_index, "search_notes", {"queries": [f"q{i}" for i in range(51)]}
            )

    def test_too_many_ids_rejected(self, kharness, mcp_no_index):
        """ids (search anchors) is likewise capped at 50."""
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError):
            kharness.call_tool(mcp_no_index, "search_notes", {"ids": list(range(51))})

    # (The over-fetch window internals live in the kernel; the outcome they
    # exist for is pinned by test_deck_filter_returns_deep_in_scope_match below.)

    def test_deck_filter_returns_deep_in_scope_match(self, kharness, mcp_sem, kbasic_note):
        """An in-deck note ranked behind out-of-deck neighbors is still returned
        — the widened window must not silently under-return.

        ``kbasic_note`` is in deck "Test"; the nearest neighbor here is in another
        deck and ranks ahead of it. A deck-scoped search must skip past the
        out-of-deck hit and still surface the in-deck one.
        """
        other = kharness.seed_note("O", deck="Other", back="A")
        _plant(kharness, "qry", [(other, 0.05), (kbasic_note, 0.20)])
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["qry"], "deck": "Test"})
        matches = result["results"][0]["matches"]
        assert [m["id"] for m in matches] == [kbasic_note]

    def test_score_rounded_to_3_decimals(self, kharness, mcp_sem, kbasic_note):
        _plant(kharness, "test", [(kbasic_note, 0.12345)])
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["test"]})
        score = result["results"][0]["matches"][0]["score"]
        assert score == round(1.0 - 0.12345, 3)

    def test_image_modality_hit_surfaces_unthresholded(self, kharness, mcp_sem):
        # An image-modality match with no text match still surfaces the note: the image ranking is
        # its own RRF signal and is NOT thresholded (the text-calibrated threshold is meaningless
        # across the CLIP gap; flooring image hits is the activation gate's job). The surfaced
        # score is the (gap-depressed but real) image cosine.
        nid = kharness.seed_note("diagram of the krebs cycle")
        _plant(
            kharness, "mitochondria", [(nid, 0.7)], modality="image"
        )  # 0.30 sim — below threshold, kept
        result = kharness.call_tool(
            mcp_sem, "search_notes", {"queries": ["mitochondria"], "threshold": 0.5}
        )
        m = result["results"][0]["matches"]
        assert [x["id"] for x in m] == [nid]
        assert m[0]["score"] == 0.3

    def test_text_modality_stays_thresholded(self, kharness, mcp_sem):
        # The text ranking keeps its threshold: a weak text-only hit with no literal match drops.
        nid = kharness.seed_note("unrelated content here")
        _plant(kharness, "xyz", [(nid, 0.9)])  # 0.10 sim — below threshold
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["xyz"], "threshold": 0.5})
        assert result["results"][0]["matches"] == []

    def test_score_is_max_over_matched_modalities(self, kharness, mcp_sem):
        # A note matching in both text and image gets the *max* similarity as its surfaced score.
        nid = kharness.seed_note("alpha")
        _plant(kharness, "alpha query", [(nid, 0.1)])  # 0.90 text
        _plant(kharness, "alpha query", [(nid, 0.7)], modality="image")  # 0.30 image
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["alpha query"]})
        m = result["results"][0]["matches"][0]
        assert m["score"] == 0.9  # max(0.90 text, 0.30 image)

    def test_image_gate_passes_strong_match(self, kharness, mcp_sem, sem_view):
        # Calibrated floor = mean + ACTIVATION_MARGIN·std = 0.20 + 1.0·0.05 = 0.25. A best
        # image sim of 0.30 clears it → the (image-only) note surfaces, scored by the image sim.
        nid = kharness.seed_note("krebs cycle diagram")
        sem_view.stats_override = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        _plant(kharness, "mitochondria", [(nid, 0.70)], modality="image")  # sim 0.30 > 0.25
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["mitochondria"]})
        m = result["results"][0]["matches"]
        assert [x["id"] for x in m] == [nid]
        assert m[0]["score"] == 0.3

    def test_image_gate_drops_weak_match(self, kharness, mcp_sem, sem_view):
        # Best image sim 0.20 is below the 0.25 floor → the image modality is gated out, so an
        # image-only match does not surface (no spurious image card for an off-topic query).
        nid = kharness.seed_note("krebs cycle diagram")
        sem_view.stats_override = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        _plant(kharness, "mitochondria", [(nid, 0.80)], modality="image")  # sim 0.20 <= 0.25
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["mitochondria"]})
        assert result["results"][0]["matches"] == []

    def test_image_gate_keeps_text_matched_note(self, kharness, mcp_sem, sem_view):
        # Gating the image modality must not drop a note that *also* matches text above threshold;
        # it surfaces with the text score, and the gated image sim is not folded into `score`.
        nid = kharness.seed_note("alpha")
        sem_view.stats_override = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        _plant(kharness, "alpha query", [(nid, 0.20)])  # sim 0.80 (above threshold)
        _plant(kharness, "alpha query", [(nid, 0.80)], modality="image")  # sim 0.20 (gated out)
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["alpha query"]})
        m = result["results"][0]["matches"][0]
        assert m["id"] == nid
        assert m["score"] == 0.8  # text sim only; the gated image sim is not the max

    def test_image_gate_judges_surviving_hit(self, kharness, mcp_sem, sem_view):
        # The gate must judge the best image hit that *survives* exclusion/scope,
        # not the raw rank-1. Here the strong rank-1 image hit is the excluded anchor; the only
        # surviving image hit is weak (below the 0.25 floor) → the modality must be gated out.
        anchor = kharness.seed_note("anchor card")
        weak = kharness.seed_note("weakly related card")
        sem_view.stats_override = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        _plant(kharness, "qry", [(anchor, 0.65), (weak, 0.80)], modality="image")
        result = kharness.call_tool(
            mcp_sem, "search_notes", {"queries": ["qry"], "exclude_ids": [anchor]}
        )
        assert result["results"][0]["matches"] == []  # gated on the surviving (weak) hit

    def test_image_gate_passes_strong_surviving_hit(self, kharness, mcp_sem, sem_view):
        # The mirror: with the strong anchor excluded, a surviving hit that itself clears the floor
        # still surfaces — the gate isn't fooled in either direction.
        anchor = kharness.seed_note("anchor card")
        strong = kharness.seed_note("strongly matching card")
        sem_view.stats_override = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        _plant(kharness, "qry", [(anchor, 0.55), (strong, 0.66)], modality="image")
        result = kharness.call_tool(
            mcp_sem, "search_notes", {"queries": ["qry"], "exclude_ids": [anchor]}
        )
        assert [x["id"] for x in result["results"][0]["matches"]] == [strong]

    def test_limit_zero_returns_all_across_two_spaces(self, kharness, mcp_sem, sem_view):
        # `limit=0` == "return all" on a TWO-SPACE (text+image) index. The index
        # is split into per-modality sub-indexes;
        # the limit=0 clamp reads `index.size`, which is the AGGREGATE across all
        # sub-indexes (text count + image count). If it ever read only one
        # sub-index's size, limit=0 would under-fetch and silently drop results.
        # Six notes, each vectored in BOTH spaces, so the aggregate size (12)
        # exceeds the note count (6); limit=0 must still return all six and not
        # hang on a runaway over-fetch.
        sem_view.stats_override = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        nids = [kharness.seed_note(f"shared subject card {i}") for i in range(6)]
        _plant(kharness, "shared subject", [(n, 0.05) for n in nids])  # strong text sim (~0.95)
        _plant(
            kharness, "shared subject", [(n, 0.30) for n in nids], modality="image"
        )  # image sim 0.70 > floor
        result = kharness.call_tool(
            mcp_sem, "search_notes", {"queries": ["shared subject"], "limit": 0}
        )
        got = {m["id"] for m in result["results"][0]["matches"]}
        assert got == set(nids)


class TestProvenance:
    """Per-result provenance: which signals surfaced each match, at what rank."""

    @staticmethod
    def _matches(kharness, mcp_app, query: str) -> list[dict]:
        result = kharness.call_tool(mcp_app, "search_notes", {"queries": [query]})
        return result["results"][0]["matches"]

    def test_text_only(self, kharness, mcp_sem):
        nid = kharness.seed_note("mitochondria powerhouse")
        _plant(kharness, "cellular energy", [(nid, 0.2)])
        m = self._matches(kharness, mcp_sem, "cellular energy")[0]
        assert [(p["signal"], p["rank"]) for p in m["provenance"]] == [("text", 1)]
        assert m["score"] == 0.8  # back-compat field stays, consistent with the text signal

    def test_image_modality_facet(self, kharness, mcp_sem):
        # The semantic signal name *is* the matched-modality facet — `image` ⇒ "matched on image".
        nid = kharness.seed_note("krebs cycle diagram card")
        _plant(
            kharness, "mitochondria", [(nid, 0.7)], modality="image"
        )  # uncalibrated → gate off → surfaces
        m = self._matches(kharness, mcp_sem, "mitochondria")[0]
        assert [p["signal"] for p in m["provenance"]] == ["image"]
        assert m["score"] == 0.3

    def test_exact_only(self, kharness, mcp_sem):
        kharness.seed_note("unique exact phrase")
        m = self._matches(kharness, mcp_sem, "exact phrase")[0]  # nothing planted → no semantic
        assert [p["signal"] for p in m["provenance"]] == ["exact"]
        assert m["score"] is None  # back-compat: exact-only carries no score
        assert m["substring"] is not None  # ...but the substring detail stays

    def test_text_and_exact(self, kharness, mcp_sem):
        nid = kharness.seed_note("Electron transport chain")
        _plant(kharness, "transport", [(nid, 0.1)])
        m = self._matches(kharness, mcp_sem, "transport")[0]
        # Both fire at rank 1 → ordered by signal name (exact < text); back-compat fields agree.
        assert {p["signal"]: p["rank"] for p in m["provenance"]} == {"text": 1, "exact": 1}
        assert m["score"] == 0.9
        assert m["substring"] is not None

    def test_ordered_by_rank_then_signal(self, kharness, mcp_sem):
        a = kharness.seed_note("alpha card")
        b = kharness.seed_note("beta card")
        nid = kharness.seed_note("gamma card")
        # nid trails a, b in text (rank 3) but leads the image ranking (rank 1).
        _plant(kharness, "qry", [(a, 0.10), (b, 0.15), (nid, 0.20)])
        _plant(kharness, "qry", [(nid, 0.65)], modality="image")
        matches = self._matches(kharness, mcp_sem, "qry")
        assert all(m["provenance"] for m in matches)  # every returned match carries provenance
        prov = {m["id"]: [(p["signal"], p["rank"]) for p in m["provenance"]] for m in matches}
        assert prov[nid] == [("image", 1), ("text", 3)]  # strongest (lowest-rank) signal first
        assert prov[a] == [("text", 1)]


class TestDerivedSearch:
    """search_notes over the kernel's OWN derived store: substring-via-store +
    the fuzzy signal. The kernel maintains the store on every upsert, so
    ``seed_note`` populates it — no host derived store is passed (search reads
    the kernel's)."""

    def test_substring_via_store_matches_find_notes(self, kharness, mcp_sem):
        # An exact substring hit comes through the store (candidate) + substring_info (authority),
        # identical to the find_notes path: matched field + the `exact` provenance.
        nid = kharness.seed_note("Electron transport chain")
        res = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["transport"]})
        m = res["results"][0]["matches"]
        assert [x["id"] for x in m] == [nid]
        assert m[0]["substring"]["matched_fields"] == ["Front"]
        assert m[0]["substring"]["source"] == "field"
        # A literal hit shares every trigram so it's *trivially* also a fuzzy match, but `fuzzy` is
        # suppressed on exact hits — `exact` is the distinguishing lexical signal.
        assert [p["signal"] for p in m[0]["provenance"]] == ["exact"]
        assert m[0].get("fuzzy") is None

    def test_fuzzy_only_hit_surfaces_with_provenance(self, kharness, mcp_sem):
        # A typo query the note doesn't literally contain surfaces via the `fuzzy` signal alone:
        # no score, no substring, provenance == [fuzzy], carrying the source/ref/snippet window.
        nid = kharness.seed_note("Mitochondria are the powerhouse")
        res = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["mitochndria"]})
        m = res["results"][0]["matches"]
        assert [x["id"] for x in m] == [nid]
        hit = m[0]
        assert hit["score"] is None
        assert hit["substring"] is None
        assert [p["signal"] for p in hit["provenance"]] == ["fuzzy"]
        assert hit["fuzzy"]["source"] == "field"
        assert hit["fuzzy"]["ref"] == "Front"
        assert "Mitochondria" in hit["fuzzy"]["snippet"]

    def test_literal_tiers_above_fuzzy(self, kharness, mcp_sem):
        # The exact-match override still wins: a literal hit floats above a fuzzy-only near-miss.
        literal = kharness.seed_note("Mitochondria diagram")
        fuzzy_only = kharness.seed_note("mitochndrial membrane")  # typo → no literal hit
        res = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["mitochondria"]})
        m = res["results"][0]["matches"]
        assert [x["id"] for x in m][0] == literal  # literal floats to the top
        prov = {x["id"]: [p["signal"] for p in x["provenance"]] for x in m}
        assert "exact" in prov[literal]
        assert prov[fuzzy_only] == ["fuzzy"]  # the near-miss is fuzzy-only

    def test_exact_hit_carries_no_fuzzy(self, kharness, mcp_sem):
        # A clean exact (literal) match must not also be badged `fuzzy`, even though it
        # shares every trigram — `fuzzy` is reserved for the distinguishing near-miss signal.
        nid = kharness.seed_note("powerhouse of the cell")
        res = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["powerhouse"]})
        m = res["results"][0]["matches"]
        hit = next(x for x in m if x["id"] == nid)
        assert "fuzzy" not in [p["signal"] for p in hit["provenance"]]
        assert hit.get("fuzzy") is None

    def test_result_capped_at_limit(self, kharness, mcp_sem):
        # The fused union (text/image/exact/fuzzy, each up to limit) is capped to limit,
        # so a broad fuzzy signal can't inflate a query's result count past the documented cap.
        for i in range(8):
            kharness.seed_note(f"mitochondrion variant {i}")  # all fuzzy-match the typo
        res = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["mitochndrion"], "limit": 3})
        assert len(res["results"][0]["matches"]) == 3

    def test_limit_zero_returns_all(self, kharness, mcp_sem):
        # limit=0 means "return all": the same broad fuzzy match that the
        # cap-to-3 test truncates must come back in full when the cap is lifted.
        for i in range(8):
            kharness.seed_note(f"mitochondrion variant {i}")
        res = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["mitochndrion"], "limit": 0})
        assert len(res["results"][0]["matches"]) == 8


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


class TestTwoTierSearch:
    """The live-search tier contract: tier='live' runs only the
    no-embedding signals and reports partial; the min-query gate keeps typing
    fragments from burning embedding calls; `version` echoes verbatim."""

    def test_live_tier_skips_semantic_and_reports_partial(self, kharness, mcp_sem):
        planted = kharness.seed_note("sem only", back="A")
        _plant(kharness, "qry", [(planted, 0.05)])
        result = kharness.call_tool(
            mcp_sem, "search_notes", {"queries": ["qry"], "tier": "live", "version": 7}
        )
        assert result["completeness"] == "partial"
        assert result["version"] == 7
        # The semantically-planted note does not surface on the live tier.
        ids = [m["id"] for m in result["results"][0]["matches"]]
        assert planted not in ids

    def test_full_tier_reports_full_and_finds_semantic(self, kharness, mcp_sem):
        planted = kharness.seed_note("sem hit", back="A")
        _plant(kharness, "qry", [(planted, 0.05)])
        result = kharness.call_tool(mcp_sem, "search_notes", {"queries": ["qry"]})
        assert result["completeness"] == "full"
        assert result["version"] is None
        assert planted in [m["id"] for m in result["results"][0]["matches"]]

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
