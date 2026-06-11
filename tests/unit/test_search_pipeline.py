"""SearchPipeline seam (#274): protocol conformance + native==reference parity.

`search_fusion.rrf_fuse` is the frozen, readable spec of the fusion semantics;
the native pipeline (`shrike_compute::rrf_fuse`) is a second implementation of
the same `SearchPipeline` protocol. The property suite here is the drift alarm
that justifies the dual implementation: randomized rankings/weights/priority
tiers in → identical fused order, scores, and provenance out.
"""

from __future__ import annotations

import importlib.util
import random

import pytest

from shrike.search_fusion import (
    RRF_K,
    NativeSearchPipeline,
    ReferenceSearchPipeline,
    SearchPipeline,
    make_search_pipeline,
    rrf_fuse,
)

requires_shrike_native = pytest.mark.skipif(
    importlib.util.find_spec("shrike_native") is None,
    reason="shrike_native extension not installed (scripts/build-native.sh)",
)

SIGNALS = ["text", "image", "exact", "fuzzy", "tag-centroid"]


def _random_case(rng: random.Random) -> tuple[dict[str, list[int]], dict[str, float], frozenset]:
    rankings: dict[str, list[int]] = {}
    for signal in rng.sample(SIGNALS, k=rng.randint(1, len(SIGNALS))):
        # Duplicates included deliberately — one-signal-one-rank is part of the spec.
        rankings[signal] = [rng.randint(1, 30) for _ in range(rng.randint(0, 25))]
    weights = {s: rng.choice([0.25, 0.5, 1.0, 2.0]) for s in rankings if rng.random() < 0.7}
    priority = frozenset(s for s in rankings if rng.random() < 0.3)
    return rankings, weights, priority


class TestProtocol:
    def test_reference_satisfies_protocol(self) -> None:
        assert isinstance(ReferenceSearchPipeline(), SearchPipeline)

    @requires_shrike_native
    def test_factory_builds_the_native_pipeline(self) -> None:
        assert isinstance(make_search_pipeline(), NativeSearchPipeline)

    def test_reference_matches_bare_rrf_fuse(self) -> None:
        rankings = {"text": [1, 2, 3], "exact": [3]}
        assert ReferenceSearchPipeline().fuse(
            rankings, priority_signals=frozenset({"exact"})
        ) == rrf_fuse(rankings, priority_signals=frozenset({"exact"}))


@requires_shrike_native
class TestNativeParity:
    def test_native_satisfies_protocol(self) -> None:
        assert isinstance(NativeSearchPipeline(), SearchPipeline)

    def test_property_native_equals_reference(self) -> None:
        reference = ReferenceSearchPipeline()
        native = NativeSearchPipeline()
        rng = random.Random(0xC0FFEE)
        for case in range(300):
            rankings, weights, priority = _random_case(rng)
            ref = reference.fuse(rankings, weights=weights, priority_signals=priority)
            nat = native.fuse(rankings, weights=weights, priority_signals=priority)
            assert len(ref) == len(nat), f"case {case}: length differs"
            for r, n in zip(ref, nat, strict=True):
                assert r.note_id == n.note_id, f"case {case}: order differs"
                # Same f64 ops in the same canonical order → bit-identical scores.
                assert r.score == n.score, f"case {case}: score differs"
                assert r.signals == n.signals, f"case {case}: provenance differs"

    def test_custom_k_carries(self) -> None:
        rankings = {"text": [1, 2], "exact": [2]}
        ref = ReferenceSearchPipeline().fuse(rankings, k=7)
        nat = NativeSearchPipeline().fuse(rankings, k=7)
        assert [(h.note_id, h.score) for h in ref] == [(h.note_id, h.score) for h in nat]
        assert RRF_K == 60  # the default the tools layer relies on
