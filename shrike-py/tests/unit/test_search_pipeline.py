"""SearchPipeline seam: protocol conformance + native==reference parity.

`search_fusion.rrf_fuse` is the frozen, readable spec of the fusion semantics;
the native pipeline (`shrike_kernel::fusion::rrf_fuse`) is a second implementation of
the same `SearchPipeline` protocol. The property suite here is the drift alarm
that justifies the dual implementation: randomized rankings/weights/priority
tiers in → identical fused order, scores, and provenance out.
"""

from __future__ import annotations

import importlib.util
import random

import pytest

from shrike.harness.search_fusion import (
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
    # Include non-finite weights in the sample: they must be sanitized
    # identically on both sides, so the property suite is the drift alarm for
    # the NaN/inf parity too.
    weight_choices = [0.25, 0.5, 1.0, 2.0, float("nan"), float("inf"), float("-inf")]
    weights = {s: rng.choice(weight_choices) for s in rankings if rng.random() < 0.7}
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

    def test_reference_sanitizes_non_finite_weight_to_default(self) -> None:
        # The reference half (no native extension needed): a non-finite
        # weight orders AND scores identically to the default-weight run, so a
        # NaN/inf weight can't reach the sort with a divergent ordering.
        rankings = {"text": [1, 2, 3], "exact": [3, 2]}
        priority = frozenset({"exact"})
        default = ReferenceSearchPipeline().fuse(rankings, priority_signals=priority)
        for bad in (float("nan"), float("inf"), float("-inf")):
            got = ReferenceSearchPipeline().fuse(
                rankings, weights={"text": bad}, priority_signals=priority
            )
            assert [h.note_id for h in got] == [h.note_id for h in default]
            assert all(h.score == d.score for h, d in zip(got, default, strict=True)), (
                f"a non-finite ({bad}) weight must score like the default"
            )


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

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_native_equals_reference_on_a_non_finite_weight(self, bad: float) -> None:
        # A non-finite weight must not break the frozen-reference parity
        # contract. Rust total_cmp total-orders NaN while the Python sort key
        # leaves NaN comparisons false (input-order-dependent), so an unhandled
        # NaN weight could diverge. Both sides sanitize a non-finite weight to
        # the default (1.0), so the fused order is identical AND equals the
        # all-default-weights order.
        rankings = {"text": [1, 2, 3], "exact": [3, 2]}
        weights = {"text": bad}
        priority = frozenset({"exact"})
        ref = [
            h.note_id
            for h in ReferenceSearchPipeline().fuse(
                rankings, weights=weights, priority_signals=priority
            )
        ]
        nat = [
            h.note_id
            for h in NativeSearchPipeline().fuse(
                rankings, weights=weights, priority_signals=priority
            )
        ]
        assert ref == nat, f"parity broken on {bad} weight: ref={ref} native={nat}"
        # The sanitizer coerces to the default, so it equals the no-weights run.
        default = [
            h.note_id for h in ReferenceSearchPipeline().fuse(rankings, priority_signals=priority)
        ]
        assert ref == default, f"non-finite weight must order like default: {ref} != {default}"
