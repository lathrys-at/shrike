"""S9-1 repro (preserved by lead; rev-S9 worktree reaped).
rrf_fuse native diverges from the frozen search_fusion.py reference on a NaN
weight (Rust total_cmp imposes a total order on NaN; Python sort key NaN compares
False → input-order-dependent). RED at fa54f8c: ref=[3,2,1] native=[2,3,1].
Run: SHRIKE_SKIP_NATIVE_STALE_CHECK=1 .venv/bin/python -m pytest <this> -q -p no:cacheprovider
Fix: validate/sanitize non-finite weights before rrf_fuse (preferred — neither
order is meaningful), OR make the reference's NaN ordering match total_cmp.
"""
from shrike.search_fusion import ReferenceSearchPipeline, NativeSearchPipeline


def test_native_equals_reference_on_nan_weight() -> None:
    ref = ReferenceSearchPipeline()
    nat = NativeSearchPipeline()
    rankings = {"text": [1, 2, 3], "exact": [3, 2]}
    weights = {"text": float("nan")}
    priority = frozenset({"exact"})
    r = [h.note_id for h in ref.fuse(rankings, weights=weights, priority_signals=priority)]
    n = [h.note_id for h in nat.fuse(rankings, weights=weights, priority_signals=priority)]
    assert r == n, f"parity broken on NaN weight: ref={r} native={n}"
