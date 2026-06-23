//! Reciprocal Rank Fusion — the kernel's fusion stage.
//!
//! [`rrf_fuse`] is a byte-faithful port of `shrike/search_fusion.py` (the
//! frozen reference spec): same canonical sorted-signal accumulation order
//! (float addition isn't associative), same dedup-at-best-rank, same
//! `(tier, -score, note_id)` ordering with the priority-signal tier. The
//! Python parity property suite pins native == reference on identical
//! inputs — the drift alarm that justifies the dual implementation.

use std::collections::{BTreeMap, HashSet};

/// Mirrors `shrike.search_fusion.RRF_K`.
pub const RRF_K: i64 = 60;

/// The signal whose literal hits tier above the rest (the one place RRF's
/// blindness to magnitude is wrong — see `search_fusion.py`).
pub const PRIORITY_SIGNAL: &str = "exact";

/// The CANONICAL per-signal RRF weights for fused search: `fuzzy` below the
/// rest — a near-miss is weaker evidence than a literal or semantic hit. The
/// single source of truth; the action defaults to these when the host passes
/// none (the host parameter is an override seam for a future config knob).
pub fn search_weights() -> BTreeMap<String, f64> {
    [
        ("text", 1.0),
        ("image", 1.0),
        ("tag", 1.0),
        ("exact", 1.0),
        ("fuzzy", 0.5),
    ]
    .into_iter()
    .map(|(s, w)| (s.to_string(), w))
    .collect()
}

/// One fused hit: `(note_id, score, [(signal, 1-based rank)...])`. The signal
/// list is in canonical (sorted-signal) accumulation order, matching the
/// insertion order of the Python implementation's `signals` dict.
pub type FusedHit = (i64, f64, Vec<(String, i64)>);

/// Reciprocal Rank Fusion — see `search_fusion.py` for the full rationale.
pub fn rrf_fuse(
    rankings: &[(String, Vec<i64>)],
    weights: &BTreeMap<String, f64>,
    k: i64,
    priority_signals: &HashSet<String>,
) -> Vec<FusedHit> {
    let mut scores: BTreeMap<i64, f64> = BTreeMap::new();
    let mut contributions: BTreeMap<i64, Vec<(String, i64)>> = BTreeMap::new();

    // Canonical (sorted) signal order — float addition isn't associative, and
    // the Python reference accumulates in sorted(rankings) order.
    let mut ordered: Vec<&(String, Vec<i64>)> = rankings.iter().collect();
    ordered.sort_by(|a, b| a.0.cmp(&b.0));

    for (signal, ids) in ordered {
        // Sanitize a non-finite weight to the default (1.0) BEFORE it reaches
        // the score accumulation and the sort. A NaN weight poisons a note's
        // score to NaN, and the two impls order NaN scores differently —
        // Rust's `total_cmp` total-orders NaN, while the reference's Python sort
        // key (`-score`) leaves NaN comparisons false → input-order-dependent —
        // so a NaN weight would break the frozen-reference parity contract. A
        // non-finite weight is meaningless as a scale, so both sides coerce it
        // to 1.0; finite-weight RRF (incl. 0.0 and negatives) is untouched.
        // `search_fusion.py` applies the identical coercion.
        let w = weights.get(signal).copied().unwrap_or(1.0);
        let w = if w.is_finite() { w } else { 1.0 };
        let mut seen: HashSet<i64> = HashSet::new();
        for (pos, note_id) in ids.iter().enumerate() {
            if !seen.insert(*note_id) {
                continue; // one signal, one rank per note (its best)
            }
            let rank = (pos + 1) as i64;
            *scores.entry(*note_id).or_insert(0.0) += w / (k + rank) as f64;
            contributions
                .entry(*note_id)
                .or_default()
                .push((signal.clone(), rank));
        }
    }

    let mut hits: Vec<FusedHit> = scores
        .iter()
        .map(|(nid, score)| (*nid, *score, contributions.remove(nid).unwrap_or_default()))
        .collect();
    hits.sort_by(|a, b| {
        let tier_a = i32::from(!a.2.iter().any(|(s, _)| priority_signals.contains(s)));
        let tier_b = i32::from(!b.2.iter().any(|(s, _)| priority_signals.contains(s)));
        // total_cmp, not partial_cmp().expect(): a defensive total order so the
        // sort can never panic the actor. Non-finite weights are
        // sanitized above, so scores are finite for any finite ranking
        // input; this is the belt-and-suspenders for any residual non-finite
        // score, and matches the reference's deterministic ordering on finite
        // scores. Reference parity holds because no NaN reaches this sort.
        tier_a
            .cmp(&tier_b)
            .then(b.1.total_cmp(&a.1))
            .then(a.0.cmp(&b.0))
    });
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn fuse_simple(rankings: &[(&str, &[i64])]) -> Vec<FusedHit> {
        let owned: Vec<(String, Vec<i64>)> = rankings
            .iter()
            .map(|(s, ids)| (s.to_string(), ids.to_vec()))
            .collect();
        rrf_fuse(&owned, &BTreeMap::new(), RRF_K, &HashSet::new())
    }

    #[test]
    fn single_signal_preserves_order() {
        let hits = fuse_simple(&[("text", &[3, 1, 2])]);
        let ids: Vec<i64> = hits.iter().map(|h| h.0).collect();
        assert_eq!(ids, vec![3, 1, 2]);
    }

    #[test]
    fn duplicate_in_one_signal_counts_once_at_best_rank() {
        let hits = fuse_simple(&[("text", &[5, 5, 6])]);
        assert_eq!(hits[0].0, 5);
        assert_eq!(hits[0].2, vec![("text".to_string(), 1)]);
    }

    #[test]
    fn priority_signal_floats_above_higher_scores() {
        let mut priority = HashSet::new();
        priority.insert("exact".to_string());
        let rankings = vec![
            ("text".to_string(), vec![1, 2, 3]),
            ("exact".to_string(), vec![3]),
        ];
        let hits = rrf_fuse(&rankings, &BTreeMap::new(), RRF_K, &priority);
        assert_eq!(hits[0].0, 3); // exact tier wins despite worse text rank
    }

    #[test]
    fn ties_break_by_note_id() {
        let hits = fuse_simple(&[("a", &[2]), ("b", &[1])]);
        // Equal scores (rank 1 in one signal each) → ascending note_id.
        assert_eq!(hits[0].0, 1);
        assert_eq!(hits[1].0, 2);
    }

    #[test]
    fn nan_weight_does_not_panic() {
        // A NaN weight from config must not panic the actor and must be
        // deterministic. It is sanitized to the default before the
        // sort, so it produces a finite, well-ordered result.
        let mut weights = BTreeMap::new();
        weights.insert("text".to_string(), f64::NAN);
        let rankings = vec![
            ("text".to_string(), vec![1, 2]),
            ("exact".to_string(), vec![2]),
        ];
        let hits = rrf_fuse(&rankings, &weights, RRF_K, &HashSet::new());
        assert_eq!(hits.len(), 2);
        for h in &hits {
            assert!(
                h.1.is_finite(),
                "a sanitized NaN weight must not poison the score"
            );
        }
    }

    #[test]
    fn non_finite_weight_is_sanitized_to_the_default_order() {
        // A NaN/inf/-inf weight is coerced to the default (1.0) before the
        // sort, so the fused ORDER is identical to the all-default-weights run.
        // This is the property the frozen-reference parity contract depends on
        // (the reference, search_fusion.py, applies the same coercion). Shape:
        // rankings {text:[1,2,3], exact:[3,2]}, priority
        // {exact}, weight {text: <non-finite>} → must equal the default order.
        let rankings = vec![
            ("text".to_string(), vec![1, 2, 3]),
            ("exact".to_string(), vec![3, 2]),
        ];
        let mut priority = HashSet::new();
        priority.insert("exact".to_string());

        let baseline = rrf_fuse(&rankings, &BTreeMap::new(), RRF_K, &priority);
        let baseline_ids: Vec<i64> = baseline.iter().map(|h| h.0).collect();

        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut weights = BTreeMap::new();
            weights.insert("text".to_string(), bad);
            let hits = rrf_fuse(&rankings, &weights, RRF_K, &priority);
            let ids: Vec<i64> = hits.iter().map(|h| h.0).collect();
            assert_eq!(
                ids, baseline_ids,
                "a non-finite ({bad}) weight must order like the default (1.0)"
            );
            for h in &hits {
                assert!(h.1.is_finite(), "sanitized weight keeps scores finite");
            }
        }
    }

    #[test]
    fn finite_weights_are_unchanged_by_the_sanitizer() {
        // The sanitizer guard must not touch finite weights — including 0.0 and a
        // negative weight, which are meaningful scales and must flow through.
        let rankings = vec![
            ("text".to_string(), vec![1, 2, 3]),
            ("fuzzy".to_string(), vec![3, 2, 1]),
        ];
        let mut weights = BTreeMap::new();
        weights.insert("text".to_string(), 2.0);
        weights.insert("fuzzy".to_string(), 0.5);
        let hits = rrf_fuse(&rankings, &weights, RRF_K, &HashSet::new());
        // text-weighted: note 1 (text rank 1) leads; every score finite.
        assert_eq!(hits[0].0, 1);
        for h in &hits {
            assert!(h.1.is_finite());
        }
    }

    // ====================================================================
    // Generative differential suite. RRF is a frozen-reference port of
    // `search_fusion.py`; the Python parity suite pins native==reference. These
    // pin the spec's INVARIANTS in-Rust over thousands of generated inputs —
    // order-invariance, the score formula, the global sort contract, dedup, and
    // weight sanitization — so a refactor that drifts from the spec fails here,
    // on the per-PR cargo lane, before the cross-language suite runs.
    // ====================================================================

    const SIGNALS: [&str; 5] = ["text", "image", "tag", "exact", "fuzzy"];

    /// A random fusion input: signals drawn from a small alphabet (so signal
    /// sets overlap), each a list of note ids from a small id space (so notes
    /// recur across AND within signals — exercising dedup), with occasional
    /// empty id lists. The dedup-by-signal-name (one entry per signal, the
    /// caller's contract) is applied after generation.
    fn rankings_strategy() -> impl Strategy<Value = Vec<(String, Vec<i64>)>> {
        // ids in 1..=6 so notes recur across AND within signals; 0..8 lengths
        // include the empty id-list edge; 0..6 signal slots include the
        // empty-input edge and let signal names collide for the dedup contract.
        let signal = prop::sample::select(SIGNALS.as_slice());
        let ids = prop::collection::vec(1_i64..=6, 0..8);
        prop::collection::vec((signal, ids), 0..6).prop_map(|entries| {
            let mut used: HashSet<&str> = HashSet::new();
            entries
                .into_iter()
                .filter(|(sig, _)| used.insert(sig))
                .map(|(sig, ids)| (sig.to_string(), ids))
                .collect()
        })
    }

    /// Per-signal weights: each signal is unlisted (defaults to 1.0), or carries
    /// 0.0, a positive scale, a negative scale, or exactly 1.0 — the finite-weight
    /// space the sanitizer must pass through untouched.
    fn weights_strategy() -> impl Strategy<Value = BTreeMap<String, f64>> {
        let per_signal = prop_oneof![
            Just(None),
            Just(Some(0.0_f64)),
            (0_u32..=400).prop_map(|n| Some(f64::from(n) / 100.0)),
            (0_u32..=200).prop_map(|n| Some(-(f64::from(n) / 100.0))),
            Just(Some(1.0_f64)),
        ];
        prop::collection::vec(per_signal, SIGNALS.len()).prop_map(|choices| {
            SIGNALS
                .iter()
                .zip(choices)
                .filter_map(|(sig, w)| w.map(|w| (sig.to_string(), w)))
                .collect()
        })
    }

    /// A non-finite f64 — the value the sanitizer must coerce to 1.0.
    fn non_finite_strategy() -> impl Strategy<Value = f64> {
        prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY),]
    }

    /// The independent oracle: per note, the (signal, best-rank) contributions
    /// in canonical sorted-signal order, and the score summed in that SAME order
    /// (float addition isn't associative — matching the order makes the score
    /// bit-comparable, not just approximately equal). Built by a different
    /// traversal than `rrf_fuse` (scan-then-group), so agreement is a real
    /// cross-check, not a tautology.
    fn oracle(
        rankings: &[(String, Vec<i64>)],
        weights: &BTreeMap<String, f64>,
        k: i64,
        priority: &HashSet<String>,
    ) -> Vec<FusedHit> {
        let mut ordered: Vec<&(String, Vec<i64>)> = rankings.iter().collect();
        ordered.sort_by(|a, b| a.0.cmp(&b.0));
        let mut contribs: BTreeMap<i64, Vec<(String, i64)>> = BTreeMap::new();
        for (sig, ids) in &ordered {
            let mut best: BTreeMap<i64, i64> = BTreeMap::new();
            for (pos, id) in ids.iter().enumerate() {
                best.entry(*id).or_insert(pos as i64 + 1); // first occurrence = best
            }
            // Re-walk ids in order to preserve first-seen note ordering within
            // the signal's contribution push (matches rrf_fuse).
            let mut pushed = HashSet::new();
            for id in ids.iter() {
                if pushed.insert(*id) {
                    contribs
                        .entry(*id)
                        .or_default()
                        .push((sig.clone(), best[id]));
                }
            }
        }
        let mut hits: Vec<FusedHit> = contribs
            .into_iter()
            .map(|(id, cs)| {
                let mut score = 0.0_f64;
                for (sig, rank) in &cs {
                    let w = weights.get(sig).copied().unwrap_or(1.0);
                    let w = if w.is_finite() { w } else { 1.0 };
                    score += w / (k + rank) as f64;
                }
                (id, score, cs)
            })
            .collect();
        hits.sort_by(|a, b| {
            let ta = i32::from(!a.2.iter().any(|(s, _)| priority.contains(s)));
            let tb = i32::from(!b.2.iter().any(|(s, _)| priority.contains(s)));
            ta.cmp(&tb).then(b.1.total_cmp(&a.1)).then(a.0.cmp(&b.0))
        });
        hits
    }

    /// A priority set: either `{exact}` or empty.
    fn priority_strategy() -> impl Strategy<Value = HashSet<String>> {
        prop::bool::ANY.prop_map(|on| {
            if on {
                [PRIORITY_SIGNAL.to_string()].into_iter().collect()
            } else {
                HashSet::new()
            }
        })
    }

    proptest! {
        #[test]
        fn matches_independent_oracle_bit_for_bit(
            rankings in rankings_strategy(),
            weights in weights_strategy(),
            priority in priority_strategy(),
        ) {
            let got = rrf_fuse(&rankings, &weights, RRF_K, &priority);
            let want = oracle(&rankings, &weights, RRF_K, &priority);
            prop_assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(want.iter()) {
                prop_assert_eq!(g.0, w.0, "note id");
                prop_assert_eq!(g.1.to_bits(), w.1.to_bits(), "score (bit-exact)");
                prop_assert_eq!(&g.2, &w.2, "contributions (signal, rank)");
            }
        }

        /// rrf_fuse sorts signals canonically before accumulating, so the order
        /// the host hands signals in must not change the result at all — ids,
        /// scores, AND contribution lists. This is what makes float-accumulation
        /// deterministic across callers.
        #[test]
        fn fused_output_is_invariant_to_input_signal_order(
            rankings in rankings_strategy(),
            weights in weights_strategy(),
        ) {
            let priority: HashSet<String> = [PRIORITY_SIGNAL.to_string()].into_iter().collect();
            let base = rrf_fuse(&rankings, &weights, RRF_K, &priority);
            // Reverse, then a rotation — two non-trivial permutations.
            let mut shuffled = rankings.clone();
            shuffled.reverse();
            if shuffled.len() > 2 {
                shuffled.rotate_left(1);
            }
            let other = rrf_fuse(&shuffled, &weights, RRF_K, &priority);
            prop_assert_eq!(base.len(), other.len());
            for (a, b) in base.iter().zip(other.iter()) {
                prop_assert_eq!(a.0, b.0);
                prop_assert_eq!(a.1.to_bits(), b.1.to_bits());
                prop_assert_eq!(&a.2, &b.2);
            }
        }

        /// The global sort contract: priority-tier ascending, then score
        /// descending, then note id ascending — checked pairwise across the whole
        /// result for random inputs (a sort bug shows as an out-of-order pair).
        #[test]
        fn output_respects_the_tier_score_id_total_order(
            rankings in rankings_strategy(),
            weights in weights_strategy(),
        ) {
            let priority: HashSet<String> = [PRIORITY_SIGNAL.to_string()].into_iter().collect();
            let hits = rrf_fuse(&rankings, &weights, RRF_K, &priority);
            let tier = |h: &FusedHit| i32::from(!h.2.iter().any(|(s, _)| priority.contains(s)));
            for pair in hits.windows(2) {
                let (a, b) = (&pair[0], &pair[1]);
                let (ta, tb) = (tier(a), tier(b));
                prop_assert!(ta <= tb, "tier order violated: {:?} before {:?}", a, b);
                if ta == tb {
                    // within a tier: score non-increasing, ties by ascending id
                    prop_assert!(
                        a.1 > b.1 || (a.1.to_bits() == b.1.to_bits() && a.0 < b.0),
                        "within-tier order violated: {:?} before {:?}",
                        a,
                        b
                    );
                }
            }
        }

        /// Completeness + dedup-across-signals: the result set is exactly the
        /// union of note ids across all signals, each once.
        #[test]
        fn every_input_note_appears_exactly_once(
            rankings in rankings_strategy(),
            weights in weights_strategy(),
        ) {
            let hits = rrf_fuse(&rankings, &weights, RRF_K, &HashSet::new());
            let mut expected: HashSet<i64> = HashSet::new();
            for (_, ids) in &rankings {
                expected.extend(ids.iter().copied());
            }
            let got: Vec<i64> = hits.iter().map(|h| h.0).collect();
            let got_set: HashSet<i64> = got.iter().copied().collect();
            prop_assert_eq!(got.len(), got_set.len(), "a note appeared twice");
            prop_assert_eq!(got_set, expected, "result set != union of inputs");
        }

        /// The frozen-reference parity hinge, generatively: replacing any signal's
        /// weight with NaN/±inf yields the IDENTICAL fused output as weight 1.0.
        #[test]
        fn non_finite_weight_orders_like_one_generatively(
            rankings in rankings_strategy(),
            target in prop::sample::select(SIGNALS.as_slice()),
            bad in non_finite_strategy(),
        ) {
            let priority: HashSet<String> = [PRIORITY_SIGNAL.to_string()].into_iter().collect();

            let mut w_bad = BTreeMap::new();
            w_bad.insert(target.to_string(), bad);
            let mut w_one = BTreeMap::new();
            w_one.insert(target.to_string(), 1.0);

            let got = rrf_fuse(&rankings, &w_bad, RRF_K, &priority);
            let want = rrf_fuse(&rankings, &w_one, RRF_K, &priority);
            prop_assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(want.iter()) {
                prop_assert_eq!(g.0, w.0);
                prop_assert_eq!(g.1.to_bits(), w.1.to_bits());
                prop_assert_eq!(&g.2, &w.2);
            }
        }
    }
}
