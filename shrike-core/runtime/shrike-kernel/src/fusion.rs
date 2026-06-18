//! Reciprocal Rank Fusion — the kernel's fusion stage (moved here so the
//! kernel stopped naming shrike-compute, whose other half dragged the whole
//! ort engine stack into the link; the crate itself has since dissolved).
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

/// The CANONICAL per-signal RRF weights for fused search: `fuzzy`
/// below the rest — a near-miss is weaker evidence than a literal or
/// semantic hit. The single source of truth; the action defaults to these
/// when the host passes none (the host parameter stays as an override
/// seam for a future config knob).
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
        // the score accumulation and the sort. A NaN weight poisons a
        // note's score to NaN, and the two impls order NaN scores differently —
        // Rust's `total_cmp` total-orders NaN, while the reference's Python sort
        // key (`-score`) leaves NaN comparisons false → input-order-dependent —
        // so a NaN weight broke the frozen-reference parity contract. A
        // non-finite weight is meaningless as a scale, so both sides coerce it
        // to 1.0; finite-weight RRF (incl. 0.0 and negatives) is unchanged.
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
}
