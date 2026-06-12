//! Reciprocal Rank Fusion — the kernel's fusion stage (#274, moved here in
//! #380 so the kernel stopped naming shrike-compute, whose other half dragged
//! the whole ort engine stack into the link; the crate itself dissolved in
//! #443).
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

/// The CANONICAL per-signal RRF weights for fused search (#388): `fuzzy`
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
        let w = weights.get(signal).copied().unwrap_or(1.0);
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
        // total_cmp, not partial_cmp().expect(): a NaN weight from config must
        // not panic the actor (#382) — identical ordering for finite scores.
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
        // #382: a NaN weight from config poisons the scores; the sort must
        // stay total (no partial_cmp panic in the actor), output deterministic.
        let mut weights = BTreeMap::new();
        weights.insert("text".to_string(), f64::NAN);
        let rankings = vec![
            ("text".to_string(), vec![1, 2]),
            ("exact".to_string(), vec![2]),
        ];
        let hits = rrf_fuse(&rankings, &weights, RRF_K, &HashSet::new());
        assert_eq!(hits.len(), 2);
    }
}
