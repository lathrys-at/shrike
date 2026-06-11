//! Fused search/upsert composition (#274).
//!
//! Two things live here:
//!
//! 1. [`rrf_fuse`] — a byte-faithful port of `shrike/search_fusion.py` (the
//!    frozen reference spec): same canonical sorted-signal accumulation order
//!    (float addition isn't associative), same dedup-at-best-rank, same
//!    `(tier, -score, note_id)` ordering with the priority-signal tier. The
//!    Python parity property suite pins native == reference on identical
//!    inputs — the drift alarm that justifies the dual implementation.
//!
//! 2. [`fused_search`] / [`fused_add`] — the embed→index hot paths composed
//!    natively so **embedding vectors never cross the FFI**: one GIL-released
//!    call embeds the texts and searches/adds them against the index engine.
//!    Lexical signals (the #98 substring authority + fuzzy) enter `search` as
//!    pre-ranked lists Python-side, per the issue's contract — native-internal
//!    lexical is an implementation detail a later consolidation may adopt.

use std::collections::{BTreeMap, HashSet};

#[cfg(feature = "fused")]
use shrike_embed::TextEmbedder;
#[cfg(feature = "fused")]
use shrike_ffi::NativeResult;
#[cfg(feature = "fused")]
use shrike_index::{ModalityRanking, MultiModalIndex};

/// Mirrors `shrike.search_fusion.RRF_K`.
pub const RRF_K: i64 = 60;

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
        tier_a
            .cmp(&tier_b)
            .then(b.1.partial_cmp(&a.1).expect("rrf scores are finite"))
            .then(a.0.cmp(&b.0))
    });
    hits
}

/// Embed query texts and rank them per modality against the index — one
/// GIL-released composition; the query vectors never leave native code.
#[cfg(feature = "fused")]
pub fn fused_search(
    embedder: &TextEmbedder,
    index: &MultiModalIndex,
    texts: &[String],
    k: usize,
    modalities: Option<&[String]>,
) -> NativeResult<Vec<BTreeMap<String, ModalityRanking>>> {
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    // Embed serially per the facade's probed chunking contract? No — the probe
    // governs the *embed batch*; here each text is one query. Use one chunk per
    // text to match the serial-equals-batched guarantee regardless of model.
    for text in texts {
        let mut v = embedder.embed_chunk(std::slice::from_ref(text))?;
        vectors.append(&mut v);
    }
    index.search_by_modality(&vectors, k, modalities)
}

/// Embed note texts and (replace-)add them under their note ids — one
/// GIL-released composition; the note vectors never leave native code.
/// Returns the count added.
#[cfg(feature = "fused")]
pub fn fused_add(
    embedder: &TextEmbedder,
    index: &MultiModalIndex,
    modality: &str,
    keys: &[i64],
    texts: &[String],
    chunk: usize,
) -> NativeResult<usize> {
    if keys.len() != texts.len() {
        return Err(shrike_ffi::NativeError::invalid_input(format!(
            "keys ({}) and texts ({}) must align",
            keys.len(),
            texts.len()
        )));
    }
    let chunk = chunk.max(1);
    let mut added = 0usize;
    for (key_chunk, text_chunk) in keys.chunks(chunk).zip(texts.chunks(chunk)) {
        let vectors = embedder.embed_chunk(text_chunk)?;
        // Replace semantics: drop the notes' existing vectors first (all
        // modalities), exactly like the orchestrator's add path.
        index.remove(key_chunk)?;
        index.add(modality, key_chunk, &vectors)?;
        added += key_chunk.len();
    }
    Ok(added)
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
}
