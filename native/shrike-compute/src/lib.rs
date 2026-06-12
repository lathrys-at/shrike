//! Fused search/upsert composition (#274).
//!
//! [`fused_search`] / [`fused_add`] — the embed→index hot paths composed
//! natively so **embedding vectors never cross the FFI**: one GIL-released
//! call embeds the texts and searches/adds them against the index engine.
//! Lexical signals (the #98 substring authority + fuzzy) enter `search` as
//! pre-ranked lists Python-side, per the issue's contract — native-internal
//! lexical is an implementation detail a later consolidation may adopt.
//!
//! `rrf_fuse` no longer lives here: it moved to `shrike_kernel::fusion`
//! (#380) so the kernel stops linking this crate's concrete embedder stack.
//! The fused paths remain typed against `shrike_embed::TextEmbedder` for
//! their only caller — the standalone Python facade — until #355 retires
//! them.

use std::collections::BTreeMap;

use shrike_embed::TextEmbedder;
use shrike_ffi::NativeResult;
use shrike_index::{ModalityRanking, MultiModalIndex};

/// Embed query texts and rank them per modality against the index — one
/// GIL-released composition; the query vectors never leave native code.
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
