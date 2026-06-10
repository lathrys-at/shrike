//! The pure-Rust kernel (#279, slice 2 — PR 1: the no-CPython keystone).
//!
//! This crate composes the native compute plane into the embedded-host shape
//! #224 specs: it owns the collection core (anki via its protobuf service
//! layer, #278), the vector index engine, the derived-text store, the fusion,
//! and **its own threading** — a dedicated collection-owning thread with a
//! FIFO channel, exactly the #224 invariant (the collection is opened *on* its
//! owning thread; no live handle crosses; a collection job never waits on
//! another collection job).
//!
//! There is **no pyo3 anywhere in this dependency tree** (epic #265 convention
//! 5, enforced by `//native:layering_check`); the no-CPython smoke test in
//! this crate links the kernel without Python and runs open → upsert → search
//! (a semantic and a lexical signal both contributing) → close — the #279
//! acceptance's executable form.
//!
//! Slice-2 series (recorded on #279): this keystone first; then the kernel's
//! action core + Rust-canonical schemas with the Pydantic contract test; then
//! the Python harness rebased onto `shrike-py` kernel bindings, retiring the
//! transitional Python schedulers from #275.

use std::collections::BTreeMap;
use std::sync::mpsc;
use std::thread::JoinHandle;

use shrike_collection::{CollectionCore, CreateOutcome, DuplicatePolicy};
use shrike_derived::DerivedEngine;
use shrike_ffi::{NativeError, NativeResult};
use shrike_index::MultiModalIndex;

/// The embedder seam the kernel needs — the Rust counterpart of the Python
/// `EmbedderBackend` protocol's compute slice. `shrike_embed::TextEmbedder`
/// satisfies it for real models; tests use a deterministic stub.
pub trait Embedder: Send + 'static {
    fn embed(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>>;
}

impl Embedder for shrike_embed::TextEmbedder {
    fn embed(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>> {
        self.embed_chunk(texts)
    }
}

type CollectionJob = Box<dyn FnOnce(&CollectionCore) + Send>;

/// The collection's owning thread: jobs run FIFO, the core never leaves it.
struct CollectionThread {
    sender: Option<mpsc::Sender<CollectionJob>>,
    handle: Option<JoinHandle<()>>,
}

impl CollectionThread {
    fn spawn(collection_path: String) -> NativeResult<Self> {
        let (sender, receiver) = mpsc::channel::<CollectionJob>();
        let (ready_tx, ready_rx) = mpsc::channel::<NativeResult<()>>();
        let handle = std::thread::Builder::new()
            .name("shrike-kernel-collection".into())
            .spawn(move || {
                // Open ON the owning thread (#224's invariant).
                let core = match CollectionCore::open(&collection_path) {
                    Ok(core) => {
                        let _ = ready_tx.send(Ok(()));
                        core
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                while let Ok(job) = receiver.recv() {
                    job(&core);
                }
                let _ = core.close();
            })
            .map_err(|e| NativeError::internal(format!("collection thread: {e}")))?;
        ready_rx
            .recv()
            .map_err(|_| NativeError::internal("collection thread died during open"))??;
        Ok(Self {
            sender: Some(sender),
            handle: Some(handle),
        })
    }

    /// Run a job on the owning thread, blocking for its result.
    fn run<T: Send + 'static>(
        &self,
        job: impl FnOnce(&CollectionCore) -> T + Send + 'static,
    ) -> NativeResult<T> {
        let (tx, rx) = mpsc::channel();
        let sender = self
            .sender
            .as_ref()
            .ok_or_else(|| NativeError::unavailable("kernel is closed"))?;
        sender
            .send(Box::new(move |core| {
                let _ = tx.send(job(core));
            }))
            .map_err(|_| NativeError::unavailable("collection thread gone"))?;
        rx.recv()
            .map_err(|_| NativeError::internal("collection job lost"))
    }

    fn close(&mut self) {
        self.sender.take(); // channel closes → thread drains, closes the core, exits
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// One fused search hit: note id, fused score, per-signal 1-based ranks.
#[derive(Debug, Clone)]
pub struct KernelHit {
    pub note_id: i64,
    pub score: f64,
    pub signals: Vec<(String, i64)>,
}

/// The kernel: one open collection + the derived/vector stores + fusion,
/// behind its own threading. No transport, no Python.
pub struct Kernel<E: Embedder> {
    collection: CollectionThread,
    index: MultiModalIndex,
    derived: DerivedEngine,
    embedder: E,
}

const TEXT: &str = "text";
const FIELD_SOURCE: &str = "field";

impl<E: Embedder> Kernel<E> {
    /// Open a collection and its sidecar stores (cache_dir holds the derived
    /// store, like the Python host's cache layout).
    pub fn open(collection_path: &str, cache_dir: &str, embedder: E) -> NativeResult<Self> {
        std::fs::create_dir_all(cache_dir)
            .map_err(|e| NativeError::internal(format!("cache dir: {e}")))?;
        let collection = CollectionThread::spawn(collection_path.to_string())?;
        let index = MultiModalIndex::new(vec![TEXT.to_string(), "image".to_string()])?;
        let derived =
            DerivedEngine::open(&format!("{}/shrike.db", cache_dir.trim_end_matches('/')), 1)?;
        Ok(Self {
            collection,
            index,
            derived,
            embedder,
        })
    }

    pub fn col_mod(&self) -> NativeResult<i64> {
        self.collection.run(|core| core.col_mod())?
    }

    pub fn notetype_id(&self, name: &str) -> NativeResult<i64> {
        let name = name.to_string();
        self.collection.run(move |core| core.notetype_id(&name))?
    }

    /// Create a note (the #77 duplicate policy applies) and index it: the
    /// embedding + vector add and the derived-text ingest happen *off* the
    /// collection thread — compute never routes back through the queue.
    pub fn upsert_note(
        &self,
        notetype_id: i64,
        deck_id: i64,
        fields: Vec<String>,
        tags: Vec<String>,
        policy: DuplicatePolicy,
    ) -> NativeResult<CreateOutcome> {
        let text = fields.join(" ");
        let refs: Vec<(String, String)> = fields
            .iter()
            .enumerate()
            .map(|(i, value)| (format!("F{i}"), value.clone()))
            .collect();
        let outcome = self
            .collection
            .run(move |core| core.create_note(notetype_id, deck_id, &fields, &tags, policy))??;
        if let CreateOutcome::Created(note_id) = outcome {
            let vectors = self.embedder.embed(std::slice::from_ref(&text))?;
            self.index.remove(&[note_id])?;
            self.index.add(TEXT, &[note_id], &vectors)?;
            self.derived.ingest(note_id, FIELD_SOURCE, &refs)?;
            let col_mod = self.collection.run(|core| core.col_mod())??;
            self.derived.set_col_mod(col_mod)?;
        }
        Ok(outcome)
    }

    pub fn delete_notes(&self, note_ids: Vec<i64>) -> NativeResult<usize> {
        let ids = note_ids.clone();
        let removed = self.collection.run(move |core| core.delete_notes(&ids))??;
        self.index.remove(&note_ids)?;
        self.derived.remove(&note_ids, None)?;
        Ok(removed)
    }

    /// Fused search: the semantic ranking (embed → per-modality engine search)
    /// and the lexical rankings (the derived store's substring + fuzzy) each
    /// rank their own candidates; RRF blends them with the exact tier on top —
    /// the same semantics as the Python host's search_notes spine.
    pub fn search(&self, query: &str, top_k: usize) -> NativeResult<Vec<KernelHit>> {
        // Semantic signal.
        let qvec = self.embedder.embed(&[query.to_string()])?;
        let semantic = self.index.search_by_modality(&qvec, top_k, None)?;
        let mut rankings: Vec<(String, Vec<i64>)> = Vec::new();
        if let Some(per_query) = semantic.first() {
            for (modality, (ids, _dists)) in per_query {
                rankings.push((modality.clone(), ids.clone()));
            }
        }

        // Lexical signals (substring authority + fuzzy), from the derived store.
        let quoted = format!("\"{}\"", query.replace('"', "\"\""));
        let exact: Vec<i64> = self
            .derived
            .match_rows(&quoted, top_k as i64, false)?
            .into_iter()
            .map(|(nid, ..)| nid)
            .collect();
        rankings.push(("exact".to_string(), exact));

        let grams: Vec<String> = {
            let lower = query.to_lowercase();
            let chars: Vec<char> = lower.chars().collect();
            (0..chars.len().saturating_sub(2))
                .map(|i| chars[i..i + 3].iter().collect::<String>())
                .collect()
        };
        if grams.len() >= 2 {
            let expr = grams
                .iter()
                .map(|g| format!("\"{}\"", g.replace('"', "\"\"")))
                .collect::<Vec<_>>()
                .join(" OR ");
            let fuzzy: Vec<i64> = self
                .derived
                .match_rows(&expr, (top_k * 4) as i64, false)?
                .into_iter()
                .map(|(nid, ..)| nid)
                .collect();
            rankings.push(("fuzzy".to_string(), fuzzy));
        }

        // Fuse (the frozen #274 semantics: weights, exact tier, determinism).
        let mut weights = BTreeMap::new();
        weights.insert("text".to_string(), 1.0);
        weights.insert("image".to_string(), 1.0);
        weights.insert("exact".to_string(), 1.0);
        weights.insert("fuzzy".to_string(), 0.5);
        let mut priority = std::collections::HashSet::new();
        priority.insert("exact".to_string());
        let fused = shrike_compute::rrf_fuse(&rankings, &weights, shrike_compute::RRF_K, &priority);
        Ok(fused
            .into_iter()
            .take(top_k)
            .map(|(note_id, score, signals)| KernelHit {
                note_id,
                score,
                signals,
            })
            .collect())
    }

    pub fn close(mut self) -> NativeResult<()> {
        self.collection.close();
        Ok(())
    }
}

#[cfg(test)]
mod no_cpython_smoke {
    //! The #279 acceptance smoke: link the kernel WITHOUT Python and run one
    //! flow end to end — open → upsert (duplicate policy live) → search with a
    //! semantic AND a lexical signal contributing → delete → close. The
    //! layering check guarantees no pyo3 is below us; this test guarantees the
    //! composition actually works with zero CPython in the process.

    use super::*;

    /// Deterministic embedder: a token-hash bag vector. Similar texts share
    /// tokens → close vectors; no model, no network, no Python.
    struct HashEmbedder;

    impl Embedder for HashEmbedder {
        fn embed(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v = vec![0.0f32; 64];
                    for token in t.to_lowercase().split_whitespace() {
                        let mut h: u64 = 1469598103934665603;
                        for b in token.bytes() {
                            h ^= b as u64;
                            h = h.wrapping_mul(1099511628211);
                        }
                        v[(h % 64) as usize] += 1.0;
                    }
                    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                    v.iter().map(|x| x / norm).collect()
                })
                .collect())
        }
    }

    fn temp_dir() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "shrike-kernel-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn open_upsert_search_close_without_python() {
        let dir = temp_dir();
        let col = dir.join("collection.anki2");
        let cache = dir.join("cache");
        let kernel =
            Kernel::open(col.to_str().unwrap(), cache.to_str().unwrap(), HashEmbedder).unwrap();

        let basic = kernel.notetype_id("Basic").unwrap();
        let make = |front: &str, back: &str| {
            kernel
                .upsert_note(
                    basic,
                    1,
                    vec![front.to_string(), back.to_string()],
                    vec!["smoke".to_string()],
                    DuplicatePolicy::Error,
                )
                .unwrap()
        };
        let CreateOutcome::Created(mito) =
            make("the mitochondria powerhouse", "energy of the cell")
        else {
            panic!("create failed")
        };
        make("newton laws of motion", "classical mechanics");
        make("paris is the capital of france", "geography");

        // Duplicate policy is live end to end.
        let dup = kernel.upsert_note(
            basic,
            1,
            vec!["the mitochondria powerhouse".into(), "x".into()],
            vec![],
            DuplicatePolicy::Skip,
        );
        assert_eq!(dup.unwrap(), CreateOutcome::SkippedDuplicate);

        // Search: semantic + lexical signals both contribute to the winner.
        let hits = kernel.search("mitochondria powerhouse", 5).unwrap();
        assert_eq!(hits[0].note_id, mito);
        let signals: Vec<&str> = hits[0].signals.iter().map(|(s, _)| s.as_str()).collect();
        assert!(
            signals.contains(&"text"),
            "semantic signal missing: {signals:?}"
        );
        assert!(
            signals.contains(&"exact") || signals.contains(&"fuzzy"),
            "lexical signal missing: {signals:?}"
        );

        // A typo'd query still finds it through the fuzzy lexical signal.
        let fuzzy_hits = kernel.search("mitochondira powerhose", 5).unwrap();
        assert!(fuzzy_hits.iter().any(|h| h.note_id == mito));

        // Delete propagates to every store.
        assert_eq!(kernel.delete_notes(vec![mito]).unwrap(), 1);
        let after = kernel.search("mitochondria powerhouse", 5).unwrap();
        assert!(after.iter().all(|h| h.note_id != mito));

        kernel.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }
}
