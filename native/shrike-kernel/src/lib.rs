//! The pure-Rust kernel (#279, slice 2 — PR 1: the no-CPython keystone).
//!
//! This crate composes the native compute plane into the embedded-host shape
//! #224 specs: it owns the collection core (anki via its protobuf service
//! layer, #278), the vector index engine, the derived-text store, and the
//! fusion — and **no threading at all** (#308). The kernel never spawns a
//! thread and assumes nothing about the runtime (no tokio assumption; anki's
//! internal runtime is its own business behind the service layer). Scheduling
//! is *injected* by the harness through the [`SerialExecutor`] contract,
//! exactly as the harness plugs transports: collection ops need serialization
//! with respect to each other, not a dedicated thread — execution may migrate
//! across threads between jobs so long as the contract holds.
//!
//! **Natively async (#310):** every kernel op is an `async fn`; the
//! transitions between collection ops are awaits that chain/fan out through
//! the compute layer (embed → index add → derived ingest). Nothing here
//! blocks by assumption and nothing names a runtime — the futures are
//! runtime-agnostic, so the harness runs them on its executor of choice
//! (asyncio via pyo3-async-runtimes, a mobile runtime, or a plain block_on).
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
use std::sync::{Arc, Mutex};

use futures::channel::oneshot;
use futures::future::BoxFuture;
use tracing::Instrument;

use shrike_collection::{CollectionCore, CreateOutcome, DuplicatePolicy};
use shrike_derived::DerivedEngine;
use shrike_ffi::{NativeError, NativeResult};
use shrike_index::MultiModalIndex;

/// The embedder seam the kernel needs — the Rust counterpart of the Python
/// `EmbedderBackend` protocol's compute slice. `shrike_embed::TextEmbedder`
/// satisfies it for real models; tests use a deterministic stub.
pub trait Embedder: Send + Sync + 'static {
    /// Embed a batch. Async so a harness can supply a genuinely asynchronous
    /// embedder (a remote service, a platform ML API); CPU-bound embedders
    /// return a ready future.
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>>;
}

impl Embedder for shrike_embed::TextEmbedder {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        Box::pin(async move { self.embed_chunk(&texts) })
    }
}

/// The scheduling contract the harness injects (#308). The kernel never
/// spawns threads or assumes a runtime; whoever assembles a kernel supplies
/// this, exactly as it plugs transports.
///
/// **Contract:**
/// - Jobs submitted through one executor run **serialized FIFO** with respect
///   to each other — never concurrently. (The collection's consistency model
///   requires serialization, not thread affinity.)
/// - Execution may happen on any thread, and may **migrate across threads
///   between jobs** — anki's service layer is internally synchronized, so no
///   thread-affinity is required, only mutual exclusion + ordering.
/// - `submit` returns a **runtime-agnostic future** that resolves once the
///   job has run; the executor decides where/when (inline, a worker, a pool —
///   anything honoring serialization). The kernel never blocks on it; it
///   awaits.
/// - **Re-entrancy is forbidden**: a job must never submit to (and await)
///   its own executor from within itself — with any conforming implementation
///   that is a deadlock by contract, not an executor bug. Compute (embedding,
///   index, derived work) runs *outside* collection jobs for exactly this
///   reason.
pub trait SerialExecutor: Send + Sync {
    fn submit(&self, job: Box<dyn FnOnce() + Send + 'static>) -> BoxFuture<'static, ()>;
}

/// The simplest conforming executor: mutual exclusion on the calling thread.
/// Serialized (the mutex), thread-agnostic (runs wherever the caller is), no
/// threads owned. A real harness may instead pin a worker thread (the Python
/// host), use a thread pool with an ordered queue, or an actor — anything
/// honoring the contract.
#[derive(Default)]
pub struct MutexExecutor {
    gate: Mutex<()>,
}

impl SerialExecutor for MutexExecutor {
    fn submit(&self, job: Box<dyn FnOnce() + Send + 'static>) -> BoxFuture<'static, ()> {
        // Degenerate-but-conforming: run inline under the gate, return ready.
        // A real harness suspends instead (queue + wake) — the contract only
        // fixes serialization, not where the work happens.
        let _guard = self.gate.lock().expect("executor gate poisoned");
        job();
        Box::pin(async {})
    }
}

/// The collection behind the injected executor: every access is one submitted
/// job; the core never escapes. (CollectionCore is Send: anki's Backend is
/// internally synchronized, which is what makes thread migration safe.)
struct SerializedCollection {
    core: Arc<CollectionCore>,
    executor: Arc<dyn SerialExecutor>,
}

impl SerializedCollection {
    async fn open(
        collection_path: String,
        executor: Arc<dyn SerialExecutor>,
    ) -> NativeResult<Self> {
        // Open through the executor too: the open IS a collection op.
        let (tx, rx) = oneshot::channel();
        executor
            .submit(Box::new(move || {
                let _ = tx.send(CollectionCore::open(&collection_path));
            }))
            .await;
        let core = rx
            .await
            .map_err(|_| NativeError::internal("executor dropped the open job"))??;
        Ok(Self {
            core: Arc::new(core),
            executor,
        })
    }

    /// Run a job against the collection, serialized; the await IS the
    /// transition point ops chain continuations onto.
    async fn run<T: Send + 'static>(
        &self,
        job: impl FnOnce(&CollectionCore) -> T + Send + 'static,
    ) -> NativeResult<T> {
        let core = Arc::clone(&self.core);
        let (tx, rx) = oneshot::channel();
        self.executor
            .submit(Box::new(move || {
                let _ = tx.send(job(&core));
            }))
            .await;
        rx.await
            .map_err(|_| NativeError::internal("executor dropped a collection job"))
    }

    async fn close(&self) -> NativeResult<()> {
        self.run(|core| core.close()).await?
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
/// every op an async fn over the injected executor. No threads owned, no
/// runtime assumed, no transport, no Python.
pub struct Kernel<E: Embedder> {
    collection: SerializedCollection,
    index: MultiModalIndex,
    derived: DerivedEngine,
    embedder: E,
}

const TEXT: &str = "text";
const FIELD_SOURCE: &str = "field";

impl<E: Embedder> Kernel<E> {
    /// Open a collection and its sidecar stores (cache_dir holds the derived
    /// store, like the Python host's cache layout). `executor` is the
    /// harness-injected scheduling (see [`SerialExecutor`]).
    pub async fn open(
        collection_path: &str,
        cache_dir: &str,
        embedder: E,
        executor: Arc<dyn SerialExecutor>,
    ) -> NativeResult<Self> {
        std::fs::create_dir_all(cache_dir)
            .map_err(|e| NativeError::internal(format!("cache dir: {e}")))?;
        let collection = SerializedCollection::open(collection_path.to_string(), executor).await?;
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

    pub async fn col_mod(&self) -> NativeResult<i64> {
        self.collection.run(|core| core.col_mod()).await?
    }

    pub async fn notetype_id(&self, name: &str) -> NativeResult<i64> {
        let name = name.to_string();
        self.collection
            .run(move |core| core.notetype_id(&name))
            .await?
    }

    /// Create a note (the #77 duplicate policy applies) and index it: the
    /// embedding + vector add and the derived-text ingest happen *off* the
    /// collection thread — compute never routes back through the queue.
    pub async fn upsert_note(
        &self,
        notetype_id: i64,
        deck_id: i64,
        fields: Vec<String>,
        tags: Vec<String>,
        policy: DuplicatePolicy,
    ) -> NativeResult<CreateOutcome> {
        // `.instrument`, never an entered guard across an await: the span
        // follows the future across polls (and threads), and the future stays
        // Send — spawnable on any multithreaded runtime.
        let span = tracing::debug_span!("kernel.upsert_note", notetype_id, deck_id);
        async move {
            let text = fields.join(" ");
            let refs: Vec<(String, String)> = fields
                .iter()
                .enumerate()
                .map(|(i, value)| (format!("F{i}"), value.clone()))
                .collect();
            // The collection write, then the compute fan-out — each transition
            // an await, never a block: embed → index add → derived ingest →
            // watermark.
            let outcome = self
                .collection
                .run(move |core| core.create_note(notetype_id, deck_id, &fields, &tags, policy))
                .await??;
            if let CreateOutcome::Created(note_id) = outcome {
                let vectors = self.embedder.embed(vec![text]).await?;
                self.index.remove(&[note_id])?;
                self.index.add(TEXT, &[note_id], &vectors)?;
                self.derived.ingest(note_id, FIELD_SOURCE, &refs)?;
                let col_mod = self.collection.run(|core| core.col_mod()).await??;
                self.derived.set_col_mod(col_mod)?;
            }
            Ok(outcome)
        }
        .instrument(span)
        .await
    }

    pub async fn delete_notes(&self, note_ids: Vec<i64>) -> NativeResult<usize> {
        let ids = note_ids.clone();
        let removed = self
            .collection
            .run(move |core| core.delete_notes(&ids))
            .await??;
        self.index.remove(&note_ids)?;
        self.derived.remove(&note_ids, None)?;
        Ok(removed)
    }

    /// Fused search: the semantic ranking (embed → per-modality engine search)
    /// and the lexical rankings (the derived store's substring + fuzzy) each
    /// rank their own candidates; RRF blends them with the exact tier on top —
    /// the same semantics as the Python host's search_notes spine.
    pub async fn search(&self, query: &str, top_k: usize) -> NativeResult<Vec<KernelHit>> {
        let span = tracing::debug_span!("kernel.search", top_k);
        async move {
            // Semantic signal — the embed is the await; the engine/derived reads
            // are fast in-memory/SQLite calls chained after it.
            let qvec = self.embedder.embed(vec![query.to_string()]).await?;
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
            let fused =
                shrike_compute::rrf_fuse(&rankings, &weights, shrike_compute::RRF_K, &priority);
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
        .instrument(span)
        .await
    }

    pub async fn close(self) -> NativeResult<()> {
        self.collection.close().await
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

    impl HashEmbedder {
        fn embed_sync(texts: &[String]) -> Vec<Vec<f32>> {
            texts
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
                .collect()
        }
    }

    impl Embedder for HashEmbedder {
        fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            Box::pin(async move { Ok(Self::embed_sync(&texts)) })
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

    /// Compile-time pin: every kernel future is Send, so a harness may spawn
    /// kernel ops on any multithreaded runtime (the #310 contract — a !Send
    /// regression, e.g. an entered span guard held across an await, fails
    /// here instead of downstream).
    fn assert_send<F: std::future::Future + Send>(f: F) -> F {
        f
    }

    #[test]
    fn open_upsert_search_close_without_python() {
        // The harness picks the runtime: here futures' minimal block_on —
        // no tokio, nothing owned by the kernel.
        futures::executor::block_on(assert_send(smoke()));
    }

    async fn smoke() {
        let dir = temp_dir();
        let col = dir.join("collection.anki2");
        let cache = dir.join("cache");
        // The harness assembles the scheduling: here the thread-free
        // MutexExecutor — serialized, no owned threads, runs on the caller.
        let kernel = Kernel::open(
            col.to_str().unwrap(),
            cache.to_str().unwrap(),
            HashEmbedder,
            Arc::new(MutexExecutor::default()),
        )
        .await
        .unwrap();

        let basic = kernel.notetype_id("Basic").await.unwrap();
        async fn make(
            kernel: &Kernel<HashEmbedder>,
            basic: i64,
            front: &str,
            back: &str,
        ) -> CreateOutcome {
            kernel
                .upsert_note(
                    basic,
                    1,
                    vec![front.to_string(), back.to_string()],
                    vec!["smoke".to_string()],
                    DuplicatePolicy::Error,
                )
                .await
                .unwrap()
        }
        let CreateOutcome::Created(mito) = make(
            &kernel,
            basic,
            "the mitochondria powerhouse",
            "energy of the cell",
        )
        .await
        else {
            panic!("create failed")
        };
        make(
            &kernel,
            basic,
            "newton laws of motion",
            "classical mechanics",
        )
        .await;
        make(
            &kernel,
            basic,
            "paris is the capital of france",
            "geography",
        )
        .await;

        // Duplicate policy is live end to end.
        let dup = kernel
            .upsert_note(
                basic,
                1,
                vec!["the mitochondria powerhouse".into(), "x".into()],
                vec![],
                DuplicatePolicy::Skip,
            )
            .await;
        assert_eq!(dup.unwrap(), CreateOutcome::SkippedDuplicate);

        // Search: semantic + lexical signals both contribute to the winner.
        let hits = kernel.search("mitochondria powerhouse", 5).await.unwrap();
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
        let fuzzy_hits = kernel.search("mitochondira powerhose", 5).await.unwrap();
        assert!(fuzzy_hits.iter().any(|h| h.note_id == mito));

        // Delete propagates to every store.
        assert_eq!(kernel.delete_notes(vec![mito]).await.unwrap(), 1);
        let after = kernel.search("mitochondria powerhouse", 5).await.unwrap();
        assert!(after.iter().all(|h| h.note_id != mito));

        kernel.close().await.unwrap();
        std::fs::remove_dir_all(dir).ok();
    }
}
