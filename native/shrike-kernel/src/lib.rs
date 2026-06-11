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

pub mod actions;
pub mod index_orchestrator;

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

    /// The model fingerprint for index drift detection (`model_id` in the
    /// sidecar) — `None` means "unknown" and skips the model-change rule.
    fn fingerprint(&self) -> Option<String> {
        None
    }

    /// The embedding dimension, when known up front (lets an empty collection
    /// materialize a ready index without a probe embed).
    fn dim(&self) -> Option<usize> {
        None
    }
}

#[cfg(feature = "onnx-embed")]
impl Embedder for shrike_embed::TextEmbedder {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        Box::pin(async move { self.embed_chunk(&texts) })
    }

    fn dim(&self) -> Option<usize> {
        // The inherent method (explicit path: this trait method shadows it).
        shrike_embed::TextEmbedder::dim(self)
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

/// The debounce/idle-timer contract the harness injects (#332 S3c-1) — the
/// sibling of [`SerialExecutor`], for the kernel's two timer consumers (the
/// index saver's debounced flush; the cooperative idle release). One-shot:
/// `schedule` arms a job after `delay_secs`; the returned handle cancels it
/// (a no-op once fired). No threads owned here either — the asyncio harness
/// backs this with `loop.call_later`, an embedded host with its own timers.
pub trait TimerHost: Send + Sync {
    fn schedule(
        &self,
        delay_secs: f64,
        job: Box<dyn FnOnce() + Send + 'static>,
    ) -> Box<dyn TimerCancel>;
}

/// Cancels a scheduled (not-yet-fired) timer job. Dropping without calling
/// `cancel` leaves the timer armed (cancellation is explicit, like the
/// asyncio handle it mirrors).
pub trait TimerCancel: Send {
    fn cancel(&self);
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
///
/// Public since #332 (S3): this is the embedded-host surface the asyncio
/// bridge binds — open/run/close as runtime-agnostic futures over whatever
/// executor the harness injected.
pub struct SerializedCollection {
    core: Arc<CollectionCore>,
    executor: Arc<dyn SerialExecutor>,
}

impl SerializedCollection {
    pub async fn open(
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
    pub async fn run<T: Send + 'static>(
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

    pub async fn close(&self) -> NativeResult<()> {
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

/// One note in a bulk upsert batch — the kernel's batch unit (every kernel op
/// is batch-shaped; the single-note call is sugar over a batch of one).
#[derive(Debug, Clone)]
pub struct NoteSpec {
    pub notetype_id: i64,
    pub deck_id: i64,
    pub fields: Vec<String>,
    pub tags: Vec<String>,
}

/// The kernel: one open collection + the index orchestrator (which owns and
/// maintains the engine) + the derived store + fusion, every op an async fn
/// over the injected executor. No threads owned, no runtime assumed, no
/// transport, no Python. Index maintenance is **kernel-internal** (#332 S3d):
/// upserts/deletes keep the orchestrator's vectors, fingerprints, and
/// watermarks current, and the debounced saver (over the injected
/// [`TimerHost`]) bounds what a crash can discard.
pub struct Kernel<E: Embedder> {
    collection: SerializedCollection,
    orchestrator: Arc<index_orchestrator::IndexOrchestrator>,
    saver: Option<Arc<index_orchestrator::DebouncedSaver>>,
    derived: DerivedEngine,
    embedder: E,
}

const TEXT: &str = "text";
const FIELD_SOURCE: &str = "field";

impl<E: Embedder> Kernel<E> {
    /// Open a collection and its sidecar stores (cache_dir holds the derived
    /// store and the index files, like the Python host's cache layout).
    /// `executor` is the harness-injected scheduling (see [`SerialExecutor`]);
    /// `timers`, when given, arms the debounced index flush (without it, the
    /// index persists only on explicit `save`/rebuild — fine for tests and
    /// one-shot hosts).
    pub async fn open(
        collection_path: &str,
        cache_dir: &str,
        embedder: E,
        executor: Arc<dyn SerialExecutor>,
        timers: Option<Arc<dyn TimerHost>>,
    ) -> NativeResult<Self> {
        std::fs::create_dir_all(cache_dir)
            .map_err(|e| NativeError::internal(format!("cache dir: {e}")))?;
        let collection = SerializedCollection::open(collection_path.to_string(), executor).await?;
        let engine = Arc::new(MultiModalIndex::new(vec![
            TEXT.to_string(),
            "image".to_string(),
        ])?);
        let orchestrator = Arc::new(index_orchestrator::IndexOrchestrator::open(
            cache_dir, engine,
        ));
        let saver = timers.map(|t| {
            index_orchestrator::DebouncedSaver::new(
                Arc::clone(&orchestrator),
                t,
                index_orchestrator::DEFAULT_SAVE_DELAY,
                index_orchestrator::DEFAULT_SAVE_THRESHOLD,
            )
        });
        let derived =
            DerivedEngine::open(&format!("{}/shrike.db", cache_dir.trim_end_matches('/')), 1)?;
        Ok(Self {
            collection,
            orchestrator,
            saver,
            derived,
            embedder,
        })
    }

    /// The orchestrator (state, status, drift) — the harness's status surface.
    pub fn index(&self) -> &index_orchestrator::IndexOrchestrator {
        &self.orchestrator
    }

    /// Bring the index in line with the collection if anything drifted while
    /// the kernel was down (the boot/reload path): reconcile incrementally
    /// when per-note fingerprints exist, rebuild otherwise; an empty
    /// collection materializes an empty-but-ready index. Returns whether any
    /// reindexing ran. The harness drives this as a background task — the
    /// kernel serves while it runs.
    pub async fn reindex_if_needed(&self) -> NativeResult<bool> {
        let col_mod = self.col_mod().await?;
        let model_id = self.embedder.fingerprint();
        if !self
            .orchestrator
            .check_drift(col_mod, model_id.as_deref(), false)
        {
            return Ok(false);
        }
        let raw = self
            .collection
            .run(|core| -> NativeResult<_> {
                let ids = core.find_notes("")?;
                core.note_embed_inputs(&ids)
            })
            .await??;
        if raw.is_empty() {
            if let Some(dim) = self.embedder.dim() {
                self.orchestrator
                    .materialize_empty(dim, col_mod, model_id.as_deref());
            }
            return Ok(true);
        }
        let inputs: Vec<index_orchestrator::EmbedInput> = raw
            .into_iter()
            .map(
                |(note_id, text, image_names)| index_orchestrator::EmbedInput {
                    note_id,
                    text,
                    image_names,
                },
            )
            .collect();
        self.orchestrator
            .reconcile(inputs, col_mod, model_id, &self.embedder, None)
            .await?;
        Ok(true)
    }

    /// Advance both derived-store and index watermarks to the collection's
    /// current `col.mod` and request a debounced index flush.
    async fn advance_watermarks(&self) -> NativeResult<()> {
        let col_mod = self.collection.run(|core| core.col_mod()).await??;
        self.derived.set_col_mod(col_mod)?;
        self.orchestrator.set_col_mod(col_mod);
        if let Some(saver) = &self.saver {
            saver.request_save();
        }
        Ok(())
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

    /// Create one note — sugar over [`upsert_notes`] with a batch of one (the
    /// batch op is the real implementation; per-item errors surface directly
    /// here since the batch is a single item).
    pub async fn upsert_note(
        &self,
        notetype_id: i64,
        deck_id: i64,
        fields: Vec<String>,
        tags: Vec<String>,
        policy: DuplicatePolicy,
    ) -> NativeResult<CreateOutcome> {
        let spec = NoteSpec {
            notetype_id,
            deck_id,
            fields,
            tags,
        };
        let mut results = self.upsert_notes(vec![spec], policy).await?;
        results.remove(0)
    }

    /// Create a batch of notes (the #77 duplicate policy applies per item) and
    /// index them — batch-shaped end to end: ONE collection job runs every
    /// create (per-item results, so one bad note never sinks the batch), ONE
    /// read job renders embed text + derived rows for everything created, ONE
    /// batched embed call produces all vectors, then one index add and a
    /// per-note derived ingest. Compute (embedding, index, derived) happens
    /// *off* the collection queue — it never routes back through it.
    pub async fn upsert_notes(
        &self,
        notes: Vec<NoteSpec>,
        policy: DuplicatePolicy,
    ) -> NativeResult<Vec<NativeResult<CreateOutcome>>> {
        // `.instrument`, never an entered guard across an await: the span
        // follows the future across polls (and threads), and the future stays
        // Send — spawnable on any multithreaded runtime.
        let span = tracing::debug_span!("kernel.upsert_notes", batch = notes.len());
        async move {
            // One serialized job for the whole batch of writes.
            let outcomes: Vec<NativeResult<CreateOutcome>> = self
                .collection
                .run(move |core| {
                    notes
                        .iter()
                        .map(|n| {
                            core.create_note(n.notetype_id, n.deck_id, &n.fields, &n.tags, policy)
                        })
                        .collect()
                })
                .await?;
            let created: Vec<i64> = outcomes
                .iter()
                .filter_map(|o| match o {
                    Ok(CreateOutcome::Created(id)) => Some(*id),
                    _ => None,
                })
                .collect();
            if !created.is_empty() {
                // The SAME normalized render + raw field rows the Python host
                // uses (#278 step 2) — embedding text and derived rows come
                // from the read surface, not an ad-hoc join. One job for the
                // whole created set.
                let ids = created.clone();
                let (raw_inputs, rows) = self
                    .collection
                    .run(move |core| -> NativeResult<_> {
                        let inputs = core.note_embed_inputs(&ids)?;
                        let rows = core.derived_field_rows(&ids)?;
                        Ok((inputs, rows))
                    })
                    .await??;
                // One orchestrator add for the whole batch (it chunks the
                // embeds internally and maintains the per-note fingerprints).
                let inputs: Vec<index_orchestrator::EmbedInput> = raw_inputs
                    .into_iter()
                    .map(
                        |(note_id, text, image_names)| index_orchestrator::EmbedInput {
                            note_id,
                            text,
                            image_names,
                        },
                    )
                    .collect();
                self.orchestrator.add(&inputs, &self.embedder, None).await?;
                // Group the derived rows per note (rows come back grouped by
                // note already; ingest replaces per (note, source)).
                let mut refs: BTreeMap<i64, Vec<(String, String)>> = BTreeMap::new();
                for (nid, _source, name, value) in rows {
                    refs.entry(nid).or_default().push((name, value));
                }
                for note_id in &created {
                    let note_refs = refs.remove(note_id).unwrap_or_default();
                    self.derived.ingest(*note_id, FIELD_SOURCE, &note_refs)?;
                }
                self.advance_watermarks().await?;
            }
            Ok(outcomes)
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
        self.orchestrator.remove(&note_ids)?;
        self.derived.remove(&note_ids, None)?;
        self.advance_watermarks().await?;
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
            let semantic = self
                .orchestrator
                .engine()
                .search_by_modality(&qvec, top_k, None)?;
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

    /// Cooperative idle-release (#64): close the collection, keeping the
    /// kernel reusable via [`reopen`]. WHEN to release is harness policy (an
    /// idle timer on its runtime); the kernel only provides the ops.
    pub async fn release(&self) -> NativeResult<()> {
        self.collection.run(|core| core.release()).await?
    }

    /// Re-acquire after a release; contention surfaces as the BUSY error
    /// tier (retryable — the caller decides, nothing waits).
    pub async fn reopen(&self) -> NativeResult<()> {
        self.collection.run(|core| core.reopen()).await?
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

        fn fingerprint(&self) -> Option<String> {
            Some("hash-embedder:v1".to_string())
        }

        fn dim(&self) -> Option<usize> {
            Some(64)
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
            None,
        )
        .await
        .unwrap();

        // Boot path: a fresh empty collection materializes a ready index, so
        // the upserts below maintain per-note fingerprints incrementally.
        assert!(kernel.reindex_if_needed().await.unwrap());
        assert!(!kernel.reindex_if_needed().await.unwrap()); // now current

        let basic = kernel.notetype_id("Basic").await.unwrap();
        let spec = |front: &str, back: &str| NoteSpec {
            notetype_id: basic,
            deck_id: 1,
            fields: vec![front.to_string(), back.to_string()],
            tags: vec!["smoke".to_string()],
        };
        // The single-note sugar (delegates to the batch op).
        let CreateOutcome::Created(mito) = kernel
            .upsert_note(
                basic,
                1,
                vec![
                    "the mitochondria powerhouse".into(),
                    "energy of the cell".into(),
                ],
                vec!["smoke".to_string()],
                DuplicatePolicy::Error,
            )
            .await
            .unwrap()
        else {
            panic!("create failed")
        };

        // The batch op proper: one call → one collection job for the creates,
        // one read job, ONE batched embed. The middle item is a duplicate and
        // a structural error rides per-item — neither sinks the batch.
        let batch = kernel
            .upsert_notes(
                vec![
                    spec("newton laws of motion", "classical mechanics"),
                    spec("the mitochondria powerhouse", "dupe"),
                    NoteSpec {
                        fields: vec!["".into(), "empty first field".into()],
                        ..spec("", "")
                    },
                    spec("paris is the capital of france", "geography"),
                ],
                DuplicatePolicy::Skip,
            )
            .await
            .unwrap();
        assert_eq!(batch.len(), 4);
        assert!(matches!(batch[0], Ok(CreateOutcome::Created(_))));
        assert_eq!(batch[1].as_ref().unwrap(), &CreateOutcome::SkippedDuplicate);
        assert!(batch[2].is_err(), "structural error must ride per-item");
        assert!(matches!(batch[3], Ok(CreateOutcome::Created(_))));

        // Duplicate policy is live end to end through the single-note sugar.
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

        // Restart WITHOUT a flush: the on-disk sidecars still hold the
        // boot-time materialized-empty state, so the fresh kernel detects
        // drift and reconciles — re-embedding every live note via the
        // collection read surface (find_notes + note_embed_inputs).
        kernel.close().await.unwrap();
        let kernel2 = Kernel::open(
            col.to_str().unwrap(),
            cache.to_str().unwrap(),
            HashEmbedder,
            Arc::new(MutexExecutor::default()),
            None,
        )
        .await
        .unwrap();
        assert!(kernel2.reindex_if_needed().await.unwrap()); // drift → reconcile
        assert!(!kernel2.reindex_if_needed().await.unwrap()); // now current
        let hits = kernel2.search("newton laws of motion", 5).await.unwrap();
        assert!(!hits.is_empty());
        kernel2.close().await.unwrap();
        std::fs::remove_dir_all(dir).ok();
    }
}
