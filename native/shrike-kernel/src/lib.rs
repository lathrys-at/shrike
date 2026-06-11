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
pub mod tag_centroids;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};

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

/// Embedders share freely: an `Arc`'d embedder is an embedder (the binding
/// hands one embedder to the kernel and keeps a handle for query embeds).
impl<T: Embedder> Embedder for Arc<T> {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        (**self).embed(texts)
    }

    fn fingerprint(&self) -> Option<String> {
        (**self).fingerprint()
    }

    fn dim(&self) -> Option<usize> {
        (**self).dim()
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
    /// transition point ops chain continuations onto. Re-acquires first if
    /// idle-released (#64's open-on-demand, kernel-side — so a cooperative
    /// release between any two jobs self-heals; contention surfaces as the
    /// BUSY tier from `ensure_open`).
    pub async fn run<T: Send + 'static>(
        &self,
        job: impl FnOnce(&CollectionCore) -> T + Send + 'static,
    ) -> NativeResult<T> {
        let core = Arc::clone(&self.core);
        let (tx, rx) = oneshot::channel();
        self.executor
            .submit(Box::new(move || {
                let _ = tx.send(core.ensure_open().map(|_| job(&core)));
            }))
            .await;
        rx.await
            .map_err(|_| NativeError::internal("executor dropped a collection job"))?
    }

    pub async fn close(&self) -> NativeResult<()> {
        self.run(|core| core.close()).await?
    }

    /// The shared core, for a harness that runs its own (executor-disciplined)
    /// direct ops over the same collection the kernel owns.
    pub fn core_arc(&self) -> Arc<CollectionCore> {
        Arc::clone(&self.core)
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
pub struct Kernel {
    collection: SerializedCollection,
    orchestrator: Arc<index_orchestrator::IndexOrchestrator>,
    saver: Option<Arc<index_orchestrator::DebouncedSaver>>,
    derived: DerivedEngine,
    /// The attachable embedding service (#342's first registry slot):
    /// swappable at runtime — the harness attaches on embedding start,
    /// detaches on stop, and a model swap is detach + attach. Ops that need
    /// embedding degrade (lexical-only search, unindexed-but-created upserts)
    /// when the slot is empty, mirroring the Python host's gating.
    embed: RwLock<Option<Arc<EmbedService>>>,
    /// Tag-centroid state (#178/#179): the live key→tag map for the engine's
    /// `tag.text` space + the hygiene knobs. Centroids recompute at the tail
    /// of every index-changing op (a pure function of in-engine text vectors
    /// + membership, so no extra watermark).
    tag_keys: tag_centroids::TagKeyMap,
    tag_config: tag_centroids::TagCentroidConfig,
}

/// One attached embedding capability: the text embedder + optionally its
/// image half (present only for an image-advertising backend with a media
/// resolver).
pub struct EmbedService {
    pub embedder: Arc<dyn Embedder>,
    pub images: Option<KernelImages>,
}

impl EmbedService {
    /// The image pair as the borrow shape the orchestrator ops take.
    fn images_pair(
        &self,
    ) -> Option<(
        &dyn index_orchestrator::ImageEmbedder,
        &dyn index_orchestrator::ImageResolver,
    )> {
        self.images.as_ref().map(|(e, r)| (&**e, &**r))
    }
}

/// The injected image pair: who embeds image bytes + who resolves filenames
/// to bytes (lazily, at embed time).
pub type KernelImages = (
    Box<dyn index_orchestrator::ImageEmbedder>,
    Box<dyn index_orchestrator::ImageResolver>,
);

const FIELD_SOURCE: &str = "field";

/// The NOTE-item vector spaces (#178): every note search is scoped to these,
/// so other entity kinds sharing the engine (per-(tag, modality) centroids in
/// `tag.*` spaces) can never surface a non-note key from a note query — the
/// no-leakage property is structural, not a post-filter.
pub const NOTE_MODALITIES: &[&str] = &["text", "image"];

/// The per-modality tag-centroid spaces (#178/#179): `tag.text` holds the
/// renormalized mean of member notes' TEXT vectors per tag (never a
/// cross-modal mean — the modality gap makes one semantically empty).
pub const TAG_TEXT_SPACE: &str = "tag.text";

impl Kernel {
    /// Open a collection and its sidecar stores (cache_dir holds the derived
    /// store and the index files, like the Python host's cache layout).
    /// `executor` is the harness-injected scheduling (see [`SerialExecutor`]);
    /// `timers`, when given, arms the debounced index flush (without it, the
    /// index persists only on explicit `save`/rebuild — fine for tests and
    /// one-shot hosts).
    pub async fn open(
        collection_path: &str,
        cache_dir: &str,
        executor: Arc<dyn SerialExecutor>,
        timers: Option<Arc<dyn TimerHost>>,
    ) -> NativeResult<Self> {
        std::fs::create_dir_all(cache_dir)
            .map_err(|e| NativeError::internal(format!("cache dir: {e}")))?;
        let collection = SerializedCollection::open(collection_path.to_string(), executor).await?;
        let engine = Arc::new(MultiModalIndex::new(
            NOTE_MODALITIES
                .iter()
                .map(|m| m.to_string())
                .chain(std::iter::once(TAG_TEXT_SPACE.to_string()))
                .collect(),
        )?);
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
        tracing::debug!(collection = collection_path, "kernel opened");
        Ok(Self {
            collection,
            orchestrator,
            saver,
            derived,
            embed: RwLock::new(None),
            tag_keys: tag_centroids::TagKeyMap::default(),
            tag_config: tag_centroids::TagCentroidConfig::default(),
        })
    }

    /// The live tag-key map (key → tag string) for the `tag.text` space.
    pub fn tag_keys(&self) -> &tag_centroids::TagKeyMap {
        &self.tag_keys
    }

    /// Recompute every tag centroid from the engine's current text vectors +
    /// one membership pass (#179). Cheap (hundreds of tags, in-memory vector
    /// reads); runs at the tail of every index-changing op and is a no-op
    /// shortcut when no embedder is attached (no text vectors to mean).
    pub async fn refresh_tag_centroids(&self) -> NativeResult<usize> {
        if self.embed_service().is_none() {
            return Ok(0);
        }
        let (rows, total) = self
            .collection
            .run(|core| -> NativeResult<_> {
                let rows = core.note_tag_rows()?;
                let total = core.find_notes("")?.len();
                Ok((rows, total))
            })
            .await??;
        let built = tag_centroids::recompute(
            self.orchestrator.engine(),
            &rows,
            total,
            &self.tag_config,
            &self.tag_keys,
        )?;
        if let Some(saver) = &self.saver {
            saver.request_save();
        }
        Ok(built)
    }

    /// Best-effort refresh: the tag layer is conditionally-present and must
    /// never fail the index op it rides on.
    async fn refresh_tags_best_effort(&self) {
        if let Err(e) = self.refresh_tag_centroids().await {
            tracing::warn!(error = ?e, "tag centroid refresh failed");
        }
    }

    /// Attach an embedding service (embedding start / model swap). The
    /// orchestrator flips back to ready if it was only unavailable; the
    /// harness follows up with `reindex_if_needed` (a model change is drift).
    pub fn attach_embedder(&self, embedder: Arc<dyn Embedder>, images: Option<KernelImages>) {
        *self.embed.write().expect("embed slot poisoned") =
            Some(Arc::new(EmbedService { embedder, images }));
        self.orchestrator.mark_ready_if_loaded();
    }

    /// Detach the embedding service (embedding stop): flush the index (the
    /// on-disk vectors are kept) and mark it unavailable. The collection and
    /// the lexical search surfaces stay fully live.
    pub fn detach_embedder(&self) {
        *self.embed.write().expect("embed slot poisoned") = None;
        let _ = self.orchestrator.save();
        self.orchestrator.mark_unavailable();
    }

    /// The currently attached embedding service, if any.
    pub fn embed_service(&self) -> Option<Arc<EmbedService>> {
        self.embed.read().expect("embed slot poisoned").clone()
    }

    /// The orchestrator (state, status, drift) — the harness's status surface.
    pub fn index(&self) -> &index_orchestrator::IndexOrchestrator {
        &self.orchestrator
    }

    /// The serialized collection — the harness's seam for sharing the core
    /// (its direct ops must honor the same executor discipline).
    pub fn collection(&self) -> &SerializedCollection {
        &self.collection
    }

    /// Bring the index in line with the collection if anything drifted while
    /// the kernel was down (the boot/reload path): reconcile incrementally
    /// when per-note fingerprints exist, rebuild otherwise; an empty
    /// collection materializes an empty-but-ready index. Returns whether any
    /// reindexing ran. The harness drives this as a background task — the
    /// kernel serves while it runs.
    pub async fn reindex_if_needed(&self) -> NativeResult<bool> {
        let Some(svc) = self.embed_service() else {
            return Ok(false); // no embedder → nothing to (re)index
        };
        let col_mod = self.col_mod().await?;
        let model_id = svc.embedder.fingerprint();
        if !self
            .orchestrator
            .check_drift(col_mod, model_id.as_deref(), svc.images.is_some())
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
            if let Some(dim) = svc.embedder.dim() {
                self.orchestrator
                    .materialize_empty(dim, col_mod, model_id.as_deref());
            }
            self.refresh_tags_best_effort().await;
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
            .reconcile(inputs, col_mod, model_id, &*svc.embedder, svc.images_pair())
            .await?;
        self.refresh_tags_best_effort().await;
        Ok(true)
    }

    /// Explicit FULL index rebuild (the `/index/rebuild` semantics): drop and
    /// re-embed everything — never the incremental path (reconcile is only
    /// the automatic drift route). Returns the note count. Errors Unavailable
    /// with no embedder attached.
    pub async fn rebuild_index(&self) -> NativeResult<usize> {
        let Some(svc) = self.embed_service() else {
            return Err(NativeError::unavailable(
                "no embedding service attached — start embedding first",
            ));
        };
        let col_mod = self.col_mod().await?;
        let model_id = svc.embedder.fingerprint();
        let raw = self
            .collection
            .run(|core| -> NativeResult<_> {
                let ids = core.find_notes("")?;
                core.note_embed_inputs(&ids)
            })
            .await??;
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
        let total = inputs.len();
        self.orchestrator
            .rebuild(inputs, col_mod, model_id, &*svc.embedder, svc.images_pair())
            .await?;
        self.refresh_tags_best_effort().await;
        Ok(total)
    }

    /// Post-write maintenance for a set of created/updated notes: ONE read
    /// job for embed inputs + derived rows, one orchestrator add (replace
    /// semantics — when an embedder is attached; notes are created and
    /// lexically indexed regardless), per-note derived ingest, and the
    /// watermark advance. The shared tail of every upsert shape — public as
    /// `reindex_notes` for harness ops that edit note text outside the
    /// upsert ops (find/replace, note-type migration).
    pub async fn reindex_notes(&self, written: &[i64]) -> NativeResult<()> {
        self.index_written(written).await
    }

    async fn index_written(&self, written: &[i64]) -> NativeResult<()> {
        if written.is_empty() {
            return Ok(());
        }
        let ids = written.to_vec();
        let (raw_inputs, rows) = self
            .collection
            .run(move |core| -> NativeResult<_> {
                let inputs = core.note_embed_inputs(&ids)?;
                let rows = core.derived_field_rows(&ids)?;
                Ok((inputs, rows))
            })
            .await??;
        let svc = self.embed_service();
        if let Some(svc) = &svc {
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
            self.orchestrator
                .add(&inputs, &*svc.embedder, svc.images_pair())
                .await?;
        }
        // Group the derived rows per note (ingest replaces per (note, source)).
        let mut refs: BTreeMap<i64, Vec<(String, String)>> = BTreeMap::new();
        for (nid, _source, name, value) in rows {
            refs.entry(nid).or_default().push((name, value));
        }
        for note_id in written {
            let note_refs = refs.remove(note_id).unwrap_or_default();
            self.derived.ingest(*note_id, FIELD_SOURCE, &note_refs)?;
        }
        self.advance_watermarks(svc.is_some()).await?;
        // Tag centroids derive from the text vectors just written (#179).
        self.refresh_tags_best_effort().await;
        Ok(())
    }

    /// The wire-shaped bulk upsert (#77): the collection core's NAMED upsert
    /// — `id`?/`note_type`/`deck`/`fields` map/`tags`, create AND update,
    /// `dry_run`, per-item results JSON — run as ONE collection job, then the
    /// kernel-internal index/derived maintenance over everything written.
    /// This is the op the MCP `upsert_notes` action rides (S3d-2).
    pub async fn upsert_notes_json(
        &self,
        notes_json: String,
        on_duplicate: String,
        dry_run: bool,
    ) -> NativeResult<String> {
        let span = tracing::debug_span!("kernel.upsert_notes_json", dry_run);
        async move {
            let results_json = self
                .collection
                .run(move |core| core.upsert_notes(&notes_json, &on_duplicate, dry_run))
                .await??;
            if !dry_run {
                let results: Vec<serde_json::Value> = serde_json::from_str(&results_json)
                    .map_err(|e| NativeError::internal(format!("upsert results: {e}")))?;
                let written: Vec<i64> = results
                    .iter()
                    .filter(|r| {
                        matches!(
                            r.get("status").and_then(serde_json::Value::as_str),
                            Some("created") | Some("updated")
                        )
                    })
                    .filter_map(|r| r.get("id").and_then(serde_json::Value::as_i64))
                    .collect();
                self.index_written(&written).await?;
            }
            Ok(results_json)
        }
        .instrument(span)
        .await
    }

    /// Advance the derived-store watermark — and, when the index was actually
    /// maintained by this op (an embedder is attached), the index watermark +
    /// a debounced flush. With no embedder the index watermark stays put, so
    /// the next attach sees drift and reconciles (cheap via the fingerprints).
    async fn advance_watermarks(&self, index_maintained: bool) -> NativeResult<()> {
        let col_mod = self.collection.run(|core| core.col_mod()).await??;
        self.derived.set_col_mod(col_mod)?;
        if index_maintained {
            self.orchestrator.set_col_mod(col_mod);
            if let Some(saver) = &self.saver {
                saver.request_save();
            }
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
            self.index_written(&created).await?;
            Ok(outcomes)
        }
        .instrument(span)
        .await
    }

    /// Drop already-deleted notes from the index + derived store (the prune
    /// path: the collection op removed them internally; this is the sidecar
    /// half of `delete_notes`).
    pub async fn forget_notes(&self, note_ids: Vec<i64>) -> NativeResult<()> {
        self.orchestrator.remove(&note_ids)?;
        self.derived.remove(&note_ids, None)?;
        self.advance_watermarks(self.embed_service().is_some())
            .await?;
        self.refresh_tags_best_effort().await;
        Ok(())
    }

    /// A metadata-only collection change (tags/decks/templates/field metadata
    /// — nothing that feeds embedding text or derived rows): advance the
    /// watermarks so the col_mod bump doesn't read as drift on next boot.
    pub async fn metadata_changed(&self) -> NativeResult<()> {
        self.advance_watermarks(self.embed_service().is_some())
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
        self.advance_watermarks(self.embed_service().is_some())
            .await?;
        self.refresh_tags_best_effort().await;
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
            // are fast in-memory/SQLite calls chained after it. With no
            // embedder attached, search degrades to the lexical signals.
            let mut rankings: Vec<(String, Vec<i64>)> = Vec::new();
            if let Some(svc) = self.embed_service() {
                let qvec = svc.embedder.embed(vec![query.to_string()]).await?;
                let note_spaces: Vec<String> =
                    NOTE_MODALITIES.iter().map(|m| m.to_string()).collect();
                let semantic = self.orchestrator.engine().search_by_modality(
                    &qvec,
                    top_k,
                    Some(&note_spaces),
                )?;
                if let Some(per_query) = semantic.first() {
                    for (modality, (ids, _dists)) in per_query {
                        rankings.push((modality.clone(), ids.clone()));
                    }
                }
                // Tag-centroid signal (#179): conditionally present.
                if let Some(qvec0) = qvec.first() {
                    let tag_notes = tag_centroids::tag_ranking(
                        self.orchestrator.engine(),
                        &self.tag_keys,
                        qvec0,
                        tag_centroids::TAG_ACTIVATION,
                        tag_centroids::TAG_TOP_TAGS,
                        tag_centroids::TAG_RANK_CAP,
                    );
                    if !tag_notes.is_empty() {
                        rankings.push(("tag".to_string(), tag_notes));
                    }
                }
            }

            // Lexical signals (substring authority + fuzzy), from the derived store.
            let quoted = format!("\"{}\"", query.replace('"', "\"\""));
            let exact: Vec<i64> = self
                .derived
                .match_rows(&quoted, top_k as i64, false, None)?
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
                    .match_rows(&expr, (top_k * 4) as i64, false, None)?
                    .into_iter()
                    .map(|(nid, ..)| nid)
                    .collect();
                rankings.push(("fuzzy".to_string(), fuzzy));
            }

            // Fuse (the frozen #274 semantics: weights, exact tier, determinism).
            let mut weights = BTreeMap::new();
            weights.insert("text".to_string(), 1.0);
            weights.insert("image".to_string(), 1.0);
            weights.insert("tag".to_string(), 1.0);
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
    fn detach_degrades_and_reattach_recovers() {
        // The embed slot is runtime-swappable (#342): detached, the kernel
        // still creates notes and serves lexical search; re-attached, the
        // stale index watermark makes reindex catch up on what it missed.
        futures::executor::block_on(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
                Arc::new(MutexExecutor::default()),
                None,
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();
            let basic = kernel.notetype_id("Basic").await.unwrap();

            kernel.detach_embedder();
            let outcome = kernel
                .upsert_note(
                    basic,
                    1,
                    vec!["paris is the capital of france".into(), "geo".into()],
                    vec![],
                    DuplicatePolicy::Error,
                )
                .await
                .unwrap();
            let CreateOutcome::Created(nid) = outcome else {
                panic!("create must work without an embedder")
            };
            // Lexical-only: the literal hit lands, no semantic signal exists.
            let hits = kernel.search("capital of france", 5).await.unwrap();
            assert_eq!(hits[0].note_id, nid);
            assert!(hits[0].signals.iter().all(|(s, _)| s != "text"));

            // Re-attach: the index watermark stayed put, so reindex sees
            // drift and embeds the note created while detached.
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            assert!(kernel.reindex_if_needed().await.unwrap());
            let hits = kernel.search("capital of france", 5).await.unwrap();
            assert!(hits[0].signals.iter().any(|(s, _)| s == "text"));

            kernel.close().await.unwrap();
            std::fs::remove_dir_all(dir).ok();
        });
    }

    #[test]
    fn kernel_ops_reopen_after_cooperative_release() {
        // The #64 open-on-demand, kernel-side: an idle release between ops
        // (or between one op's jobs) self-heals on the next serialized job
        // instead of erroring CollectionNotOpen.
        futures::executor::block_on(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
                Arc::new(MutexExecutor::default()),
                None,
            )
            .await
            .unwrap();
            let basic = kernel.notetype_id("Basic").await.unwrap();

            kernel.release().await.unwrap();
            let outcome = kernel
                .upsert_note(
                    basic,
                    1,
                    vec!["created while released".into(), "b".into()],
                    vec![],
                    DuplicatePolicy::Error,
                )
                .await
                .unwrap();
            assert!(matches!(outcome, CreateOutcome::Created(_)));

            // And the wire-shaped op too, straight after another release.
            kernel.release().await.unwrap();
            let results = kernel
                .upsert_notes_json(
                    r#"[{"note_type": "Basic", "deck": "Default",
                         "fields": {"Front": "second", "Back": "b"}}]"#
                        .to_string(),
                    "error".to_string(),
                    false,
                )
                .await
                .unwrap();
            assert!(results.contains("\"created\""), "got: {results}");

            kernel.close().await.unwrap();
            std::fs::remove_dir_all(dir).ok();
        });
    }

    #[test]
    fn tag_centroids_build_and_never_leak_into_note_search() {
        // The #178/#179 layer end to end: tagged upserts → centroids in the
        // tag.text space (hygiene-filtered) → note searches structurally
        // blind to tag keys.
        futures::executor::block_on(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
                Arc::new(MutexExecutor::default()),
                None,
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();

            // 5 notes; 2 share `bio::cell` (under the 50% coverage cap, over
            // the 2-member floor — and the hierarchy rolls up to `bio`).
            let notes: Vec<serde_json::Value> = (0..5)
                .map(|i| {
                    serde_json::json!({
                        "note_type": "Basic", "deck": "Default",
                        "fields": {"Front": format!("note number {i}"), "Back": "b"},
                        "tags": if i < 2 { vec!["bio::cell"] } else { vec![] },
                    })
                })
                .collect();
            kernel
                .upsert_notes_json(
                    serde_json::json!(notes).to_string(),
                    "error".to_string(),
                    false,
                )
                .await
                .unwrap();

            let keys = kernel.tag_keys();
            assert!(!keys.is_empty(), "centroids built on the upsert tail");
            let cell_key = tag_centroids::tag_key("bio::cell");
            assert_eq!(keys.lookup(cell_key).as_deref(), Some("bio::cell"));
            assert_eq!(
                keys.lookup(tag_centroids::tag_key("bio")).as_deref(),
                Some("bio"),
                "hierarchy rolls up"
            );
            let engine = kernel.index().engine();
            assert!(engine.modality_get(TAG_TEXT_SPACE, cell_key).is_some());

            // A note search never surfaces a tag key.
            let hits = kernel.search("note number", 20).await.unwrap();
            assert!(hits.iter().all(|h| keys.lookup(h.note_id).is_none()));

            kernel.close().await.unwrap();
            std::fs::remove_dir_all(dir).ok();
        });
    }

    #[test]
    fn tag_signal_surfaces_off_topic_members() {
        // The #179 payoff: a member whose own text doesn't match the query
        // still surfaces because its TAG's centroid (dominated by on-topic
        // siblings) activates and expands.
        futures::executor::block_on(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
                Arc::new(MutexExecutor::default()),
                None,
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();

            // Tag `krebs`: two on-topic notes + one member with unrelated
            // text; plus untagged off-topic filler for hygiene coverage.
            let mut notes: Vec<serde_json::Value> = vec![
                serde_json::json!({"note_type": "Basic", "deck": "Default",
                    "fields": {"Front": "krebs cycle citric acid", "Back": "b"},
                    "tags": ["krebs"]}),
                serde_json::json!({"note_type": "Basic", "deck": "Default",
                    "fields": {"Front": "krebs cycle mitochondria atp", "Back": "b"},
                    "tags": ["krebs"]}),
                serde_json::json!({"note_type": "Basic", "deck": "Default",
                    "fields": {"Front": "zzz unrelated mnemonic", "Back": "b"},
                    "tags": ["krebs"]}),
            ];
            for i in 0..6 {
                notes.push(serde_json::json!({"note_type": "Basic", "deck": "Default",
                    "fields": {"Front": format!("filler topic {i}"), "Back": "b"},
                    "tags": []}));
            }
            let results: Vec<serde_json::Value> = serde_json::from_str(
                &kernel
                    .upsert_notes_json(
                        serde_json::json!(notes).to_string(),
                        "error".to_string(),
                        false,
                    )
                    .await
                    .unwrap(),
            )
            .unwrap();
            let mnemonic_id = results[2]["id"].as_i64().unwrap();

            assert!(!kernel.tag_keys().is_empty(), "centroid state built");
            let key = tag_centroids::tag_key("krebs");
            assert_eq!(kernel.tag_keys().members(key).len(), 3);
            let hits = kernel.search("krebs cycle citric acid", 9).await.unwrap();
            let with_tag: Vec<&KernelHit> = hits
                .iter()
                .filter(|h| h.signals.iter().any(|(s, _)| s == "tag"))
                .collect();
            assert!(!with_tag.is_empty(), "the tag signal contributed");
            assert!(
                hits.iter().any(|h| h.note_id == mnemonic_id),
                "the off-topic member surfaced via its tag"
            );

            kernel.close().await.unwrap();
            std::fs::remove_dir_all(dir).ok();
        });
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
            Arc::new(MutexExecutor::default()),
            None,
        )
        .await
        .unwrap();
        // The harness attaches the embedding service (#342's registry slot).
        kernel.attach_embedder(Arc::new(HashEmbedder), None);

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
            Arc::new(MutexExecutor::default()),
            None,
        )
        .await
        .unwrap();
        kernel2.attach_embedder(Arc::new(HashEmbedder), None);
        assert!(kernel2.reindex_if_needed().await.unwrap()); // drift → reconcile
        assert!(!kernel2.reindex_if_needed().await.unwrap()); // now current
        let hits = kernel2.search("newton laws of motion", 5).await.unwrap();
        assert!(!hits.is_empty());
        kernel2.close().await.unwrap();
        std::fs::remove_dir_all(dir).ok();
    }
}
