//! The pure-Rust kernel.
//!
//! This crate composes the native compute plane into the embedded-host shape:
//! it owns the collection core (anki via its protobuf service
//! layer), the vector index engine, the derived-text store, the
//! fusion — and, since the tokio pivot, **its own runtime**
//! ([`runtime`]). The kernel is idiomatic async Rust: every op is an
//! `async fn` composing with ordinary awaits (embed → index add → derived
//! ingest), collection access serializes through a task-actor
//! ([`SerializedCollection`]), and hosts adapt the *action exchange* — an op
//! in, a completion-backed future out via [`spawn_op`] — never scheduling.
//! (anki keeps its own runtime for sync; the kernel guarantees sync ops never
//! run on a runtime worker thread, not that only one runtime exists — see
//! [`runtime`] for the `spawn_blocking` discipline and its panic-repro
//! gate.)
//!
//! There is **no pyo3 anywhere in this dependency tree** (enforced by
//! `//shrike-core:layering_check`); the no-CPython smoke test in
//! this crate links the kernel without Python and runs open → upsert → search
//! (a semantic and a lexical signal both contributing) → close — the
//! acceptance's executable form.

#![deny(missing_docs)]
#![deny(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc
)]

pub mod actions;
pub mod embed_set;
pub mod fusion;
pub mod index_orchestrator;
pub mod index_set;
pub mod ingest;
pub mod maintenance;
pub mod recognize;
pub mod tag_centroids;
pub mod watermark;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};

use futures::channel::oneshot;
#[cfg(test)]
use futures::future::BoxFuture;
use tracing::Instrument;

use shrike_collection::CollectionCore;
use shrike_collection::{Collection, CreateOutcome, DuplicatePolicy};
use shrike_derived::DerivedEngine;
use shrike_error::{ErrorKind, NativeError, NativeResult, ResultExt};
use shrike_index::MultiModalIndex;
use shrike_store::{DerivedStore, VectorIndex};

pub mod runtime;
pub use runtime::{
    block_on, drive_compute, drive_io, drive_io_until_shutdown, drive_sync, init_driven_runtime,
    is_driven, shutdown_driven_pools, spawn_op, submit_blocking, submit_compute,
};

// The multi-engine routing key: re-exported so the pyo3 binding maps
// the harness's purpose string onto it as `shrike_kernel::RecognitionPurpose`
// without reaching into the `recognize` module.
pub use recognize::RecognitionPurpose;

// The embed-slot set: the ordered set of embedding spaces the kernel
// holds in place of a single service. Re-exported so the bindings name
// the carrier as `shrike_kernel::EmbedSpaces` without reaching into the module.
pub use embed_set::{EmbedSpace, EmbedSpaces};

// The engine contract: traits live in shrike-engine-api — the kernel
// consumes them and re-exports for downstream paths; it names no engine.
pub use shrike_engine_api::{
    Embedder, ImageEmbedder, ImageResolver, Locator, MediaItem, Recognition, Recognizer, Segment,
};

// The export request/scope/format types: the collection contract's,
// re-exported so the pyo3 binding constructs them as `shrike_kernel::*`
// without reaching past the kernel into the collection crate.
pub use shrike_collection::{ExportOutcome, ExportRequest, ExportScope, PackageFormat};

/// The collection as a task-actor: every access is one job sent to a
/// single spawned task that runs them **inline, sequentially** — FIFO
/// serialization by construction, no thread affinity (the task owns the
/// receiver and migrates freely across runtime workers between polls;
/// `CollectionCore` is Send). Jobs are synchronous closures and never await,
/// which makes the old re-entrancy rule structural. On a `current_thread`
/// runtime the actor shares the one thread driving everything — the
/// degenerate single-thread mode works by construction (no `block_in_place`
/// anywhere).
///
/// Inline jobs briefly occupy whichever worker polls the actor — strictly
/// less thread-hungry than the retired permanently-dedicated worker thread,
/// and engine compute lives on the separate blocking pool so embeds never
/// compete. Because jobs run inline on a runtime worker, a sync anki call
/// that `block_on`s would panic if invoked *directly* in a job (any
/// runtime-worker thread is a runtime context). anki's `block_on` lives only
/// on the sync/AnkiWeb service paths, none of which Shrike dispatches today
/// (pinned in shrike-collection); when client sync lands, those
/// sync ops MUST ride `spawn_blocking` rather than an inline job — a
/// blocking-pool thread is a legal `block_on` site. The discipline and its
/// panic-repro gate live in [`runtime`].
pub struct SerializedCollection {
    core: Arc<dyn Collection>,
    /// `None` after [`SerializedCollection::shutdown`] — dropping the sender
    /// is what ends the actor loop.
    jobs: Mutex<Option<tokio::sync::mpsc::UnboundedSender<Job>>>,
    /// Awaited by [`SerializedCollection::shutdown`] so a kernel close
    /// drains the actor before returning (nothing mid-job at teardown).
    actor: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

impl SerializedCollection {
    /// Open a collection at `collection_path` inside the actor's first job, so
    /// no collection access ever happens off the actor task.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection cannot be opened (e.g. it is locked
    /// by another process) or the actor drops the open job.
    pub async fn open(collection_path: String) -> NativeResult<Self> {
        // The open IS the actor's first job: the core is created inside the
        // task, so no collection access ever happens outside it.
        let (opened_tx, opened_rx) = oneshot::channel();
        let (jobs_tx, mut jobs_rx) = tokio::sync::mpsc::unbounded_channel::<Job>();
        let actor = runtime::handle().spawn(async move {
            // Open on the sync pool (driven) / blocking pool (default), never
            // inline on the runtime task: the collection is a sync resource and
            // anki's sync paths `block_on`, which a runtime worker forbids.
            // `dispatch_sync` is the one routing seam.
            let opened = runtime::dispatch_sync(move || {
                CollectionCore::open(&collection_path).map(Arc::new)
            })
            .await;
            let core = match opened {
                Ok(core) => core,
                Err(e) => {
                    let _ = opened_tx.send(Err(e));
                    return;
                }
            };
            let _ = opened_tx.send(Ok(Arc::clone(&core)));
            // Each job runs on the sync pool too, awaited here so serialization
            // (one job at a time, FIFO) is preserved by construction.
            while let Some(job) = jobs_rx.recv().await {
                let _ = runtime::dispatch_sync(move || {
                    job();
                    Ok::<(), NativeError>(())
                })
                .await;
            }
        });
        let core = opened_rx
            .await
            .map_err(|_| NativeError::internal("the collection actor dropped the open job"))??;
        Ok(Self {
            core,
            jobs: Mutex::new(Some(jobs_tx)),
            actor: Mutex::new(Some(actor)),
        })
    }

    /// The actor around a PRE-BUILT store (compose): same loop, same
    /// serialization discipline — only construction differs (an injected
    /// impl is built by the host's assembly, not on the actor task; the
    /// path-opening convenience above keeps anki construction inside it).
    pub fn from_store(core: Arc<dyn Collection>) -> Self {
        let (jobs_tx, mut jobs_rx) = tokio::sync::mpsc::unbounded_channel::<Job>();
        let actor = runtime::handle().spawn(async move {
            while let Some(job) = jobs_rx.recv().await {
                // Sync-pool routed, awaited to keep FIFO serialization.
                let _ = runtime::dispatch_sync(move || {
                    job();
                    Ok::<(), NativeError>(())
                })
                .await;
            }
        });
        Self {
            core,
            jobs: Mutex::new(Some(jobs_tx)),
            actor: Mutex::new(Some(actor)),
        }
    }

    /// A live sender, or the actor-is-gone error (post-shutdown).
    fn sender(&self) -> NativeResult<tokio::sync::mpsc::UnboundedSender<Job>> {
        self.jobs
            .lock()
            .expect("jobs slot poisoned")
            .clone()
            .ok_or_else(|| NativeError::internal("the collection actor is gone"))
    }

    /// Run a job against the collection, serialized; the await IS the
    /// transition point ops chain continuations onto. Re-acquires first if
    /// idle-released (open-on-demand, kernel-side — so a cooperative
    /// release between any two jobs self-heals; contention surfaces as the
    /// BUSY tier from `ensure_open`).
    ///
    /// # Errors
    ///
    /// Returns an error if the actor is gone (post-shutdown), the collection
    /// cannot be (re)opened (the BUSY tier), or the job is dropped before it
    /// returns.
    pub async fn run<T: Send + 'static>(
        &self,
        job: impl FnOnce(&dyn Collection) -> T + Send + 'static,
    ) -> NativeResult<T> {
        let core = Arc::clone(&self.core);
        let (tx, rx) = oneshot::channel();
        self.sender()?
            .send(Box::new(move || {
                let _ = tx.send(core.ensure_open().map(|_| job(&*core)));
            }))
            .map_err(|_| NativeError::internal("the collection actor is gone"))?;
        rx.await
            .map_err(|_| NativeError::internal("executor dropped a collection job"))?
    }

    /// Close the collection on the actor. Idempotent: once the actor is
    /// drained (a prior close/shutdown) the collection went down with it, so
    /// already-gone is the already-closed outcome, not an error — typed here
    /// rather than string-matched by the caller. Mirrors `run` except
    /// for that one rule.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection cannot be (re)opened to close it, or
    /// the close job is dropped before it returns; an already-drained actor is
    /// the already-closed success, not an error.
    pub async fn close(&self) -> NativeResult<()> {
        let Ok(sender) = self.sender() else {
            return Ok(());
        };
        let core = Arc::clone(&self.core);
        let (tx, rx) = oneshot::channel();
        let job: Job = Box::new(move || {
            let _ = tx.send(core.ensure_open().and_then(|_| core.close()));
        });
        if sender.send(job).is_err() {
            return Ok(()); // drained while queueing: same already-closed outcome
        }
        rx.await
            .map_err(|_| NativeError::internal("executor dropped a collection job"))?
    }

    /// Drain the actor: take-and-drop the sender (the channel closes; the
    /// loop ends once queued jobs ran) and await the task — close returns
    /// with nothing in flight (the interpreter-teardown guard). Idempotent.
    ///
    /// # Panics
    ///
    /// Panics if the jobs or actor slot mutex is poisoned (a prior holder
    /// panicked).
    pub async fn shutdown(&self) {
        drop(self.jobs.lock().expect("jobs slot poisoned").take());
        let handle = self.actor.lock().expect("actor slot poisoned").take();
        if let Some(handle) = handle {
            let _ = handle.await;
        }
    }

    /// The shared core, for a harness that runs its own (executor-disciplined)
    /// direct ops over the same collection the kernel owns.
    pub fn core_arc(&self) -> Arc<dyn Collection> {
        Arc::clone(&self.core)
    }
}

/// One fused search hit: note id, fused score, per-signal 1-based ranks.
#[derive(Debug, Clone)]
pub struct KernelHit {
    /// The matched note's id.
    pub note_id: i64,
    /// The RRF-fused score.
    pub score: f64,
    /// Per-signal `(name, 1-based rank)` provenance.
    pub signals: Vec<(String, i64)>,
}

/// One note in a bulk upsert batch — the kernel's batch unit (every kernel op
/// is batch-shaped; the single-note call is sugar over a batch of one).
#[derive(Debug, Clone)]
pub struct NoteSpec {
    /// The note type to create the note under.
    pub notetype_id: i64,
    /// The deck the note's cards land in.
    pub deck_id: i64,
    /// The note's field values, in note-type field order.
    pub fields: Vec<String>,
    /// The note's tags.
    pub tags: Vec<String>,
}

/// The kernel: one open collection + the index orchestrator (which owns and
/// maintains the engine) + the derived store + fusion, every op an idiomatic
/// async fn on the kernel's own runtime. No transport, no Python.
/// Index maintenance is **kernel-internal**:
/// upserts/deletes keep the orchestrator's vectors, fingerprints, and
/// watermarks current, and the debounced saver (tokio::time)
/// bounds what a crash can discard.
pub struct Kernel {
    /// Arc so the tag refresher's background task can read through the
    /// actor without holding the kernel itself.
    collection: Arc<SerializedCollection>,
    /// The N-space index coordinator: one orchestrator+saver per
    /// embedding space, keyed by content fingerprint. The PRIMARY space lives
    /// at the base index dir directly (the in-place migration rule), so a
    /// single-space deployment is byte-identical to the single
    /// orchestrator. The index/search paths consume [`IndexSet::primary`];
    /// removal + the watermark advance fan out to every space.
    index_set: Arc<index_set::IndexSet>,
    derived: Arc<dyn DerivedStore>,
    /// The attachable embedding spaces (the first registry slot, an ordered
    /// SET): swappable at runtime — the harness attaches on
    /// embedding start, detaches on stop, and a model swap is detach + attach.
    /// Ops that need embedding degrade (lexical-only search,
    /// unindexed-but-created upserts) when the set is empty, mirroring the
    /// Python host's gating. Arc so the tag refresher shares the gate.
    /// The index/search paths consume the PRIMARY text space
    /// ([`EmbedSpaces::primary`]) so single-space stays byte-identical.
    embed: Arc<RwLock<EmbedSpaces>>,
    /// Tag-centroid state: the live key→tag map for the engine's
    /// `tag.text` space + the hygiene knobs. Centroids refresh in the
    /// background after membership-relevant index ops, coalesced under
    /// write bursts; boot/rebuild paths refresh synchronously.
    tag_keys: Arc<tag_centroids::TagKeyMap>,
    tag_refresh: Arc<tag_centroids::TagRefresher>,
    /// The recognition services (the second registry slot):
    /// OCR/ASR/describe engines the harness attaches at runtime, exactly like
    /// the embed slot — but **keyed by purpose** so OCR, ASR, and VLM
    /// describe can be attached independently, each sweeping its own pending
    /// set / source / fingerprint / destination. The kernel runs the pipeline
    /// over whatever is registered; recognition for a purpose is simply off
    /// when its slot is empty.
    recognize: RwLock<BTreeMap<recognize::RecognitionPurpose, Arc<RecognizeService>>>,
    recognition_gate: recognize::RecognitionGate,
    /// Bounds concurrent LONG-RUNNING recognition dispatches: a VLM
    /// describe / ASR call parks a blocking-pool thread for its whole duration
    /// while holding model residency, so an unbounded fan-out (several
    /// purposes, several collections sharing one runtime, or a future
    /// batch-parallel sweep) could starve the pool. Each long-running
    /// `recognize()` acquires a permit; OCR never touches it. A `Semaphore`
    /// (not a batch-size cap) because it bounds CONCURRENCY without shrinking a
    /// single sweep's batch — throughput per call is unchanged.
    slow_recognition: Arc<tokio::sync::Semaphore>,
    /// The persistent ingest actor: the single writer of the vector index,
    /// the derived store, and the watermarks. Maintained collection writes
    /// enqueue `{ids, col.mod, kind}` INSIDE their write job and return; the
    /// actor drains in FIFO (== `col.mod`) order. Bulk ops (reindex/rebuild/
    /// recognition store) ride the same channel as awaited jobs, so every
    /// index/derived mutation serializes through one consumer. See [`ingest`].
    ingest: ingest::IngestHandle,
    /// The per-secondary-space cross-space IMAGE activation floor,
    /// keyed by space key. A dedicated CLIP secondary is image-only
    /// (`WriteMode::ImageOnly`), so the engine's own intra-modal calibration —
    /// which samples a space's *text* sub-index as pseudo-queries — finds no
    /// text vectors there and yields no floor. This map is the harness-driven
    /// replacement: calibrated by CLIP-text-embedding a sample of note texts
    /// (the live dual encoder's text side) and firing them at the secondary's
    /// image vectors, the intra-modal method with the pseudo-queries sourced
    /// from the encoder instead of stored vectors. Recomputed at every (re)build / model
    /// change; `build_cross_space` reads it to gate each secondary's `image`
    /// ranking. Empty until the first calibration (the floor is then a no-op
    /// and only the relative gate applies — today's behaviour).
    secondary_floors: Arc<RwLock<BTreeMap<String, f64>>>,
}

/// One attached recognition capability: the engine + the media resolver it
/// reads bytes through (independent of the embed slot — an OCR-only
/// deployment has no image embedder).
pub struct RecognizeService {
    /// The recognition engine (OCR/ASR/describe).
    pub recognizer: Arc<dyn Recognizer>,
    /// The media resolver the engine reads bytes through.
    pub resolver: Arc<dyn ImageResolver>,
}

/// How many long-running recognition dispatches (VLM describe / ASR) may run
/// concurrently across the kernel. A small bound: these calls park a
/// blocking-pool thread for their whole duration while holding model
/// residency, so the ceiling protects the pool from an unbounded fan-out
/// (multiple purposes, several collections sharing the runtime, a future
/// batch-parallel sweep). 2 keeps a describe and an ASR sweep able to overlap
/// (different engines) without letting either multiply without bound; OCR is
/// fast/bounded and never acquires a permit.
pub const SLOW_RECOGNITION_CONCURRENCY: usize = 2;

/// The derived-store source recognized image text lands under.
pub const OCR_SOURCE: &str = "ocr";
/// The derived-store meta key holding the recognizer fingerprint.
pub const RECOGNIZER_FINGERPRINT_KEY: &str = "recognizer_fingerprint";

/// One attached embedding capability: the text embedder + optionally its
/// image half (present only for an image-advertising backend with a media
/// resolver).
pub struct EmbedService {
    /// The text embedder.
    pub embedder: Arc<dyn Embedder>,
    /// The image half (present only for an image-advertising backend with a
    /// media resolver).
    pub images: Option<KernelImages>,
}

impl EmbedService {
    /// The image pair as the borrow shape the orchestrator ops take.
    fn images_pair(&self) -> Option<(&dyn ImageEmbedder, &dyn ImageResolver)> {
        self.images.as_ref().map(|(e, r)| (&**e, &**r))
    }
}

/// The injected image pair: who embeds image bytes + who resolves filenames
/// to bytes (lazily, at embed time).
pub type KernelImages = (Box<dyn ImageEmbedder>, Box<dyn ImageResolver>);

pub(crate) const FIELD_SOURCE: &str = "field";

/// A shared, pre-enumerated `(note_id, media_names)` set for one media kind:
/// the multi-purpose recognition sweep enumerates each kind once and
/// hands this Arc to every same-kind purpose's sweep, so the collection is
/// never re-scanned per purpose.
type MediaRefs = Arc<[(i64, Vec<String>)]>;

/// The NOTE-item vector spaces: every note search is scoped to these,
/// so other entity kinds sharing the engine (per-(tag, modality) centroids in
/// `tag.*` spaces) can never surface a non-note key from a note query — the
/// no-leakage property is structural, not a post-filter.
pub const NOTE_MODALITIES: &[&str] = &["text", "image"];

/// The per-modality tag-centroid spaces: `tag.text` holds the
/// renormalized mean of member notes' TEXT vectors per tag (never a
/// cross-modal mean — the modality gap makes one semantically empty).
pub const TAG_TEXT_SPACE: &str = "tag.text";

/// Deterministically truncate `items` to at most `cap` via a PARTIAL LCG
/// Fisher-Yates: only the first `cap` slots are drawn, so a large
/// collection isn't fully shuffled to take the sample. Stable across runs (a
/// fixed seed) — the calibration stats are statistical, never byte-pinned.
/// Mirrors the engine's own sampler (`engine.rs::calibrate_activation`).
fn deterministic_sample<T>(items: &mut Vec<T>, cap: usize) {
    let n = items.len();
    if n <= cap {
        return;
    }
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in 0..cap {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = i + ((rng >> 33) as usize % (n - i));
        items.swap(i, j);
    }
    items.truncate(cap);
}

impl Kernel {
    /// Open a collection and its sidecar stores (cache_dir holds the derived
    /// store and the index files, like the Python host's cache layout).
    /// Scheduling AND timing are the kernel's own: the owned runtime
    /// spawns the collection actor, and the debounced index flush rides
    /// tokio::time unconditionally. Saver defaults apply; a host with tuning
    /// flags uses [`Self::open_with`].
    ///
    /// # Errors
    ///
    /// Returns an error if the cache dir cannot be created, the collection
    /// cannot be opened, or a sidecar store fails to open.
    pub async fn open(collection_path: &str, cache_dir: &str) -> NativeResult<Self> {
        Self::open_with(collection_path, cache_dir, None, None).await
    }

    /// [`Self::open`] with the index-flush tuning the host's
    /// `--index-save-*` flags carry: `save_delay` is the idle
    /// debounce in seconds, `save_threshold` the unsaved-change count that
    /// forces an immediate flush. `None` = the built-in default.
    ///
    /// # Errors
    ///
    /// Returns an error if a cache/derived/index dir cannot be created, a path
    /// is non-UTF-8, the collection cannot be opened, or a sidecar store fails
    /// to open.
    pub async fn open_with(
        collection_path: &str,
        cache_dir: &str,
        save_delay: Option<f64>,
        save_threshold: Option<u64>,
    ) -> NativeResult<Self> {
        // The derived store opens its file under cache_dir before assemble
        // runs, so the dir must exist first (assemble re-creates idempotently
        // for composed callers).
        std::fs::create_dir_all(cache_dir).context(ErrorKind::Internal, "cache dir")?;
        let collection = Arc::new(SerializedCollection::open(collection_path.to_string()).await?);
        let engine: Arc<dyn VectorIndex> = Arc::new(MultiModalIndex::new(
            NOTE_MODALITIES
                .iter()
                .map(|m| m.to_string())
                .chain(std::iter::once(TAG_TEXT_SPACE.to_string()))
                .collect(),
        )?);
        // The derived store is namespaced per collection, mirroring the
        // index: `<cache_dir>/derived/<namespace>/shrike.db`, so a daemon
        // serving several collections never shares one `shrike.db` (which would
        // cross-contaminate substring/fuzzy/OCR search). Migrate an existing
        // flat `<cache_dir>/shrike.db` into this collection's namespace first,
        // so the single-collection user keeps their built derived data.
        shrike_cache::migrate_flat_derived(cache_dir, collection_path);
        let derived_path = shrike_cache::derived_db_path(cache_dir, collection_path);
        if let Some(parent) = derived_path.parent() {
            std::fs::create_dir_all(parent).context(ErrorKind::Internal, "derived dir")?;
        }
        let derived_path = derived_path.to_str().ok_or_else(|| {
            NativeError::internal(format!(
                "non-UTF-8 derived path: {}",
                derived_path.display()
            ))
        })?;
        let derived: Arc<dyn DerivedStore> = Arc::new(DerivedEngine::open(
            derived_path,
            DerivedEngine::SCHEMA_VERSION,
        )?);
        tracing::debug!(collection = collection_path, "kernel opened");
        // The vector index is namespaced per collection: each collection
        // gets its own `<cache_dir>/index/<path-derived-id>/` so a daemon
        // serving several collections never collides their indexes. The layout
        // also migrates an existing flat (single-collection) layout into this
        // collection's namespace, losslessly, so the long-standing text-only
        // user never pays a spurious rebuild on upgrade.
        let layout = shrike_cache::IndexLayout::for_collection(cache_dir, collection_path);
        Self::assemble(
            collection,
            engine,
            derived,
            cache_dir,
            layout,
            save_delay,
            save_threshold,
        )
    }

    /// The injection seam: a kernel over PRE-BUILT stores — the
    /// deployment ladder's composition point (remote/platform impls swap in
    /// here; [`Self::open`] is the all-local convenience over it). The
    /// collection arrives as the bare store; the kernel wraps it in its own
    /// task-actor (serialization is kernel policy, not the store's).
    ///
    /// # Errors
    ///
    /// Returns an error if the cache/index dir cannot be created during
    /// assembly.
    pub fn compose(
        collection: Arc<dyn Collection>,
        index: Arc<dyn VectorIndex>,
        derived: Arc<dyn DerivedStore>,
        cache_dir: &str,
        save_delay: Option<f64>,
        save_threshold: Option<u64>,
    ) -> NativeResult<Self> {
        let collection = Arc::new(SerializedCollection::from_store(collection));
        // The injection seam carries pre-built stores and no collection path,
        // so it can't derive a per-collection index namespace: the index
        // stays flat under `cache_dir`. The composing caller owns multiplexing
        // if it ever serves several collections through this seam.
        let layout = shrike_cache::IndexLayout::flat(cache_dir);
        Self::assemble(
            collection,
            index,
            derived,
            cache_dir,
            layout,
            save_delay,
            save_threshold,
        )
    }

    fn assemble(
        collection: Arc<SerializedCollection>,
        engine: Arc<dyn VectorIndex>,
        derived: Arc<dyn DerivedStore>,
        cache_dir: &str,
        index_layout: shrike_cache::IndexLayout,
        save_delay: Option<f64>,
        save_threshold: Option<u64>,
    ) -> NativeResult<Self> {
        std::fs::create_dir_all(cache_dir).context(ErrorKind::Internal, "cache dir")?;
        std::fs::create_dir_all(&index_layout.dir).context(ErrorKind::Internal, "index dir")?;
        // The N-space index coordinator: the PRIMARY space is the engine
        // built/injected above, opened at the base index dir DIRECTLY (no
        // subdir → the in-place migration rule, byte-identical to the single space).
        // Its modalities are the engine's own (NOTE_MODALITIES + tag.text for an
        // opened kernel; the injected engine's for the compose seam). Secondary
        // spaces are built by the factory over their own modalities.
        let primary_modalities = engine.modality_names();
        let engine_factory: index_set::EngineFactory = Arc::new(|mods: &[String]| {
            let e: Arc<dyn VectorIndex> = Arc::new(MultiModalIndex::new(mods.to_vec())?);
            Ok(e)
        });
        let index_set = index_set::IndexSet::open(
            index_layout.dir,
            index_layout.owner,
            engine,
            primary_modalities,
            save_delay.unwrap_or(index_orchestrator::DEFAULT_SAVE_DELAY),
            save_threshold.unwrap_or(index_orchestrator::DEFAULT_SAVE_THRESHOLD),
            engine_factory,
        )?;
        let embed: Arc<RwLock<EmbedSpaces>> = Arc::new(RwLock::new(EmbedSpaces::default()));
        let tag_keys = Arc::new(tag_centroids::TagKeyMap::default());
        let tag_config = tag_centroids::TagCentroidConfig::default();
        // Tag centroids bind to the PRIMARY/dedicated text space's engine +
        // saver: a pure function of THAT space's text vectors, never
        // fanned out. The primary's engine/saver Arcs are fixed for the kernel's
        // life (only secondaries are ever added), so these stay valid.
        let tag_refresh = tag_centroids::TagRefresher::new(
            Arc::clone(&collection),
            index_set.tag_engine(),
            Arc::clone(&tag_keys),
            tag_config.clone(),
            index_set.primary_saver(),
            Arc::clone(&embed),
        );
        let recognition_gate = recognize::RecognitionGate::default();
        // The single writer over the shared sub-stores; the kernel keeps the
        // same Arcs for its concurrent read paths (search/status). All
        // index/derived mutation funnels through the spawned drain task.
        let ingestor = ingest::Ingestor::new(
            Arc::clone(&collection),
            Arc::clone(&index_set),
            Arc::clone(&derived),
            Arc::clone(&embed),
            recognition_gate.clone(),
            Arc::clone(&tag_keys),
            Arc::clone(&tag_refresh),
        );
        let ingest = ingest::spawn(ingestor);
        Ok(Self {
            collection,
            index_set,
            derived,
            embed,
            tag_keys,
            tag_refresh,
            recognize: RwLock::new(BTreeMap::new()),
            recognition_gate,
            slow_recognition: Arc::new(tokio::sync::Semaphore::new(SLOW_RECOGNITION_CONCURRENCY)),
            ingest,
            secondary_floors: Arc::new(RwLock::new(BTreeMap::new())),
        })
    }

    /// Attach (or swap) the OCR recognition service — the slot pattern,
    /// second instance. The OCR-defaulting convenience over
    /// [`attach_recognizer_with`]: existing hosts and kernel tests keep
    /// the single-arg shape and target the OCR purpose. The harness follows up
    /// by driving the pending sweep.
    pub fn attach_recognizer(
        &self,
        recognizer: Arc<dyn Recognizer>,
        resolver: Arc<dyn ImageResolver>,
    ) {
        self.attach_recognizer_with(recognize::RecognitionPurpose::Ocr, recognizer, resolver);
    }

    /// Attach (or swap) the recognition service for a specific purpose
    /// — OCR, ASR, or VLM describe, each routed to its own pending set /
    /// source / fingerprint / destination by the sweep.
    ///
    /// # Panics
    ///
    /// Panics if the recognize-slot lock is poisoned (a prior holder panicked).
    pub fn attach_recognizer_with(
        &self,
        purpose: recognize::RecognitionPurpose,
        recognizer: Arc<dyn Recognizer>,
        resolver: Arc<dyn ImageResolver>,
    ) {
        self.recognize
            .write()
            .expect("recognize slot poisoned")
            .insert(
                purpose,
                Arc::new(RecognizeService {
                    recognizer,
                    resolver,
                }),
            );
    }

    /// Detach the OCR recognition service (the OCR-defaulting convenience).
    /// Already-derived text stays (it remains valid output of the engine that
    /// produced it); only new recognition stops.
    pub fn detach_recognizer(&self) {
        self.detach_recognizer_for(recognize::RecognitionPurpose::Ocr);
    }

    /// Detach the recognition service for a specific purpose.
    ///
    /// # Panics
    ///
    /// Panics if the recognize-slot lock is poisoned (a prior holder panicked).
    pub fn detach_recognizer_for(&self, purpose: recognize::RecognitionPurpose) {
        self.recognize
            .write()
            .expect("recognize slot poisoned")
            .remove(&purpose);
    }

    /// The OCR recognition service, if attached (the OCR-defaulting
    /// convenience).
    pub fn recognize_service(&self) -> Option<Arc<RecognizeService>> {
        self.recognize_service_for(recognize::RecognitionPurpose::Ocr)
    }

    /// The recognition service for a specific purpose, if attached.
    ///
    /// # Panics
    ///
    /// Panics if the recognize-slot lock is poisoned (a prior holder panicked).
    pub fn recognize_service_for(
        &self,
        purpose: recognize::RecognitionPurpose,
    ) -> Option<Arc<RecognizeService>> {
        self.recognize
            .read()
            .expect("recognize slot poisoned")
            .get(&purpose)
            .cloned()
    }

    /// The purposes with a currently-attached recognizer (sorted) — the
    /// harness drives one sweep per attached purpose, and `/status` reports
    /// per-purpose state.
    ///
    /// # Panics
    ///
    /// Panics if the recognize-slot lock is poisoned (a prior holder panicked).
    pub fn attached_recognition_purposes(&self) -> Vec<recognize::RecognitionPurpose> {
        self.recognize
            .read()
            .expect("recognize slot poisoned")
            .keys()
            .copied()
            .collect()
    }

    /// Every recognition purpose, in sweep order — the basis for the
    /// vector-minting and hidden-lexical source sets below (so adding a
    /// purpose updates both without a second edit).
    const ALL_PURPOSES: &'static [recognize::RecognitionPurpose] = &[
        recognize::RecognitionPurpose::Ocr,
        recognize::RecognitionPurpose::Describe,
        recognize::RecognitionPurpose::Asr,
    ];

    /// The derived `source` strings whose rows are vector-minting: the
    /// union of every purpose's source (all recognition purposes mint
    /// vectors — only the LEXICAL surfaces differ). `compose_embed_inputs`
    /// reads OCR/ASR/VLM recognized text from these.
    pub(crate) fn vector_minting_sources() -> Vec<&'static str> {
        Self::ALL_PURPOSES.iter().map(|p| p.source()).collect()
    }

    /// The derived `source` strings HIDDEN from the lexical (substring/fuzzy)
    /// surfaces — every [`recognize::Destination::VectorOnly`]
    /// purpose's source. Passed into the derived store so a VLM-describe row
    /// is stored (for provenance + reconcile) but never reachable via
    /// literal/typo search.
    pub fn hidden_lexical_sources() -> Vec<&'static str> {
        Self::ALL_PURPOSES
            .iter()
            .filter(|p| p.destination() == recognize::Destination::VectorOnly)
            .map(|p| p.source())
            .collect()
    }

    /// The live tag-key map (key → tag string) for the `tag.text` space.
    pub fn tag_keys(&self) -> &tag_centroids::TagKeyMap {
        &self.tag_keys
    }

    /// Recompute every tag centroid from the engine's current text vectors +
    /// one membership pass. Cheap (hundreds of tags, in-memory vector
    /// reads); runs at the tail of every index-changing op and is a no-op
    /// shortcut when no embedder is attached (no text vectors to mean).
    ///
    /// # Errors
    ///
    /// Returns an error if the collection read fails or the engine rejects the
    /// `tag.text` rebuild.
    pub async fn refresh_tag_centroids(&self) -> NativeResult<usize> {
        // Runs on the ingest actor (the sole writer of the `tag.text` space).
        self.ingest.refresh_tag_centroids().await
    }

    /// Await the background tag refresher fully quiescent — no run armed, in
    /// flight, or pending a coalesced re-run. A test barrier: it guarantees no
    /// late background recompute can still fire and read a collection mutation
    /// the test makes AFTER this returns. (The recompute reads the collection
    /// BEFORE taking its exclusion lock, so a synchronous `refresh_tag_centroids`
    /// alone can't fence a not-yet-started background run.)
    #[cfg(test)]
    async fn await_tag_quiesce(&self) {
        // Await the REAL quiescence event, unbounded — no iteration cap to race a
        // starved scheduler. A refresher that never settles hangs and Bazel's
        // per-test timeout catches it; a slow one just takes another poll.
        while !self.tag_refresh.is_idle() {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }

    /// Attach an embedding service (embedding start / model swap) — the N=1
    /// convenience over [`attach_embedder_space`]. The space key is read from
    /// the embedder's own fingerprint (the CONTENT fingerprint), so a
    /// re-attach of the same model replaces its space in place rather than
    /// stacking a duplicate. The orchestrator flips back to ready if it was
    /// only unavailable; the harness follows up with `reindex_if_needed` (a
    /// model change is drift).
    ///
    /// Existing hosts/tests keep this single-embedder shape: with one declared
    /// embedder the set holds exactly one space and [`embed_service`] returns
    /// it, so the index/search paths stay byte-identical to the single-slot
    /// era.
    pub fn attach_embedder(&self, embedder: Arc<dyn Embedder>, images: Option<KernelImages>) {
        let key = embedder.fingerprint();
        self.attach_embedder_space(key, embedder, images);
    }

    /// Attach (or replace) the embedding space keyed by `key` — the
    /// general by-space-key entry the harness fan-out and the bindings drive.
    /// A space whose key matches an already-attached one is replaced in place
    /// (a model swap that keeps the fingerprint, or a re-attach); a new key is
    /// appended in declaration order. `None` is a keyless backend (it never
    /// collides — each keyless attach is a fresh slot).
    ///
    /// # Panics
    ///
    /// Panics if the embed-slot lock is poisoned (a prior holder panicked).
    pub fn attach_embedder_space(
        &self,
        key: Option<String>,
        embedder: Arc<dyn Embedder>,
        images: Option<KernelImages>,
    ) {
        let has_images = images.is_some();
        self.embed
            .write()
            .expect("embed slot poisoned")
            .attach(key.clone(), Arc::new(EmbedService { embedder, images }));
        // Lockstep with the index set: bind this embedding space's key to
        // an index space. The FIRST keyed attach claims the un-keyed PRIMARY
        // index space (no new dir → in-place migration); a NEW key materializes
        // a secondary orchestrator. A keyless backend (`None`) leaves the index
        // set on the primary (the degenerate single-space path). The index/
        // search path consumes the primary.
        if let Some(key) = key.as_deref() {
            let modalities: Vec<String> = if has_images {
                NOTE_MODALITIES.iter().map(|m| m.to_string()).collect()
            } else {
                vec![NOTE_MODALITIES[0].to_string()]
            };
            if let Err(e) = self.index_set.bind_space(key, &modalities) {
                tracing::warn!(error = ?e, %key, "binding index space failed; serving on the primary");
            }
        }
        self.index_set.primary().mark_ready_if_loaded();
    }

    /// Detach EVERY embedding space (embedding stop) — the N=1 convenience.
    /// Flush the index (the on-disk vectors are kept) and mark it unavailable.
    /// The collection and the lexical search surfaces stay fully live.
    ///
    /// # Panics
    ///
    /// Panics if the embed-slot lock is poisoned (a prior holder panicked).
    pub fn detach_embedder(&self) {
        self.embed.write().expect("embed slot poisoned").clear();
        // Flush every space's index, then mark the primary (the served index)
        // unavailable. Secondary on-disk vectors are likewise kept.
        for orch in self.index_set.all_orchestrators() {
            let _ = orch.save();
        }
        self.index_set.primary().mark_unavailable();
    }

    /// Detach a single embedding space by key. Flushes + marks the
    /// index unavailable only when the LAST space leaves (the index serves the
    /// primary space; while another space remains, it stays live). Returns
    /// whether a space was removed.
    ///
    /// # Panics
    ///
    /// Panics if the embed-slot lock is poisoned (a prior holder panicked).
    pub fn detach_embedder_space(&self, key: &str) -> bool {
        let (removed, now_empty) = {
            let mut spaces = self.embed.write().expect("embed slot poisoned");
            let removed = spaces.detach(key);
            (removed, spaces.is_empty())
        };
        if now_empty {
            for orch in self.index_set.all_orchestrators() {
                let _ = orch.save();
            }
            self.index_set.primary().mark_unavailable();
        }
        removed
    }

    /// The PRIMARY text embedding space, if any — the one engine the
    /// index/search paths consume. With one declared embedder it is
    /// the sole attached space (byte-identical to the single slot).
    ///
    /// # Panics
    ///
    /// Panics if the embed-slot lock is poisoned (a prior holder panicked).
    pub fn embed_service(&self) -> Option<Arc<EmbedService>> {
        self.embed.read().expect("embed slot poisoned").primary()
    }

    /// The number of attached embedding spaces — status surface; the
    /// index path consumes only the primary.
    ///
    /// # Panics
    ///
    /// Panics if the embed-slot lock is poisoned (a prior holder panicked).
    pub fn embed_space_count(&self) -> usize {
        self.embed.read().expect("embed slot poisoned").len()
    }

    /// Every attached embedding space's service in declaration order —
    /// the carrier the query fan-out and status read. The index
    /// path does NOT consume this set (it stays on the primary).
    ///
    /// # Panics
    ///
    /// Panics if the embed-slot lock is poisoned (a prior holder panicked).
    pub fn embed_spaces(&self) -> Vec<Arc<EmbedService>> {
        self.embed.read().expect("embed slot poisoned").services()
    }

    /// The SECONDARY text-capable embedding spaces as `(key, service)` pairs
    /// — the cross-space query fan-out embeds the query into each and
    /// searches its own index space. EMPTY in the N=1 / single-space case.
    ///
    /// # Panics
    ///
    /// Panics if the embed-slot lock is poisoned (a prior holder panicked).
    pub fn secondary_embed_spaces(&self) -> Vec<(String, Arc<EmbedService>)> {
        self.embed
            .read()
            .expect("embed slot poisoned")
            .secondary_text_capable_keyed()
    }

    /// The SEPARATE image-primary write route, or `None`. Returns the
    /// image-primary's orchestrator + service ONLY when the per-modality-primary
    /// image space is a DISTINCT space from the text-primary — i.e. a dedicated
    /// text embedder + a separate CLIP (the no-omni deployment). When the
    /// text-primary IS image-capable (an omni/CLIP primary, the N=1 case), the
    /// text-primary already writes text+image, so there is NO separate image
    /// route and this is `None` → byte-identical to the single-space path.
    ///
    /// `index-narrow`: image items route to this ONE image-primary, never to
    /// every image-capable space.
    #[cfg(test)]
    fn image_only_route(
        &self,
    ) -> Option<(
        Arc<index_orchestrator::IndexOrchestrator>,
        Arc<EmbedService>,
    )> {
        let (text_key, image) = {
            let spaces = self.embed.read().expect("embed slot poisoned");
            (
                spaces.text_primary_keyed().and_then(|(k, _)| k),
                spaces.image_primary_keyed(),
            )
        };
        let (image_key, image_svc) = image?;
        let image_key = image_key?; // a keyless image space has no index space
                                    // Same space as the text-primary (omni primary) → no separate route.
        if Some(&image_key) == text_key.as_ref() {
            return None;
        }
        let orch = self.index_set.orchestrator_for(&image_key)?;
        Some((orch, image_svc))
    }

    /// Embed `source_texts` into every SECONDARY text-capable space and search
    /// each space's own index engine, producing the cross-space semantic inputs
    /// `actions::search_notes` fuses. Each secondary space embeds the
    /// query with ITS OWN model and searches ITS OWN engine (`search_by_modality`
    /// over the note-item modalities), and the per-source best query cosine is
    /// captured for the relative activation gate. EMPTY when there are no
    /// secondary spaces (the N=1 case) — the caller then runs exactly today's
    /// single-space path. `fetch_k` is the per-space rank cap.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding the query into a secondary space fails or
    /// its engine search fails.
    ///
    /// # Panics
    ///
    /// Panics if the `secondary_floors` lock is poisoned (a prior holder
    /// panicked).
    pub async fn build_cross_space(
        &self,
        source_texts: &[String],
        fetch_k: usize,
    ) -> NativeResult<Vec<actions::SpaceSemantic>> {
        let secondaries = self.secondary_embed_spaces();
        if secondaries.is_empty() || source_texts.is_empty() {
            return Ok(Vec::new());
        }
        let note_spaces: Vec<String> = NOTE_MODALITIES.iter().map(|m| m.to_string()).collect();
        let mut out: Vec<actions::SpaceSemantic> = Vec::new();
        for (key, svc) in secondaries {
            // The space must have a materialized index engine (bound at attach).
            let Some(orch) = self.index_set.orchestrator_for(&key) else {
                continue;
            };
            // Embed every source with THIS space's model, then rank per modality
            // on THIS space's engine.
            let qvectors = svc.embedder.embed(source_texts.to_vec()).await?;
            let engine = orch.engine_arc();
            let rows = engine.search_by_modality(&qvectors, fetch_k, Some(&note_spaces))?;
            let per_source: Vec<actions::SpaceSourceHits> = rows
                .into_iter()
                .map(|modality_hits| {
                    let best_query_cosine = actions::best_query_cosine_of(&modality_hits);
                    actions::SpaceSourceHits {
                        modality_hits,
                        best_query_cosine,
                    }
                })
                .collect();
            // This space's cross-space image floor: the harness-driven
            // calibration (CLIP-text pseudo-queries → image vectors), keyed by
            // space. A dedicated CLIP secondary is image-only, so the engine's
            // own intra-modal stats (text→image) are empty there —
            // `secondary_floors` is the replacement. `None` (uncalibrated, too few image samples)
            // → the floor is a no-op and only the relative gate applies.
            let image_floor = self
                .secondary_floors
                .read()
                .expect("secondary_floors poisoned")
                .get(&key)
                .copied();
            out.push(actions::SpaceSemantic {
                space_key: key,
                per_source,
                image_floor,
            });
        }
        Ok(out)
    }

    /// Recalibrate every SECONDARY cross-space's image activation floor
    /// and store it in `secondary_floors`. The harness drives this at every
    /// (re)build / model change (the embedder coupling lives here, where the
    /// CLIP backend Arcs are — engine.rs stays embedder-free).
    ///
    /// The method is the intra-modal one exactly, with the pseudo-queries
    /// sourced from the LIVE CLIP-text encoder rather than stored text vectors
    /// (a dedicated CLIP secondary indexes images only, so it has none): take a
    /// deterministic sample of image-bearing notes, CLIP-text-embed each note's
    /// text through the secondary's OWN backend, search it against that space's
    /// image vectors, record the best NON-SELF cosine (exclude the note's own
    /// image), and set `floor = mean + margin·std` when ≥ `CALIB_MIN`
    /// samples land. A space with too few image notes gets no floor (the gate
    /// then rides the relative comparison alone). Returns the per-space derived
    /// floor (`None` = uncalibrated) for the harness to log / surface.
    ///
    /// # Errors
    ///
    /// Returns an error if a collection read, query embedding, or engine search
    /// fails during calibration.
    ///
    /// # Panics
    ///
    /// Panics if the `secondary_floors` lock is poisoned (a prior holder
    /// panicked).
    pub async fn calibrate_secondary_floors(
        &self,
        margin: f64,
    ) -> NativeResult<Vec<(String, Option<f64>)>> {
        let secondaries = self.secondary_embed_spaces();
        if secondaries.is_empty() {
            self.secondary_floors
                .write()
                .expect("secondary_floors poisoned")
                .clear();
            return Ok(Vec::new());
        }
        // Sample image-bearing notes once (shared across spaces): the same
        // embed inputs a rebuild reads, filtered to notes that carry an image,
        // deterministically truncated to the calibration cap.
        let raw = self
            .collection
            .run(|core| -> NativeResult<_> {
                let ids = core.find_notes("")?;
                core.note_embed_inputs(&ids)
            })
            .await??;
        let inputs =
            ingest::compose_embed_inputs(&*self.derived, &self.recognition_gate, raw, None);
        let mut sample: Vec<(i64, String)> = inputs
            .into_iter()
            .filter(|i| !i.image_names.is_empty() && !i.text.trim().is_empty())
            .map(|i| (i.note_id, i.text))
            .collect();
        deterministic_sample(&mut sample, index_orchestrator::CALIB_SAMPLE);

        let note_spaces: Vec<String> = NOTE_MODALITIES.iter().map(|m| m.to_string()).collect();
        let mut derived: Vec<(String, Option<f64>)> = Vec::new();
        for (key, svc) in secondaries {
            let floor = self
                .calibrate_one_secondary_floor(&key, &svc, &sample, &note_spaces, margin)
                .await?;
            {
                let mut floors = self
                    .secondary_floors
                    .write()
                    .expect("secondary_floors poisoned");
                match floor {
                    Some(f) => {
                        floors.insert(key.clone(), f);
                    }
                    None => {
                        floors.remove(&key);
                    }
                }
            }
            derived.push((key, floor));
        }
        Ok(derived)
    }

    /// One secondary space's image floor: CLIP-text-embed the sample texts on
    /// this space's backend, search its image vectors, collect best non-self
    /// cosines, return `mean + margin·std` (or `None` below `CALIB_MIN`). No-op
    /// (`None`) when the space has no image vectors. `margin` is the harness-
    /// resolved `search.cross_space_fusion.margin` — the precision/recall
    /// dial; `ACTIVATION_MARGIN` (1.0) is its default.
    async fn calibrate_one_secondary_floor(
        &self,
        key: &str,
        svc: &EmbedService,
        sample: &[(i64, String)],
        note_spaces: &[String],
        margin: f64,
    ) -> NativeResult<Option<f64>> {
        let Some(orch) = self.index_set.orchestrator_for(key) else {
            return Ok(None);
        };
        let engine = orch.engine_arc();
        // The image sub-index must be non-empty (the space indexes images).
        if engine
            .modality_sizes()
            .iter()
            .all(|(m, n)| m != "image" || *n == 0)
        {
            return Ok(None);
        }
        if sample.len() < index_orchestrator::CALIB_MIN {
            return Ok(None);
        }
        // CLIP-text-embed every sample note's text on THIS space's encoder.
        let texts: Vec<String> = sample.iter().map(|(_, t)| t.clone()).collect();
        let qvectors = svc.embedder.embed(texts).await?;
        // CALIB_K so a pseudo-query whose own image is the nearest hit still has
        // a non-self hit to record (mirrors the engine's intra-modal sampling).
        let rows =
            engine.search_by_modality(&qvectors, index_orchestrator::CALIB_K, Some(note_spaces))?;
        let mut best_sims: Vec<f64> = Vec::with_capacity(sample.len());
        for ((self_id, _), modality_hits) in sample.iter().zip(rows.iter()) {
            let Some((keys, dists)) = modality_hits.get("image") else {
                continue;
            };
            // The best NON-SELF image hit (exclude the note's own image vector).
            if let Some((_, dist)) = keys.iter().zip(dists.iter()).find(|(k, _)| **k != *self_id) {
                best_sims.push(1.0 - *dist as f64);
            }
        }
        if best_sims.len() < index_orchestrator::CALIB_MIN {
            return Ok(None);
        }
        let n = best_sims.len() as f64;
        let mean = best_sims.iter().sum::<f64>() / n;
        let var = best_sims
            .iter()
            .map(|s| (s - mean) * (s - mean))
            .sum::<f64>()
            / n;
        // The one floor formula (`mean + margin·std`), shared with the
        // primary's via `actions::activation_floor`. The margin is the harness-
        // resolved dial; `ACTIVATION_MARGIN` is its 1.0 default.
        Ok(actions::activation_floor(Some((mean, var.sqrt())), margin))
    }

    /// The PRIMARY orchestrator (state, status, drift) — the harness's status
    /// surface and the one engine the index/search paths consume.
    /// With one declared embedder it is the sole space at the base dir,
    /// so the wire status + persistence are byte-identical to the single space.
    pub fn index(&self) -> Arc<index_orchestrator::IndexOrchestrator> {
        self.index_set.primary()
    }

    /// How many ingest-drain items/jobs the sole writer caught panicking. Zero
    /// in normal operation; non-zero means the drain hit an unexpected fault
    /// (most likely a poisoned lock), skipped that work, and survived — the
    /// `/status` degraded-writer signal.
    pub fn ingest_drain_panics(&self) -> u64 {
        self.ingest.drain_panics()
    }

    /// The N-space index coordinator — removal + the watermark advance
    /// fan out across it.
    pub fn index_set(&self) -> &index_set::IndexSet {
        &self.index_set
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
    ///
    /// # Errors
    ///
    /// Returns an error if the collection read, embedding, or an index
    /// reconcile/rebuild fails.
    #[must_use = "whether reindexing ran is the caller's signal to refresh status"]
    pub async fn reindex_if_needed(&self) -> NativeResult<bool> {
        // The reconcile runs on the ingest actor (the sole writer), serialized
        // with the hot path; the kernel serves reads while it runs.
        self.ingest.reindex_if_needed().await
    }

    /// Explicit FULL index rebuild (the `/index/rebuild` semantics): drop and
    /// re-embed everything — never the incremental path (reconcile is only
    /// the automatic drift route). Returns the note count. Errors Unavailable
    /// with no embedder attached.
    ///
    /// # Errors
    ///
    /// Returns `Unavailable` when no embedder is attached, or an error if the
    /// collection read, embedding, or the engine rebuild fails.
    pub async fn rebuild_index(&self) -> NativeResult<usize> {
        self.ingest.rebuild_index().await
    }

    /// Post-write maintenance for a set of created/updated notes: ONE read
    /// job for embed inputs + derived rows, one orchestrator add (replace
    /// semantics — when an embedder is attached; notes are created and
    /// lexically indexed regardless), per-note derived ingest, and the
    /// watermark advance. The shared tail of every upsert shape — public as
    /// `reindex_notes` for harness ops that edit note text outside the
    /// upsert ops (find/replace, note-type migration).
    ///
    /// # Errors
    ///
    /// Returns an error if the collection read, embedding, an index add, or the
    /// derived ingest fails.
    pub async fn reindex_notes(&self, written: &[i64]) -> NativeResult<()> {
        if written.is_empty() {
            return Ok(());
        }
        // No own collection write precedes this (the find/replace or note-type
        // migration already committed): read `col.mod` AND enqueue the
        // maintenance item in ONE collection job, so the enqueue is FIFO-ordered
        // with every concurrent write (queue order == col.mod order).
        let enq = self.ingest.enqueuer();
        let ids = written.to_vec();
        self.collection
            .run(move |core| -> NativeResult<()> {
                let col_mod = core.col_mod()?;
                enq.enqueue(ingest::IngestItem {
                    ids,
                    col_mod,
                    kind: ingest::MaintKind::Maintain,
                    membership_may_have_changed: false,
                });
                Ok(())
            })
            .await?
    }

    /// Full derived-text (FTS5) rebuild, entirely kernel-side: one collection
    /// job collects the field rows, the build runs on the blocking pool against
    /// the kernel's own engine — the rows never cross the FFI. Runs on the
    /// ingest actor (the sole writer), so a concurrent per-op derived ingest can
    /// no longer land inside the snapshot→build→prune window — the old
    /// commit-then-verify convergence loop is gone (#828 closed structurally).
    /// Returns `(row_count, col_mod)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection read or the derived-store build fails.
    pub async fn rebuild_derived(&self) -> NativeResult<(usize, i64)> {
        self.ingest.rebuild_derived().await
    }

    /// One bounded recognition sweep across EVERY attached purpose:
    /// for each of OCR / ASR / VLM-describe that has a recognizer
    /// attached, recognize up to `max_items` of ITS pending media, persist
    /// gated text + segments per its destination, and re-embed the affected
    /// notes so recognition vectors mint. Each purpose sweeps independently
    /// (its own pending set / source / fingerprint key / destination), and
    /// the per-purpose chunk-Err-aborts-before-persist contract holds per
    /// sweep — a down describe endpoint leaves its backlog intact without
    /// touching OCR. Returns an AGGREGATED report (summed counts, max
    /// remaining) so the harness's existing `remaining > 0` driver loop is
    /// unchanged; `Unavailable` only when NO purpose is attached.
    ///
    /// # Errors
    ///
    /// Returns an error if a collection read, a recognizer call (e.g. a down
    /// endpoint), or the post-recognition persist/re-embed fails for any
    /// purpose — which aborts the call with that purpose's backlog intact.
    pub async fn recognize_pending(
        &self,
        max_items: usize,
    ) -> NativeResult<recognize::SweepReport> {
        let purposes = self.attached_recognition_purposes();
        if purposes.is_empty() {
            return Ok(recognize::SweepReport::Unavailable);
        }
        // Enumerate each NEEDED media kind ONCE (no per-purpose
        // collection re-scan — OCR and describe are both Image, so a single
        // `note_image_refs` pass over the 100k-note collection serves both).
        // The shared Arc'd refs are handed to every purpose's sweep.
        let kinds: std::collections::BTreeSet<recognize::MediaKind> =
            purposes.iter().map(|p| p.media_kind()).collect();
        let mut kind_refs: BTreeMap<recognize::MediaKind, MediaRefs> = BTreeMap::new();
        for kind in kinds {
            let refs: MediaRefs = self.note_media_refs(kind).await?.into();
            kind_refs.insert(kind, refs);
        }
        // Each purpose's sweep is independent: an Err from one (a down
        // endpoint) propagates and aborts THIS call before the harness loops
        // again — that purpose's backlog stays pending, exactly the
        // chunk-Err-aborts-before-persist contract, now scoped per purpose.
        // `agg` is Some iff at least one purpose's sweep actually Ran (a
        // batch reached an engine) — counts summed, remaining max-ed.
        let mut agg: Option<(usize, usize, usize)> = None; // (recognized, stored, remaining)
        for purpose in purposes {
            let refs = Arc::clone(&kind_refs[&purpose.media_kind()]);
            let report = self
                .recognize_pending_for_refs(purpose, max_items, &refs)
                .await?;
            if let recognize::SweepReport::Ran {
                recognized,
                stored,
                remaining,
            } = report
            {
                let (r, s, rem) = agg.get_or_insert((0, 0, 0));
                *r += recognized;
                *s += stored;
                *rem = (*rem).max(remaining);
            }
        }
        match agg {
            Some((recognized, stored, remaining)) => Ok(recognize::SweepReport::Ran {
                recognized,
                stored,
                remaining,
            }),
            // Every attached purpose was Idle (nothing pending) — the sweep
            // had no work, the harness's driver stops.
            None => Ok(recognize::SweepReport::Idle),
        }
    }

    /// Drive bounded recognition sweeps to quiescence INSIDE the kernel: loop
    /// [`recognize_pending`](Self::recognize_pending) (each call keeps the
    /// bounded-batch yield so other actor jobs interleave) until nothing is
    /// pending, a no-progress batch, or `max_batches`. One FFI crossing instead
    /// of one per batch; the "stored ⇒ visible" boundary lives in the runtime
    /// that owns both stores. The loop logic — and its stop conditions — are the
    /// Python harness driver's, ported verbatim:
    ///
    /// - status != `ran` (Idle/Unavailable) OR `remaining == 0` → drained.
    /// - `recognized == 0` → no-progress STOP (an unreadable prefix of the
    ///   pending order; skipped items stay pending, so re-taking the same
    ///   window would loop forever — a later sweep trigger retries when the read
    ///   may have healed). Keyed on `recognized == 0`, NOT `stored == 0`: a
    ///   batch that recognized items but gated them all out did real work.
    /// - `batches >= max_batches` → ceiling (may leave pending).
    ///
    /// The per-purpose chunk-Err-aborts-before-persist contract is preserved by
    /// `recognize_pending`'s own `?`-propagation: a down endpoint aborts the run
    /// with its backlog intact.
    ///
    /// # Errors
    ///
    /// Returns an error if a batch's collection read, recognizer call, or
    /// persist/re-embed fails (aborting before persist, backlog intact).
    pub async fn recognize_all_pending(
        &self,
        batch_size: usize,
        max_batches: Option<usize>,
    ) -> NativeResult<recognize::SweepRunReport> {
        let mut total_stored = 0usize;
        let mut batches = 0usize;
        loop {
            let report = self.recognize_pending(batch_size).await?;
            match report {
                recognize::SweepReport::Ran {
                    recognized,
                    stored,
                    remaining,
                } => {
                    total_stored += stored;
                    batches += 1;
                    let drained = remaining == 0;
                    let no_progress = recognized == 0;
                    let ceiling = max_batches.is_some_and(|mb| batches >= mb);
                    if drained || no_progress || ceiling {
                        if no_progress && !drained {
                            tracing::warn!(
                                remaining,
                                "recognition sweep stopped on a no-progress batch (unreadable prefix)"
                            );
                        }
                        return Ok(recognize::SweepRunReport {
                            last: recognize::SweepReport::Ran {
                                recognized,
                                stored,
                                remaining,
                            },
                            total_stored,
                            batches,
                        });
                    }
                }
                // status != "ran" (Idle / Unavailable) — the loop's other stop.
                other => {
                    return Ok(recognize::SweepRunReport {
                        last: other,
                        total_stored,
                        batches,
                    });
                }
            }
        }
    }

    /// One bounded recognition sweep for a SINGLE purpose — the
    /// per-engine routing of the sweep. Pending = a resolvable
    /// media ref of this purpose's media kind with no row for THIS source AND
    /// no below-gate marker — or all of them after this purpose's
    /// recognizer-fingerprint changes (its own meta key, so an OCR upgrade
    /// never re-derives ASR/VLM and vice versa). Persists per the purpose's
    /// destination: a VectorOnly (VLM) row is still stored (for provenance +
    /// reconcile) but the kernel excludes its source from the lexical
    /// surfaces. On a recognizer chunk `Err`, the sweep aborts BEFORE
    /// persisting anything or advancing this purpose's fingerprint meta —
    /// everything stays pending and the next sweep retries (the load-bearing
    /// describe-engine contract, preserved per purpose).
    ///
    /// # Errors
    ///
    /// Returns an error if a collection read, the recognizer call, or the
    /// post-recognition persist/re-embed fails — which aborts the sweep before
    /// persisting, leaving the purpose's backlog pending.
    pub async fn recognize_pending_for(
        &self,
        purpose: recognize::RecognitionPurpose,
        max_items: usize,
    ) -> NativeResult<recognize::SweepReport> {
        // Enumerate this purpose's media kind once, then delegate. The
        // multi-purpose driver shares one enumeration across same-kind
        // purposes; this single-purpose entry point is the
        // test/binding convenience.
        if self.recognize_service_for(purpose).is_none() {
            return Ok(recognize::SweepReport::Unavailable);
        }
        let refs: MediaRefs = self.note_media_refs(purpose.media_kind()).await?.into();
        self.recognize_pending_for_refs(purpose, max_items, &refs)
            .await
    }

    /// [`recognize_pending_for`] over a PRE-ENUMERATED media-ref set:
    /// the multi-purpose driver enumerates each media kind once and shares the
    /// Arc'd refs across same-kind purposes, so a sweep never re-scans the
    /// collection per purpose.
    async fn recognize_pending_for_refs(
        &self,
        purpose: recognize::RecognitionPurpose,
        max_items: usize,
        raw: &[(i64, Vec<String>)],
    ) -> NativeResult<recognize::SweepReport> {
        let Some(svc) = self.recognize_service_for(purpose) else {
            return Ok(recognize::SweepReport::Unavailable);
        };
        let source = purpose.source();
        let fingerprint_key = purpose.fingerprint_key();

        // Fingerprint drift: a changed engine invalidates ALL of THIS
        // purpose's recognized text (the analog of the embedder's model_id
        // rebuild), keyed by this purpose's own meta key.
        let fingerprint = svc.recognizer.fingerprint().unwrap_or_default();
        let stored = self.derived.meta_get(fingerprint_key)?.unwrap_or_default();
        if !stored.is_empty() && stored != fingerprint {
            let stale: Vec<i64> = self
                .derived
                .refs_for_source(source)?
                .into_iter()
                .map(|(nid, _)| nid)
                .collect();
            if !stale.is_empty() {
                tracing::info!(
                    notes = stale.len(),
                    %source,
                    "recognizer fingerprint changed; invalidating recognized text"
                );
                self.derived.remove(&stale, Some(source))?;
            }
            // Below-gate markers ride the same invalidation: the new
            // engine may read what the old one couldn't, so gated items
            // re-enter the pending set exactly like stored rows re-derive.
            self.derived.clear_gated(source)?;
        }

        // Pending set: resolvable media of this purpose's kind without a row
        // for THIS source — and without a below-gate marker: an item
        // the gate dropped is DONE (its outcome can't change until the
        // fingerprint does), not pending, so it is never re-recognized and an
        // all-gated window converges instead of re-taking itself forever.
        // `raw` is the pre-enumerated (note_id, names) set shared across
        // same-kind purposes: the pending diff needs only the names.
        //
        // Done-set keyed `note_id -> {name}`: the per-pair probe no
        // longer clones the name for a `(i64, String)` lookup — it borrows
        // `name` against the note's set. Built ONCE here, then this call DRAINS
        // the whole pending set in bounded chunks below, so the O(collection)
        // enumeration + done-load is paid once per drain, not once per batch
        // (the harness loops `recognize_pending` until `remaining == 0`; a
        // per-batch re-enumeration was ~O(image-notes × backlog/batch)).
        let mut done: std::collections::HashMap<i64, std::collections::HashSet<String>> =
            std::collections::HashMap::new();
        for (nid, name) in self
            .derived
            .refs_for_source(source)?
            .into_iter()
            .chain(self.derived.gated_refs_for_source(source)?)
        {
            done.entry(nid).or_default().insert(name);
        }
        let mut pending: Vec<(i64, String)> = Vec::new();
        for (note_id, names) in raw {
            let note_done = done.get(note_id);
            for name in names {
                let already = note_done.is_some_and(|s| s.contains(name));
                if !already && svc.resolver.exists(name) {
                    pending.push((*note_id, name.clone()));
                }
            }
        }
        if pending.is_empty() {
            self.derived.meta_set(fingerprint_key, &fingerprint)?;
            return Ok(recognize::SweepReport::Idle);
        }

        // Drain the whole pending set in bounded chunks within THIS call:
        // each chunk dispatches at most `max_items` to the recognizer (the
        // load-bearing bounded-batch property — a chunk parks one blocking-pool
        // thread; the slow-recognition permit still bounds concurrency), and we
        // `.await` between chunks so collection ops interleave exactly as the
        // old per-`recognize_pending`-call yield did. A chunk that recognizes
        // NOTHING (an unreadable prefix of the pending order) stops the drain —
        // the same no-progress halt the harness applied between calls, now
        // applied between chunks; the leftover stays pending and a later sweep
        // trigger retries. A chunk `Err` (down endpoint) propagates and aborts
        // the drain before that chunk's persist, leaving its backlog pending.
        let total_pending = pending.len();
        let mut total_recognized = 0usize;
        let mut total_stored = 0usize;
        let mut drained = 0usize;
        for chunk in pending.chunks(max_items.max(1)) {
            let (recognized, stored) = self.recognize_chunk(&svc, purpose, source, chunk).await?;
            total_recognized += recognized;
            total_stored += stored;
            drained += chunk.len();
            if recognized == 0 {
                break; // no-progress: an unreadable prefix; retry next sweep.
            }
        }
        self.derived.meta_set(fingerprint_key, &fingerprint)?;
        Ok(recognize::SweepReport::Ran {
            recognized: total_recognized,
            stored: total_stored,
            remaining: total_pending.saturating_sub(drained),
        })
    }

    /// Recognize + persist ONE bounded chunk of pending `(note_id, name)` pairs
    /// for a purpose (the inner step of the drain loop). Returns
    /// `(recognized, stored)`. A recognizer `Err` propagates BEFORE any persist
    /// (the down-endpoint contract); the fingerprint meta is advanced by the
    /// caller once the drain completes.
    #[allow(clippy::type_complexity)]
    async fn recognize_chunk(
        &self,
        svc: &RecognizeService,
        purpose: recognize::RecognitionPurpose,
        source: &str,
        pending: &[(i64, String)],
    ) -> NativeResult<(usize, usize)> {
        // One pass, many consumers: recognize the batch, keep text AND
        // segments. A read that fails after the exists() check (TOCTOU
        // delete, transient error, resolver bug) SKIPS the item rather than
        // recognizing empty bytes — nothing is stored for it, so the
        // next sweep re-offers it. `sent` and `items` are built together so
        // the recognizer's output stays aligned with what was actually sent.
        let mut sent: Vec<(i64, String)> = Vec::with_capacity(pending.len());
        let mut items: Vec<MediaItem> = Vec::with_capacity(pending.len());
        for (note_id, name) in pending {
            match svc.resolver.read(name) {
                Some(bytes) => {
                    items.push(MediaItem::from_named(name, bytes));
                    sent.push((*note_id, name.clone()));
                }
                None => {
                    tracing::warn!(
                        media = %name,
                        note_id,
                        %source,
                        "media read failed after exists(); skipping until the next sweep"
                    );
                }
            }
        }
        // The chunk-Err propagates HERE — before any persist or fingerprint
        // advance below — so a down endpoint leaves the backlog intact.
        // A long-running purpose (describe/ASR) holds a concurrency permit
        // ONLY across the recognize() call: the call parks a
        // blocking-pool thread for its whole duration while holding model
        // residency, so the kernel bounds how many run at once. The permit is
        // released the instant recognition returns (before the cheap persist),
        // so it never delays anything but the next SLOW dispatch. OCR is fast
        // and never acquires — its dispatch is byte-identical.
        let recognitions = if items.is_empty() {
            Vec::new()
        } else {
            let _permit = if purpose.is_long_running() {
                Some(
                    self.slow_recognition
                        .acquire()
                        .await
                        .expect("slow-recognition semaphore never closes"),
                )
            } else {
                None
            };
            svc.recognizer.recognize(items).await?
        };
        if recognitions.len() != sent.len() {
            return Err(NativeError::internal(format!(
                "recognizer returned {} results for {} items",
                recognitions.len(),
                sent.len()
            )));
        }

        // Persist per note: ingest REPLACES a note's rows for the source, so
        // merge with what already exists. A gated-out item stores no text row
        // (a zero-text row would index "" — FTS5 pollution) but DOES persist
        // a below-gate marker, so the next sweep's pending diff counts
        // it done instead of re-recognizing it forever. Markers are cleared
        // with the rows on a fingerprint change (above), so an engine upgrade
        // re-judges them like everything else.
        // Scoped to the batch's notes: the merge previously read the
        // whole table per sweep.
        let sent_ids: Vec<i64> = sent
            .iter()
            .map(|(nid, _)| *nid)
            .collect::<std::collections::BTreeSet<i64>>()
            .into_iter()
            .collect();
        let mut existing: std::collections::HashMap<i64, Vec<(String, String)>> =
            std::collections::HashMap::new();
        for (nid, r, text) in self.derived.texts_for_source_for_notes(source, &sent_ids)? {
            existing.entry(nid).or_default().push((r, text));
        }
        let mut stored_count = 0usize;
        let mut touched: std::collections::BTreeMap<i64, Vec<(String, String)>> =
            std::collections::BTreeMap::new();
        let mut segments: Vec<(i64, String, String)> = Vec::new();
        let mut gated: Vec<(i64, String)> = Vec::new();
        for ((note_id, name), recognition) in sent.iter().zip(recognitions.iter()) {
            match self.recognition_gate.judge(recognition) {
                recognize::GateOutcome::Drop => {
                    // The below-gate marker is ALSO the negative cache for a
                    // per-item PERMANENT failure: an engine converts a
                    // 4xx-class rejection (this image is oversized/unsupported)
                    // into the empty recognition (`text="", confidence=0.0`),
                    // which gates as Drop here — so a permanently-failed item is
                    // judged once and not re-offered every sweep (expensive
                    // against a paid endpoint), and clears on a fingerprint
                    // change exactly like a stored row re-derives. This is
                    // deliberately the SAME path as a genuine below-substance
                    // drop (no sibling `failed` table): behaviour is identical
                    // and no consumer needs the diagnostic split. NB an
                    // ENDPOINT-level failure (transport/auth/exhausted retries)
                    // never reaches here — it Err's the chunk above, before any
                    // persist, leaving the whole backlog pending.
                    gated.push((*note_id, name.clone()));
                    continue;
                }
                recognize::GateOutcome::Lexical | recognize::GateOutcome::LexicalAndVector => {
                    touched
                        .entry(*note_id)
                        .or_insert_with(|| existing.get(note_id).cloned().unwrap_or_default())
                        .push((name.clone(), recognition.text.clone()));
                    if !recognition.segments.is_empty() {
                        let json = serde_json::to_string(&recognition.segments)
                            .context(ErrorKind::Internal, "segments")?;
                        segments.push((*note_id, name.clone(), json));
                    }
                    stored_count += 1;
                }
            }
        }
        let affected: Vec<i64> = touched.keys().copied().collect();
        // Hand the derived writes + the re-embed to the ingest actor: routing
        // them through the sole writer serializes them with rebuild, so a
        // rebuild's prune can no longer drop a row this sweep ingested (#828)
        // and the recognition vector add no longer races a reindex (#650/#644).
        // The affected notes re-embed because their hash now folds the
        // recognized text. The fingerprint meta is advanced by the DRAIN CALLER
        // once the whole drain completes — not here per chunk — so a mid-drain
        // abort never marks this purpose's recognition as up-to-date.
        self.ingest
            .store_recognition(ingest::RecognitionWrite {
                source: source.to_string(),
                touched: touched.into_iter().collect(),
                segments,
                gated,
                affected,
            })
            .await?;

        // `recognized` counts what was actually sent. A skipped (unreadable)
        // item stores nothing, so it STAYS PENDING; the drain loop stops on a
        // chunk that recognized nothing (an unreadable prefix), and a later
        // sweep trigger (boot, /reload, cooperative re-acquire) retries the read.
        Ok((sent.len(), stored_count))
    }

    /// `(note_id, media_names)` for every note referencing media of `kind` —
    /// the sweep's pending-set source, routed by media kind. Image refs
    /// come from `note_image_refs` (the `<img src>` extractor); audio refs from
    /// `note_sound_refs` (the `[sound:…]` extractor) — both scoped reads.
    async fn note_media_refs(
        &self,
        kind: recognize::MediaKind,
    ) -> NativeResult<Vec<(i64, Vec<String>)>> {
        match kind {
            recognize::MediaKind::Image => {
                self.collection.run(|core| core.note_image_refs()).await?
            }
            recognize::MediaKind::Audio => {
                self.collection.run(|core| core.note_sound_refs()).await?
            }
        }
    }

    /// The wire-shaped bulk upsert: the collection core's NAMED upsert
    /// — `id`?/`note_type`/`deck`/`fields` map/`tags`, create AND update,
    /// `dry_run`, typed per-item results — run as ONE collection job, then
    /// the kernel-internal index/derived maintenance over everything written.
    /// This is the op the MCP `upsert_notes` action rides.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection write job fails; the index/derived
    /// maintenance tail is best-effort (a failure there logs and still returns
    /// the per-item results).
    pub async fn upsert_notes_wire(
        &self,
        notes: Vec<shrike_schemas::NoteInput>,
        policy: DuplicatePolicy,
        dry_run: bool,
    ) -> NativeResult<Vec<shrike_schemas::UpsertNoteResult>> {
        let span = tracing::debug_span!("kernel.upsert_notes_wire", dry_run);
        async move {
            // Commit the write and enqueue the maintenance item in the SAME
            // collection job: the actor's FIFO then makes queue order == col.mod
            // order (the load-bearing ordering invariant — the enqueue must NOT
            // slip into a post-await continuation). The slow embed runs off the
            // drain, so this write never waits on it. A dry run mutates nothing,
            // so it enqueues nothing.
            let enq = self.ingest.enqueuer();
            let results = self
                .collection
                .run(move |core| -> NativeResult<_> {
                    let results = core.upsert_notes(&notes, policy, dry_run)?;
                    if !dry_run {
                        // Typed outcomes: written ids come straight off the
                        // variants — no parse-own-output round-trip.
                        let written: Vec<i64> = results
                            .iter()
                            .filter_map(|r| match r {
                                shrike_schemas::UpsertNoteResult::Created { id, .. }
                                | shrike_schemas::UpsertNoteResult::Updated { id, .. } => Some(*id),
                                _ => None,
                            })
                            .collect();
                        if !written.is_empty() {
                            enq.enqueue(ingest::IngestItem {
                                ids: written,
                                col_mod: core.col_mod()?,
                                kind: ingest::MaintKind::Maintain,
                                membership_may_have_changed: false,
                            });
                        }
                    }
                    Ok(results)
                })
                .await??;
            Ok(results)
        }
        .instrument(span)
        .await
    }

    /// The collection's current `col.mod`.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection read fails.
    pub async fn col_mod(&self) -> NativeResult<i64> {
        self.collection.run(|core| core.col_mod()).await?
    }

    /// Await the ingest queue drained to the current point: every maintenance
    /// item enqueued before this call has been fully processed (re-read → embed
    /// → index/derived add → watermark advance). The deterministic barrier the
    /// data plane and tests await instead of polling — once it returns, the
    /// effects of all prior writes are visible. A no-op if the actor is gone.
    pub async fn settle(&self) {
        self.ingest.flush().await;
    }

    /// Resolve a note type's id by name.
    ///
    /// # Errors
    ///
    /// Returns an error if no note type has that name or the collection read
    /// fails.
    pub async fn notetype_id(&self, name: &str) -> NativeResult<i64> {
        let name = name.to_string();
        self.collection
            .run(move |core| core.notetype_id(&name))
            .await?
    }

    /// Create one note — sugar over [`upsert_notes`] with a batch of one (the
    /// batch op is the real implementation; per-item errors surface directly
    /// here since the batch is a single item).
    ///
    /// # Errors
    ///
    /// Returns an error if the create is rejected (duplicate/validation per the
    /// policy) or the collection write fails.
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

    /// Create a batch of notes (the duplicate policy applies per item) and
    /// index them — batch-shaped end to end: ONE collection job runs every
    /// create (per-item results, so one bad note never sinks the batch), ONE
    /// read job renders embed text + derived rows for everything created, ONE
    /// batched embed call produces all vectors, then one index add and a
    /// per-note derived ingest. Compute (embedding, index, derived) happens
    /// *off* the collection queue — it never routes back through it.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection write job fails (the outer `Result`);
    /// per-note rejections ride the inner per-item `Result`s. The
    /// index/derived maintenance tail is best-effort.
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
            // One serialized job for the whole batch of writes — and it reads
            // `col.mod` AND registers the watermark token in the SAME job,
            // so the actor's FIFO orders this op's registration with its write
            // ahead of any later op's job (closing the post-await-continuation
            // race), and the watermark it may advance to is the
            // one its OWN write produced.
            let enq = self.ingest.enqueuer();
            let outcomes: Vec<NativeResult<CreateOutcome>> = self
                .collection
                .run(move |core| -> NativeResult<_> {
                    let outcomes: Vec<NativeResult<CreateOutcome>> = notes
                        .iter()
                        .map(|n| {
                            core.create_note(n.notetype_id, n.deck_id, &n.fields, &n.tags, policy)
                        })
                        .collect();
                    // Enqueue the maintenance item INSIDE the write job so queue
                    // order == col.mod order (the load-bearing invariant); the
                    // slow embed runs off the drain, never blocking this write.
                    let created: Vec<i64> = outcomes
                        .iter()
                        .filter_map(|o| match o {
                            Ok(CreateOutcome::Created(id)) => Some(*id),
                            _ => None,
                        })
                        .collect();
                    if !created.is_empty() {
                        enq.enqueue(ingest::IngestItem {
                            ids: created,
                            col_mod: core.col_mod()?,
                            kind: ingest::MaintKind::Maintain,
                            membership_may_have_changed: false,
                        });
                    }
                    Ok(outcomes)
                })
                .await??;
            Ok(outcomes)
        }
        .instrument(span)
        .await
    }

    /// Drop already-deleted notes from the index + derived store (the prune
    /// path: the collection op removed them internally; this is the sidecar
    /// half of `delete_notes`). No own collection write precedes it, so it
    /// reads `col.mod` + registers the watermark token in one actor job.
    ///
    /// # Errors
    ///
    /// Returns an error if reading `col.mod` to register the watermark fails;
    /// the sidecar removals are best-effort (a failure leaves the watermark
    /// behind for the next drift to heal).
    pub async fn forget_notes(&self, note_ids: Vec<i64>) -> NativeResult<()> {
        if note_ids.is_empty() {
            return Ok(());
        }
        // No own write precedes this (the notes are already deleted): read
        // `col.mod` AND enqueue the remove in ONE collection job, so the enqueue
        // is FIFO-ordered (queue order == col.mod order).
        let enq = self.ingest.enqueuer();
        self.collection
            .run(move |core| -> NativeResult<()> {
                let col_mod = core.col_mod()?;
                enq.enqueue(ingest::IngestItem {
                    ids: note_ids,
                    col_mod,
                    kind: ingest::MaintKind::Remove,
                    membership_may_have_changed: false,
                });
                Ok(())
            })
            .await?
    }

    /// A metadata-only collection change (tags/decks/templates/field metadata
    /// — nothing that feeds embedding text or derived rows): advance the
    /// watermarks so the col_mod bump doesn't read as drift on next boot.
    /// There is no index/derived tail that could fail (the vectors/rows are
    /// unchanged), so both watermark sides complete as success — but still only
    /// advance past concurrent in-flight writes the tracker certifies.
    ///
    /// `membership_may_have_changed` is the tag-centroid relevance probe:
    /// only a TAG-membership change (a note gained/lost a tag, a tag was
    /// renamed/cleared) can move a centroid, so only those ops request the
    /// refresh. Deck rename/reparent, template/CSS find-replace, and field
    /// metadata bump col_mod but touch NO centroid input — requesting a refresh
    /// there is pure O(collection) waste (`note_tag_rows` + `note_count` +
    /// a whole-collection recompute) behind no relevance signal (per-op
    /// tails do no O(collection) work; the refresh runs only when relevant).
    ///
    /// # Errors
    ///
    /// Returns an error if reading `col.mod` to register the watermark fails.
    pub async fn metadata_changed(&self, membership_may_have_changed: bool) -> NativeResult<()> {
        // The metadata write already committed in its own prior job: read
        // `col.mod` AND enqueue an advance-only item in ONE collection job, so
        // the enqueue is FIFO-ordered with any concurrent upsert (queue order ==
        // col.mod order). The actor advances both watermarks (no index/derived
        // change to make) and requests the tag refresh when membership moved.
        let enq = self.ingest.enqueuer();
        self.collection
            .run(move |core| -> NativeResult<()> {
                let col_mod = core.col_mod()?;
                enq.enqueue(ingest::IngestItem {
                    ids: Vec::new(),
                    col_mod,
                    kind: ingest::MaintKind::AdvanceOnly,
                    membership_may_have_changed,
                });
                Ok(())
            })
            .await?
    }

    /// Delete notes and drop their sidecars in ONE maintained op: the
    /// existence partition (`deleted`/`not_found`), the anki delete, and the
    /// `col.mod`+watermark capture all run in the SAME collection write job,
    /// then the shared sidecar tail drops vectors/derived rows. This is the op
    /// the MCP `delete_notes` action rides — it replaces the host's old
    /// `wrapper.delete_notes` (its own `nid:` existence pre-check, a separate
    /// round trip) + separate `kernel.forget_notes` (the sidecar tail, two more
    /// round trips) with one write job + the tail. Returns `{deleted,
    /// not_found}`: a requested id that no longer exists is `not_found`, never
    /// silently counted as deleted.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection delete job fails; the sidecar tail is
    /// best-effort.
    pub async fn delete_notes(
        &self,
        note_ids: Vec<i64>,
    ) -> NativeResult<shrike_schemas::DeleteNotesResponse> {
        let ids = note_ids.clone();
        // Existence partition + delete + enqueue the remove in ONE job: the
        // `nid:` find scopes the existence check to the requested ids (no full
        // scan), and the enqueue inside the write job keeps queue order ==
        // col.mod order. The vector/derived-row removal runs off the drain.
        let enq = self.ingest.enqueuer();
        let deleted = self
            .collection
            .run(move |core| -> NativeResult<Vec<i64>> {
                let existing: std::collections::HashSet<i64> = if ids.is_empty() {
                    std::collections::HashSet::new()
                } else {
                    let nid = ids
                        .iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    core.find_notes(&format!("nid:{nid}"))?
                        .into_iter()
                        .collect()
                };
                let deleted: Vec<i64> = ids
                    .iter()
                    .copied()
                    .filter(|i| existing.contains(i))
                    .collect();
                if !deleted.is_empty() {
                    core.delete_notes(&deleted)?;
                    enq.enqueue(ingest::IngestItem {
                        ids: deleted.clone(),
                        col_mod: core.col_mod()?,
                        kind: ingest::MaintKind::Remove,
                        membership_may_have_changed: false,
                    });
                }
                Ok(deleted)
            })
            .await??;
        let not_found: Vec<i64> = note_ids
            .iter()
            .copied()
            .filter(|i| !deleted.contains(i))
            .collect();
        Ok(shrike_schemas::DeleteNotesResponse { deleted, not_found })
    }

    /// Import an `.apkg`/`.colpkg` package, then bring the index in line.
    ///
    /// Import is an OPAQUE bulk mutation: anki's importer adds/updates/remaps an
    /// unknown set of notes and bumps `col.mod`. So the tail is the boot/reload
    /// drift path, NOT a maintained per-note tail: run the import RPC on the
    /// collection actor, then drive `reindex_if_needed` — a whole-collection
    /// drift reconcile that fingerprint-diffs and re-embeds only the
    /// changed/new notes (dropping deleted), holistically correct across the
    /// import's notetype/deck remaps. Crucially this does NOT advance the index
    /// watermark before reconciling: the `col.mod` bump IS the drift signal, so
    /// `reindex_if_needed` sees it and reconciles. The derived-store rebuild is
    /// the harness's job (it owns that store), mirroring `reload`. Returns
    /// `(summary, reindexed)` — the per-bucket counts and whether the index
    /// reconciled (false only when no embedder is attached).
    ///
    /// # Errors
    ///
    /// Returns an error if the import RPC, the index reconcile, or the derived
    /// rebuild fails.
    pub async fn import_package(
        &self,
        package_path: String,
        options: shrike_collection::ImportOptions,
    ) -> NativeResult<(String, bool)> {
        let summary = self
            .collection
            .run(
                move |core| -> NativeResult<shrike_collection::ImportSummary> {
                    core.import_package(&package_path, options)
                },
            )
            .await??;
        // Import bumps col.mod, so BOTH derived caches drift — bring each in
        // line in one op, exactly as `harness.reload` does. The col.mod bump is
        // the drift signal for both; we never advance a watermark before
        // reconciling (that would suppress it).
        //
        // 1) Vector index: reconcile (fingerprint-diffed — only changed/new
        //    notes re-embed; no-op without an embedder attached).
        let reindexed = self.reindex_if_needed().await?;
        // 2) Derived-text (FTS5) store: rebuild on drift, so the imported notes
        //    are immediately findable by substring/fuzzy lexical search — not
        //    only after an unrelated boot/reload/re-acquire trigger. Conditioned
        //    on the store actually lagging the collection's col_mod (an import
        //    always moves it, but the check keeps this a no-op if it somehow
        //    didn't, and skips a build the store doesn't need).
        let col_mod = self.col_mod().await?;
        if self.derived.get_col_mod() != Some(col_mod) {
            self.rebuild_derived().await?;
        }
        Ok((summary.to_json().to_string(), reindexed))
    }

    // ── media + maintenance ops ──────────────────
    // The media tools and the prune as maintained kernel ops: the
    // host keeps only the tool signatures (and the serving-URL fill, which is
    // host config). Media never touches embedding text, so none of these do
    // index work — except prune, whose deletions carry their own sidecar
    // tail below.

    /// The full store_media batch: each item's byte source (base64 decode /
    /// SSRF-guarded URL download) prepares CONCURRENTLY — each item is a
    /// `tokio::spawn`'d task on the runtime. The URL fetch rides the async
    /// IP-pinned client (no blocking-pool thread parks on a network wait); the
    /// CPU base64 decode routes through the compute pool (`dispatch_compute`), so
    /// it lands off the IO driver. The batch then writes as ONE collection job
    /// (`path` items run their containment gates under that job; they carry no
    /// prepare work).
    ///
    /// # Errors
    ///
    /// Returns an error if a prepare task panics (a bug; expected source
    /// failures are per-item `Failed`) or the collection write job fails.
    pub async fn store_media(
        &self,
        items: Vec<shrike_schemas::StoreMediaItem>,
        allow_private_fetch: bool,
        path_roots: Vec<String>,
    ) -> NativeResult<Vec<shrike_schemas::StoreMediaResult>> {
        let handles: Vec<_> = items
            .into_iter()
            .enumerate()
            .map(|(i, item)| {
                runtime::handle().spawn(async move {
                    shrike_media::prepare_media_item_with_decode(
                        i as i64,
                        item,
                        allow_private_fetch,
                        |data| {
                            runtime::dispatch_compute(move || shrike_media::decode_media_b64(&data))
                        },
                    )
                    .await
                })
            })
            .collect();
        let mut prepared = Vec::with_capacity(handles.len());
        for handle in handles {
            // A JoinError is a PANIC in the prepare (every expected failure
            // is already a per-item Failed) — a bug fails the whole batch
            // deliberately, never a per-item error.
            prepared.push(
                handle
                    .await
                    .context(ErrorKind::Internal, "media prepare task")?,
            );
        }
        self.collection
            .run(move |core| core.store_prepared_media(&prepared, &path_roots))
            .await?
    }

    /// Resolve filenames to where their bytes live (never the bytes; the
    /// host fills the serving `url`).
    ///
    /// # Errors
    ///
    /// Returns an error if the collection read fails.
    pub async fn fetch_media(
        &self,
        filenames: Vec<String>,
    ) -> NativeResult<Vec<shrike_schemas::MediaFetchResult>> {
        self.collection
            .run(move |core| core.fetch_media(&filenames))
            .await?
    }

    /// List media files (sorted, optional glob + limit).
    ///
    /// # Errors
    ///
    /// Returns an error if the collection read fails.
    pub async fn list_media(
        &self,
        pattern: Option<String>,
        limit: Option<usize>,
    ) -> NativeResult<shrike_schemas::ListMediaResponse> {
        self.collection
            .run(move |core| core.list_media(pattern.as_deref(), limit))
            .await?
    }

    /// Move media files to Anki's recoverable trash.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection write fails.
    pub async fn delete_media(
        &self,
        filenames: Vec<String>,
    ) -> NativeResult<shrike_schemas::DeleteMediaResponse> {
        self.collection
            .run(move |core| core.delete_media(&filenames))
            .await?
    }

    /// Read-only media diagnostics.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection media check fails.
    pub async fn media_check(&self) -> NativeResult<shrike_schemas::CollectionCheckResponse> {
        self.collection.run(move |core| core.media_check()).await?
    }

    /// Export the collection (or a scope of it) to an Anki package.
    ///
    /// Read-only on the collection's data, but it holds the collection for the
    /// whole package write — so it rides the collection task-actor (serializing
    /// against other ops on this collection, exactly like a write; export is
    /// exclusive by nature). The host has already gated `out_path` (the
    /// path-safety check); the kernel trusts it and performs the anki export.
    /// `out_path` extension picks the format isn't done here — the host passes
    /// an explicit [`PackageFormat`] so the kernel never guesses.
    ///
    /// # Errors
    ///
    /// Returns an error if the anki export fails (e.g. the scope is invalid or
    /// the package cannot be written).
    pub async fn export_package(
        &self,
        out_path: String,
        format: shrike_collection::PackageFormat,
        scope: shrike_collection::ExportScope,
        with_scheduling: bool,
        with_media: bool,
        legacy: bool,
    ) -> NativeResult<shrike_schemas::ExportPackageResult> {
        let req = shrike_collection::ExportRequest {
            out_path,
            format,
            scope,
            with_scheduling,
            with_media,
            legacy,
        };
        let outcome = self
            .collection
            .run(move |core| core.export_package(&req))
            .await??;
        Ok(shrike_schemas::ExportPackageResult {
            note_count: outcome.note_count,
            out_path: outcome.out_path,
        })
    }

    /// The prune with its maintenance tail: deletions drop their sidecars
    /// like `delete_notes`;
    /// a tags-only prune is a metadata-only watermark advance. The tail is
    /// best-effort — a failure logs and the response still returns (the next
    /// boot's drift check repairs).
    ///
    /// # Errors
    ///
    /// Returns an error if the prune collection job fails; the maintenance tail
    /// is best-effort.
    pub async fn collection_prune(
        &self,
        unused_tags: bool,
        empty_notes: bool,
        empty_cards: bool,
        unused_media: bool,
        dry_run: bool,
    ) -> NativeResult<shrike_schemas::CollectionPruneResponse> {
        let (response, removed_note_ids) = self
            .collection
            .run(move |core| {
                core.prune(unused_tags, empty_notes, empty_cards, unused_media, dry_run)
            })
            .await??;
        if !dry_run {
            let tags_removed = response.unused_tags.as_ref().is_some_and(|t| t.removed > 0);
            let tail = if !removed_note_ids.is_empty() {
                self.forget_notes(removed_note_ids).await
            } else if tags_removed {
                // clear_unused_tags removes only tags on NO notes → no note's
                // membership moves and no centroid (min_members > 0) changes →
                // watermark bump only, no refresh.
                self.metadata_changed(false).await
            } else {
                Ok(())
            };
            if let Err(e) = tail {
                tracing::warn!(error = %e, "index maintenance after prune failed");
            }
        }
        Ok(response)
    }

    // ── tag + deck ops ────────────────────
    // Tags and deck names are not embedding text: each op is one collection
    // job plus the best-effort metadata watermark tail when anything changed
    // (mirroring the host's _bump_col_mod_after_metadata_change, whose
    // remaining callers retire with the note-type re-home).

    /// The shared metadata tail: advance the watermarks so a vectors-
    /// unchanged col_mod bump doesn't read as drift on next boot. Best-effort
    /// — cache bookkeeping never fails an op already committed.
    ///
    /// `membership_may_have_changed` gates the tag-centroid refresh: a
    /// deck/template/field-metadata edit passes `false` (no centroid input
    /// moved); a tag-membership edit passes `true`.
    async fn metadata_tail(&self, changed: bool, membership_may_have_changed: bool) {
        if !changed {
            return;
        }
        if let Err(e) = self.metadata_changed(membership_may_have_changed).await {
            tracing::warn!(error = %e, "watermark bump after metadata change failed");
        }
    }

    /// Edit tags on a note set (`set` full-replace XOR add/remove — the
    /// exclusivity is the host's input validation).
    ///
    /// # Errors
    ///
    /// Returns an error if the collection write fails; the metadata tail is
    /// best-effort.
    pub async fn update_note_tags(
        &self,
        note_ids: Vec<i64>,
        set_tags: Option<Vec<String>>,
        add: Vec<String>,
        remove: Vec<String>,
    ) -> NativeResult<shrike_schemas::UpdateNoteTagsResponse> {
        let response = self
            .collection
            .run(move |core| core.update_note_tags(&note_ids, set_tags.as_deref(), &add, &remove))
            .await??;
        // Tag membership changed → refresh centroids.
        self.metadata_tail(response.notes_modified > 0, true).await;
        Ok(response)
    }

    /// Rename a tag collection-wide (empty `note_ids`) or exactly on a set.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection write fails; the metadata tail is
    /// best-effort.
    pub async fn rename_tag(
        &self,
        old: String,
        new: String,
        note_ids: Vec<i64>,
    ) -> NativeResult<shrike_schemas::RenameTagResponse> {
        let response = self
            .collection
            .run(move |core| core.rename_tag(&old, &new, &note_ids))
            .await??;
        // A rename remaps a centroid's tag key → refresh centroids.
        self.metadata_tail(response.notes_modified > 0, true).await;
        Ok(response)
    }

    /// Create or rename decks in bulk (id present = rename; never merges).
    ///
    /// # Errors
    ///
    /// Returns an error if the collection write fails; the metadata tail is
    /// best-effort.
    pub async fn upsert_decks(
        &self,
        decks: Vec<shrike_schemas::DeckInput>,
    ) -> NativeResult<Vec<shrike_schemas::UpsertDeckResult>> {
        let results = self
            .collection
            .run(move |core| core.upsert_decks(&decks))
            .await??;
        let changed = results.iter().any(|r| {
            matches!(
                r,
                shrike_schemas::UpsertDeckResult::Created { .. }
                    | shrike_schemas::UpsertDeckResult::Updated { .. }
            )
        });
        // Deck names are not centroid inputs → watermark bump only, no refresh.
        self.metadata_tail(changed, false).await;
        Ok(results)
    }

    /// Delete decks by reference — only if empty.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection write fails; the metadata tail is
    /// best-effort.
    pub async fn delete_decks(
        &self,
        refs: Vec<String>,
    ) -> NativeResult<shrike_schemas::DeleteDecksResponse> {
        let response = self
            .collection
            .run(move |core| core.delete_decks(&refs))
            .await??;
        // delete_decks is empty-only (never deletes a note) → no membership
        // change; watermark bump only, no centroid refresh.
        self.metadata_tail(!response.deleted.is_empty(), false)
            .await;
        Ok(response)
    }

    // ── note-type ops ─────────────────────
    // The note-type tools as maintained kernel ops. The
    // structural edits (upsert/fields/templates/delete) carry NO tail: a
    // field-list change can alter embedding text, so their col.mod bump
    // deliberately reads as drift on the next boot (a removed field makes a
    // rebuild correct; for the rest it's conservative — the pre-re-home
    // contract). Template/CSS text and field metadata never feed embedding
    // text, so those two advance the watermarks; migration moves note fields,
    // so an apply re-embeds the changed notes.

    /// Create/update note-type definitions in bulk (the position-keyed
    /// replace with the unsound-move rejection), per-item results.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection write job fails.
    pub async fn upsert_note_types(
        &self,
        note_types: Vec<shrike_schemas::NoteTypeInput>,
    ) -> NativeResult<Vec<shrike_schemas::NoteTypeResult>> {
        self.collection
            .run(move |core| core.upsert_note_types(&note_types))
            .await?
    }

    /// Identity-based field ops (add/remove/rename/reposition), atomic.
    ///
    /// # Errors
    ///
    /// Returns an error if an op is unsound (bad name/position) or the
    /// collection write fails.
    pub async fn update_note_type_fields(
        &self,
        note_type_name: String,
        operations: Vec<shrike_schemas::FieldOp>,
    ) -> NativeResult<shrike_schemas::UpdateNoteTypeFieldsResponse> {
        self.collection
            .run(move |core| core.update_note_type_fields(&note_type_name, &operations))
            .await?
    }

    /// Identity-based template ops (add/remove/rename/reposition), atomic.
    ///
    /// # Errors
    ///
    /// Returns an error if an op is unsound (bad name/position) or the
    /// collection write fails.
    pub async fn update_note_type_templates(
        &self,
        note_type_name: String,
        operations: Vec<shrike_schemas::TemplateOp>,
    ) -> NativeResult<shrike_schemas::UpdateNoteTypeTemplatesResponse> {
        self.collection
            .run(move |core| core.update_note_type_templates(&note_type_name, &operations))
            .await?
    }

    /// Literal-or-regex rewrite over one model's template HTML + CSS, with
    /// its watermark tail: template/CSS text isn't embedding text, so a real
    /// replace advances the watermarks instead of reading as drift (a no-op
    /// replace saves nothing and bumps nothing). Best-effort — a tail
    /// failure logs and the response still returns.
    ///
    /// # Errors
    ///
    /// Returns an error if the rewrite collection job fails (e.g. an invalid
    /// regex); the watermark tail is best-effort.
    #[allow(clippy::too_many_arguments)]
    pub async fn find_replace_note_types(
        &self,
        note_type_name: String,
        search: String,
        replacement: String,
        regex: bool,
        match_case: bool,
        front: bool,
        back: bool,
        css: bool,
    ) -> NativeResult<shrike_schemas::FindReplaceNoteTypesResponse> {
        let response = self
            .collection
            .run(move |core| {
                core.find_and_replace_note_types(
                    &note_type_name,
                    &search,
                    &replacement,
                    regex,
                    match_case,
                    front,
                    back,
                    css,
                )
            })
            .await??;
        if response.replacements > 0 {
            // Template/CSS text is not a centroid input → no refresh.
            if let Err(e) = self.metadata_changed(false).await {
                tracing::warn!(error = %e, "watermark advance after find_replace_note_types failed");
            }
        }
        Ok(response)
    }

    /// Per-field editor metadata (font/size/description), with the
    /// unconditional watermark tail: editor cosmetics never touch embedding
    /// text, but the persist bumps col.mod. Best-effort, like the tag/deck
    /// metadata ops.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection write fails; the watermark tail is
    /// best-effort.
    pub async fn update_note_type_field_metadata(
        &self,
        note_type_name: String,
        updates: Vec<shrike_schemas::FieldMetadataInput>,
    ) -> NativeResult<shrike_schemas::UpdateNoteTypeFieldMetadataResponse> {
        let response = self
            .collection
            .run(move |core| core.update_note_type_field_metadata(&note_type_name, &updates))
            .await??;
        // Field editor metadata (font/size/description) is not a centroid input
        // → watermark bump only, no refresh.
        if let Err(e) = self.metadata_changed(false).await {
            tracing::warn!(error = %e, "watermark advance after field-metadata update failed");
        }
        Ok(response)
    }

    /// Change notes' note type via name maps. Migration moves field
    /// content — embedding text — under unchanged ids, so an apply re-embeds
    /// and re-ingests the changed notes (best-effort: the migration is
    /// committed either way; the next boot's drift check repairs). Dry-run
    /// touches nothing.
    ///
    /// # Errors
    ///
    /// Returns an error if the migration collection job fails (e.g. an unknown
    /// name or an invalid field map); the re-embed tail is best-effort.
    pub async fn migrate_note_type(
        &self,
        note_ids: Vec<i64>,
        new_note_type: String,
        field_map: std::collections::BTreeMap<String, String>,
        template_map: std::collections::BTreeMap<String, String>,
        dry_run: bool,
    ) -> NativeResult<shrike_schemas::MigrateNoteTypeResponse> {
        let response = self
            .collection
            .run(move |core| {
                core.migrate_note_type(
                    &note_ids,
                    &new_note_type,
                    &field_map,
                    &template_map,
                    dry_run,
                )
            })
            .await??;
        if !dry_run && !response.changed.is_empty() {
            if let Err(e) = self.reindex_notes(&response.changed).await {
                tracing::warn!(error = %e, "index maintenance after migrate_note_type failed");
            }
        }
        Ok(response)
    }

    /// Delete note types by id, only-if-unused, per-item results.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection write job fails.
    pub async fn delete_note_types(
        &self,
        ids: Vec<i64>,
    ) -> NativeResult<Vec<shrike_schemas::DeleteNoteTypeResult>> {
        self.collection
            .run(move |core| core.delete_note_types(&ids))
            .await?
    }

    /// Fused search with the kernel's default arguments — a thin delegate to
    /// [`actions::search_notes`], the ONE fused-search spine: no
    /// scope, no threshold, no image floor, the canonical `fusion` weights.
    /// The query embeds here when an embedder is attached; otherwise the
    /// action degrades to the lexical signals. `score` carries the wire
    /// contract's semantic similarity when present (the raw RRF magnitude is
    /// deliberately not exposed, matching the host surface).
    ///
    /// # Errors
    ///
    /// Returns an error if query embedding (when an embedder is attached) or any
    /// index/derived/collection read in the fusion fails.
    pub async fn search(&self, query: &str, top_k: usize) -> NativeResult<Vec<KernelHit>> {
        let span = tracing::debug_span!("kernel.search", top_k);
        async move {
            // The embed future is built (its compute scheduled by the eager
            // adapter) before the action's lexical reads run — the same
            // overlap the host path gets by embedding before dispatch.
            let svc = self.embed_service();
            let (vectors, semantic) = match &svc {
                Some(svc) => (svc.embedder.embed(vec![query.to_string()]).await?, true),
                None => (Vec::new(), false),
            };
            let sources = vec![actions::SearchSource {
                label: "query".to_string(),
                text: query.to_string(),
                is_query: true,
            }];
            // The PRIMARY space's engine carries the host-supplied query
            // vectors + the lexical/tag signals; cross-space fusion adds
            // each SECONDARY text-capable space's gated image ranking. With one
            // space `cross_space` is empty → byte-identical to the single-space
            // path. `index_size` sums across every space.
            let primary = self.index_set.primary();
            let cross_space = if semantic {
                self.build_cross_space(&[query.to_string()], top_k).await?
            } else {
                Vec::new()
            };
            let index_size: usize = self
                .index_set
                .all_orchestrators()
                .iter()
                .map(|o| o.engine().size())
                .sum();
            let args = actions::SearchArgs {
                top_k,
                threshold: 0.0,
                weights: BTreeMap::new(), // empty = the canonical fusion set
                semantic,
                index_size,
                hidden_lexical_sources: Self::hidden_lexical_sources()
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
                cross_space,
                ..Default::default()
            };
            let engine = primary.engine_arc();
            let derived = Arc::clone(&self.derived);
            let tag_keys = Arc::clone(&self.tag_keys);
            let groups = self
                .collection
                .run(move |core| {
                    actions::search_notes(
                        core,
                        Some(&*engine),
                        Some(&*derived),
                        Some(&tag_keys),
                        &sources,
                        &vectors,
                        &args,
                    )
                })
                .await??;
            Ok(groups
                .into_iter()
                .next()
                .map(|g| g.matches)
                .unwrap_or_default()
                .into_iter()
                .map(|m| KernelHit {
                    note_id: m.note.id,
                    score: m.score.unwrap_or(0.0),
                    signals: m
                        .provenance
                        .into_iter()
                        .map(|c| (c.signal, c.rank))
                        .collect(),
                })
                .collect())
        }
        .instrument(span)
        .await
    }

    /// Cooperative idle-release: close the collection, keeping the
    /// kernel reusable via [`reopen`]. WHEN to release is harness policy (an
    /// idle timer on its runtime); the kernel only provides the ops.
    ///
    /// # Errors
    ///
    /// Returns an error if releasing the collection fails (or the actor is
    /// gone).
    pub async fn release(&self) -> NativeResult<()> {
        self.collection.run(|core| core.release()).await?
    }

    /// Re-acquire after a release; contention surfaces as the BUSY error
    /// tier (retryable — the caller decides, nothing waits).
    ///
    /// # Errors
    ///
    /// Returns the BUSY error tier if another process holds the collection, or
    /// an error if the re-acquire otherwise fails.
    pub async fn reopen(&self) -> NativeResult<()> {
        self.collection.run(|core| core.reopen()).await?
    }

    /// Close the collection and drain the actor — close returns with nothing
    /// in flight (the interpreter-teardown guard). Works
    /// through `&self` so shared handles (the binding's `Arc<Kernel>`) can
    /// close; idempotent (`SerializedCollection::close` treats a drained
    /// actor as already-closed).
    ///
    /// # Errors
    ///
    /// Returns an error if closing the collection fails; a drained actor is the
    /// already-closed success, not an error.
    pub async fn close(&self) -> NativeResult<()> {
        // Shutdown drain ordering (the spec's fixed order): the ingest actor
        // drains its queue and flushes the index savers durably FIRST — while
        // the collection is still open, since a drain re-reads note content —
        // so the durable watermark never lands ahead of a never-flushed index
        // write. Only then do the maintenance coordinators and the collection
        // close.
        self.ingest.shutdown().await;
        // A sleeping coalesced tag refresh has nothing to read once the
        // collection actor drains — abort it next.
        self.tag_refresh.shutdown();
        let result = self.collection.close().await;
        self.collection.shutdown().await;
        result
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Shared collision-proof, self-cleaning scratch dirs for the kernel's
    //! ~143-test shared binary.
    //!
    //! The original `{env::temp_dir()}/...-{pid}-{seq}` helpers had a
    //! deterministic test-isolation bug (reproducible at N=1, no concurrency
    //! needed): macOS recycles pids and the per-process seq counter restarts at
    //! 0, so a later process regenerates an EARLIER run's path. They also never
    //! deleted the dir, and `CollectionCore::open` is open-or-create — so the new
    //! process reopens the leftover `c.anki2` and `count(*)` returns the old rows
    //! plus its own (the #880 count-doubling cluster).
    //!
    //! Two properties fix it:
    //!
    //! - **Random leaf, never the pid.** The per-test dir name is a random
    //!   128-bit token (see [`random_token`]), so it can't collide with a
    //!   lingering dir from any other process — the bug is structurally gone, not
    //!   just made rarer. `$TEST_TMPDIR`/`$TMPDIR` is only the BASE dir.
    //! - **Cleanup.** [`ScratchDir`] removes its tree on drop, so even a panicking
    //!   test leaves nothing behind.

    use std::path::{Path, PathBuf};

    /// The base dir for scratch dirs: Bazel's `$TEST_TMPDIR` when present, else
    /// the shared `$TMPDIR` for a bare `cargo test`. This is only the BASE —
    /// uniqueness comes from the random leaf below, never from the base, so a
    /// shared `$TMPDIR` across many `cargo test` processes is safe.
    fn scratch_root() -> PathBuf {
        std::env::var_os("TEST_TMPDIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
    }

    /// A RANDOM, unique scratch path WITHOUT a cleanup guard — for the few
    /// helpers that hand the dir to a long-lived owner (an orchestrator) and so
    /// can't tie its lifetime to a local [`ScratchDir`].
    ///
    /// The leaf is a random 128-bit token, deliberately NOT derived from
    /// `std::process::id()`: pids RECYCLE across processes and the per-process
    /// counter restarts from 0, so a `{pid}-{seq}` leaf regenerates an IDENTICAL
    /// path in a later process — and because `CollectionCore::open` is
    /// open-or-create, that path reopens the earlier run's leftover `c.anki2` and
    /// `count(*)` returns its rows plus the new ones (the exact #880 count
    /// doubling, deterministic at N=1). A random leaf can't collide even when a
    /// prior run's dir lingers.
    pub(crate) fn collision_proof_dir(prefix: &str) -> PathBuf {
        scratch_root().join(format!("{prefix}-{}", random_token()))
    }

    /// A 128-bit random hex token, unique per call. Seeds from the OS via
    /// `RandomState` (a fresh non-deterministic hasher per call) and mixes in a
    /// monotonic counter + a nanosecond stamp, so two calls in the same process
    /// — and the same call across processes — never coincide. No `process::id()`
    /// anywhere (its recycling is the whole bug). Std-only, so no new crate edge
    /// for the Bazel crate-universe to splice.
    fn random_token() -> String {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let mix = |salt: u64| -> u64 {
            let mut h = RandomState::new().build_hasher();
            h.write_u64(salt);
            h.write_u64(SEQ.fetch_add(1, Ordering::Relaxed));
            h.write_u64(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0),
            );
            h.finish()
        };
        format!(
            "{:016x}{:016x}",
            mix(0xA5A5_A5A5_A5A5_A5A5),
            mix(0x5A5A_5A5A_5A5A_5A5A)
        )
    }

    /// A scratch directory that deletes itself (and everything under it) on drop.
    /// Deref-es to its [`Path`], so it stands in for a `PathBuf` at call sites.
    pub(crate) struct ScratchDir {
        path: PathBuf,
    }

    impl ScratchDir {
        /// Create a fresh scratch dir with the given prefix, unique across
        /// processes and runs. The prefix only aids debugging — uniqueness comes
        /// from the random leaf (see [`collision_proof_dir`]), never the pid.
        pub(crate) fn new(prefix: &str) -> Self {
            let dir = collision_proof_dir(prefix);
            std::fs::create_dir_all(&dir).unwrap();
            Self { path: dir }
        }

        /// A child path under the scratch dir (e.g. `c.anki2`, `cache`).
        pub(crate) fn join(&self, child: impl AsRef<Path>) -> PathBuf {
            self.path.join(child)
        }
    }

    impl std::ops::Deref for ScratchDir {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.path
        }
    }

    impl AsRef<Path> for ScratchDir {
        fn as_ref(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.path).ok();
        }
    }

    #[cfg(test)]
    mod acceptance {
        //! Deterministic plant-toggle proofs for the #880 count-doubling mode —
        //! no concurrency or oversubscription needed.

        use super::*;
        use shrike_collection::{Collection, CollectionCore, DuplicatePolicy};

        fn add_one(core: &dyn Collection, front: &str) {
            let req = serde_json::json!([
                {"note_type": "Basic", "deck": "D", "fields": {"Front": front, "Back": "b"}}
            ]);
            let notes: Vec<shrike_schemas::NoteInput> = serde_json::from_value(req).unwrap();
            core.upsert_notes(&notes, DuplicatePolicy::Allow, false)
                .unwrap();
        }

        fn note_count(core: &CollectionCore) -> i64 {
            core.query("", false, 1000).unwrap().notes.len() as i64
        }

        /// The BUG, pinned: reopening the SAME path over a populated, undeleted
        /// `c.anki2` reads the leftover rows PLUS the new one — open-or-create
        /// preserves rows. This is what a recycled `{pid}-{seq}` path did across
        /// processes; it must stay reproducible so the fix's value is provable.
        #[test]
        fn reopening_a_leftover_collection_pollutes_the_count() {
            let dir = ScratchDir::new("plant-leftover");
            let path = dir.join("c.anki2");
            let path_str = path.to_str().unwrap();

            // Run 1: leave 2 notes behind WITHOUT removing the dir.
            let first = CollectionCore::open(path_str).unwrap();
            add_one(&first, "one");
            add_one(&first, "two");
            assert_eq!(note_count(&first), 2);
            first.close().unwrap();

            // Run 2 reusing the path: a fresh test that adds ONE note now sees 3
            // — the leftover pollution the random-leaf fix prevents.
            let second = CollectionCore::open(path_str).unwrap();
            assert_eq!(
                note_count(&second),
                2,
                "open-or-create reopened the leftover"
            );
            add_one(&second, "three");
            assert_eq!(
                note_count(&second),
                3,
                "leftover (2) + own (1) = the count-doubling symptom"
            );
            second.close().unwrap();
        }

        /// The FIX: independent helper calls never share a path, so one test's
        /// leftover can never be another's slot — even if a prior dir lingers.
        #[test]
        fn independent_scratch_dirs_never_collide() {
            // Many calls, all distinct — the random leaf, not a recycled pid+seq.
            let n = 200;
            let dirs: Vec<PathBuf> = (0..n)
                .map(|_| collision_proof_dir("plant-unique"))
                .collect();
            let unique: std::collections::BTreeSet<&PathBuf> = dirs.iter().collect();
            assert_eq!(unique.len(), n, "every scratch path is unique");

            // A populated leftover at one path does NOT pollute a fresh one.
            let leftover = ScratchDir::new("plant-toggle");
            let lpath = leftover.join("c.anki2");
            let c = CollectionCore::open(lpath.to_str().unwrap()).unwrap();
            add_one(&c, "stale");
            c.close().unwrap();

            let fresh = ScratchDir::new("plant-toggle");
            assert_ne!(fresh.path, leftover.path, "fresh dir is a different path");
            let fc = CollectionCore::open(fresh.join("c.anki2").to_str().unwrap()).unwrap();
            assert_eq!(
                note_count(&fc),
                0,
                "fresh collection is empty — no pollution"
            );
            fc.close().unwrap();
        }

        /// The structural guarantee: the leaf is `{prefix}-{32 hex}`, i.e. a pure
        /// random token, NOT the `{prefix}-{pid}-{seq}` shape whose pid recycling
        /// was the root cause. A regression that reintroduces a pid/seq leaf
        /// changes this shape and fails here regardless of scheduling.
        #[test]
        fn scratch_leaf_is_a_random_hex_token_not_pid_seq() {
            for _ in 0..50 {
                let p = collision_proof_dir("plant-shape");
                let leaf = p.file_name().unwrap().to_string_lossy();
                let token = leaf
                    .strip_prefix("plant-shape-")
                    .unwrap_or_else(|| panic!("unexpected leaf shape {leaf}"));
                assert_eq!(token.len(), 32, "token is a 128-bit hex string: {leaf}");
                assert!(
                    token.bytes().all(|b| b.is_ascii_hexdigit()),
                    "token is pure hex (no pid/seq decimals): {leaf}"
                );
            }
        }
    }
}

#[cfg(test)]
mod no_cpython_smoke {
    //! The acceptance smoke: link the kernel WITHOUT Python and run one
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

    /// Test shim: the wire-shaped upsert with the pre-typed-seam call
    /// shape — JSON in, serialized results out.
    async fn upsert_wire(
        kernel: &Kernel,
        notes_json: String,
        on_duplicate: String,
        dry_run: bool,
    ) -> String {
        let notes: Vec<shrike_schemas::NoteInput> = serde_json::from_str(&notes_json).unwrap();
        let policy = DuplicatePolicy::parse(&on_duplicate).unwrap();
        let results = kernel
            .upsert_notes_wire(notes, policy, dry_run)
            .await
            .unwrap();
        serde_json::to_string(&results).unwrap()
    }

    fn temp_dir() -> crate::test_support::ScratchDir {
        crate::test_support::ScratchDir::new("shrike-kernel")
    }

    /// Await the op-tail (coalesced background) tag refresh reaching the expected
    /// tag-state shape. Unbounded — no iteration cap to race a starved scheduler:
    /// a refresh that never reaches the shape hangs and Bazel's per-test timeout
    /// catches it; a slow one just takes another poll. The poll exists only
    /// because the background refresh exposes no completion channel to await.
    async fn wait_for_tags(kernel: &Kernel, pred: impl Fn(&tag_centroids::TagKeyMap) -> bool) {
        while !pred(kernel.tag_keys()) {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }

    /// Compile-time pin: every kernel future is Send, so a harness may spawn
    /// kernel ops on any multithreaded runtime (a !Send
    /// regression, e.g. an entered span guard held across an await, fails
    /// here instead of downstream).
    /// Dropping the edge wrapper DETACHES observation — the
    /// spawned op runs to completion (never an abort; a half-applied
    /// collection write would be corruption). The regression guard against
    /// a JoinHandle-shaped wrapper.
    #[test]
    fn dropping_the_op_wrapper_never_aborts_the_work() {
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let collection = SerializedCollection::open(
                dir.join("collection.anki2").to_string_lossy().into_owned(),
            )
            .await
            .unwrap();
            let collection = Arc::new(collection);

            // A slow write job, spawned through the edge and DROPPED.
            let col = Arc::clone(&collection);
            let (started_tx, started_rx) = oneshot::channel::<()>();
            let wrapper = crate::spawn_op(async move {
                col.run(move |core| {
                    let _ = started_tx.send(());
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    core.col_mod()
                })
                .await?
            });
            drop(wrapper); // detach, not abort
            started_rx.await.expect("the job still started");

            // The actor stays healthy and serialized: the NEXT job runs
            // after the detached one completed (FIFO), proving it wasn't
            // torn down mid-flight.
            let _stamp = collection.run(|core| core.col_mod()).await.unwrap();
            collection.shutdown().await;
        });
    }

    /// The actor serializes FIFO: jobs sent in order run in order, even with
    /// every completion awaited concurrently.
    #[test]
    fn collection_jobs_run_fifo() {
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let collection = SerializedCollection::open(
                dir.join("collection.anki2").to_string_lossy().into_owned(),
            )
            .await
            .unwrap();
            let log = Arc::new(Mutex::new(Vec::new()));
            let futures: Vec<_> = (0..16)
                .map(|i| {
                    let log = Arc::clone(&log);
                    collection.run(move |_core| log.lock().unwrap().push(i))
                })
                .collect();
            futures::future::join_all(futures).await;
            assert_eq!(*log.lock().unwrap(), (0..16).collect::<Vec<_>>());
            collection.shutdown().await;
        });
    }

    fn assert_send<F: std::future::Future + Send>(f: F) -> F {
        f
    }

    #[test]
    fn detach_degrades_and_reattach_recovers() {
        // The embed slot is runtime-swappable: detached, the kernel
        // still creates notes and serves lexical search; re-attached, the
        // stale index watermark makes reindex catch up on what it missed.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
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
            // The detached upsert's derived/FTS write rides the async ingest drain, so
            // settle before searching (eventual consistency — a bare search races it).
            kernel.settle().await;
            let hits = kernel.search("capital of france", 5).await.unwrap();
            assert_eq!(hits[0].note_id, nid);
            assert!(hits[0].signals.iter().all(|(s, _)| s != "text"));

            // Re-attach: the index watermark stayed put, so reindex sees
            // drift and embeds the note created while detached.
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            assert!(kernel.reindex_if_needed().await.unwrap());
            kernel.settle().await;
            let hits = kernel.search("capital of france", 5).await.unwrap();
            assert!(hits[0].signals.iter().any(|(s, _)| s == "text"));

            kernel.close().await.unwrap();
            // Idempotent: a second close after the actor drained is
            // already-closed, not an actor-gone error.
            kernel.close().await.unwrap();
        });
    }

    /// A second deterministic embedder with a DISTINCT fingerprint, so the
    /// embed set keys it as a separate space from `HashEmbedder`.
    struct HashEmbedder2;

    impl Embedder for HashEmbedder2 {
        fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            Box::pin(async move { Ok(HashEmbedder::embed_sync(&texts)) })
        }

        fn fingerprint(&self) -> Option<String> {
            Some("hash-embedder:v2".to_string())
        }

        fn dim(&self) -> Option<usize> {
            Some(64)
        }
    }

    #[test]
    fn embed_set_holds_ordered_spaces_primary_is_first_text_space() {
        // The embed slot is an ordered SET keyed by CONTENT fingerprint.
        // Attaching two distinct fingerprints yields a 2-element embed_spaces()
        // while embed_service() still returns the PRIMARY (first text) space —
        // the index/search paths consume exactly one engine, so N=1
        // stays byte-identical.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();

            // N=1: the sole space is the primary; one element.
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            assert_eq!(kernel.embed_space_count(), 1);
            assert_eq!(kernel.embed_spaces().len(), 1);
            assert_eq!(
                kernel
                    .embed_service()
                    .unwrap()
                    .embedder
                    .fingerprint()
                    .as_deref(),
                Some("hash-embedder:v1"),
                "the sole space is the primary"
            );

            // N=2: a distinct fingerprint is a SECOND space; the primary stays
            // the first text space.
            kernel.attach_embedder(Arc::new(HashEmbedder2), None);
            assert_eq!(kernel.embed_space_count(), 2);
            assert_eq!(kernel.embed_spaces().len(), 2);
            assert_eq!(
                kernel
                    .embed_service()
                    .unwrap()
                    .embedder
                    .fingerprint()
                    .as_deref(),
                Some("hash-embedder:v1"),
                "embed_service() still returns the PRIMARY (first) text space"
            );

            // Re-attaching the SAME fingerprint REPLACES in place (no growth) —
            // a model swap that keeps its identity, not a duplicate space.
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            assert_eq!(
                kernel.embed_space_count(),
                2,
                "same fingerprint replaces its space; the count is unchanged"
            );

            // By-key detach drops one space; the other (now the only, hence
            // primary) survives.
            assert!(kernel.detach_embedder_space("hash-embedder:v1"));
            assert_eq!(kernel.embed_space_count(), 1);
            assert_eq!(
                kernel
                    .embed_service()
                    .unwrap()
                    .embedder
                    .fingerprint()
                    .as_deref(),
                Some("hash-embedder:v2"),
                "the surviving space becomes the primary"
            );

            // The N=1 whole-clear detach empties the set.
            kernel.detach_embedder();
            assert_eq!(kernel.embed_space_count(), 0);
            assert!(kernel.embed_service().is_none());

            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn index_set_binds_spaces_in_lockstep_primary_in_place() {
        // Attaching two distinct-fingerprint embedders materializes TWO
        // index spaces in lockstep with the embed set. The PRIMARY index space
        // lives at the base index dir DIRECTLY (the in-place-no-subdir migration
        // rule → zero rebuild for existing users); the secondary gets a subdir.
        // The index/search path consumes the primary.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let cache = dir.join("cache");
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                cache.to_str().unwrap(),
            )
            .await
            .unwrap();

            // Before any embedder: the index set holds exactly the PRIMARY,
            // un-bound, at the base dir (byte-identical to the single space).
            assert_eq!(kernel.index_set().len(), 1);
            let primary_dir = kernel.index().dir.clone();
            // The base index dir is the per-collection namespaced dir, NOT a
            // space subdir of it.
            assert!(primary_dir.exists());

            // First embedder claims the primary in place — no new space, no
            // subdir (the migration rule).
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            assert_eq!(kernel.index_set().len(), 1, "first attach claims primary");
            assert_eq!(
                kernel.index().dir,
                primary_dir,
                "primary still at the base dir (no subdir → zero rebuild)"
            );

            // Second distinct embedder → a SECOND index space in a subdir.
            kernel.attach_embedder(Arc::new(HashEmbedder2), None);
            assert_eq!(kernel.index_set().len(), 2, "second attach is a new space");
            let secondary = kernel
                .index_set()
                .orchestrator_for("hash-embedder:v2")
                .expect("the secondary space is bound by its fingerprint");
            assert_ne!(secondary.dir, primary_dir, "secondary is NOT the base dir");
            assert!(
                secondary.dir.starts_with(&primary_dir),
                "secondary lives in a subdir UNDER the base dir"
            );
            // The primary stays the served index (the v1 space).
            assert_eq!(kernel.index().dir, primary_dir);

            // The end-to-end index path still works (primary-only): a
            // note created with both spaces attached embeds + is searchable.
            kernel.reindex_if_needed().await.unwrap();
            let basic = kernel.notetype_id("Basic").await.unwrap();
            let CreateOutcome::Created(nid) = kernel
                .upsert_note(
                    basic,
                    1,
                    vec!["paris is the capital of france".into(), "geo".into()],
                    vec![],
                    DuplicatePolicy::Error,
                )
                .await
                .unwrap()
            else {
                panic!("create")
            };
            // The upsert's index/derived write rides the async drain — settle
            // before the search reads it back.
            kernel.settle().await;
            let hits = kernel.search("capital of france", 5).await.unwrap();
            assert_eq!(hits[0].note_id, nid);
            assert!(
                hits[0].signals.iter().any(|(s, _)| s == "text"),
                "the primary text space's semantic signal contributes"
            );
            // The primary index holds the note.
            assert!(kernel.index().engine().contains(nid));

            // Removal fans out to EVERY space — no error on the empty
            // secondary, and the primary drops the note. The vector removal
            // rides the drain too, so settle before asserting it's gone.
            kernel.delete_notes(vec![nid]).await.unwrap();
            kernel.settle().await;
            assert!(!kernel.index().engine().contains(nid));
            assert!(
                !secondary.engine().contains(nid),
                "removal fanned out to the secondary too"
            );

            kernel.close().await.unwrap();
        });
    }

    // ── Per-space per-modality write fan-out + hashing ────────────────

    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A text-primary embedder with an embed-call counter (text re-embeds).
    struct CountingText {
        calls: AtomicUsize,
        fp: &'static str,
    }
    impl Embedder for CountingText {
        fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { Ok(HashEmbedder::embed_sync(&texts)) })
        }
        fn fingerprint(&self) -> Option<String> {
            Some(self.fp.to_string())
        }
        fn dim(&self) -> Option<usize> {
            Some(64)
        }
    }

    /// An image-primary CLIP stand-in: a text tower (used for the QUERY,
    /// and as the orchestrator's text embedder — but ImageOnly mode never calls
    /// it on the write path) plus an `embed_images` half with its own counter.
    struct CountingClip {
        text_calls: AtomicUsize,
        image_calls: AtomicUsize,
        fp: &'static str,
    }
    impl Embedder for CountingClip {
        fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            self.text_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { Ok(HashEmbedder::embed_sync(&texts)) })
        }
        fn fingerprint(&self) -> Option<String> {
            Some(self.fp.to_string())
        }
        fn dim(&self) -> Option<usize> {
            Some(64)
        }
    }

    /// The image half — counts image embeds, returns a fixed-dim vector per item.
    struct ClipImages(Arc<CountingClip>);
    impl ImageEmbedder for ClipImages {
        fn embed_images(
            &self,
            images: Vec<MediaItem>,
        ) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            self.0.image_calls.fetch_add(1, Ordering::SeqCst);
            let n = images.len();
            Box::pin(async move { Ok(vec![vec![1.0f32; 8]; n]) })
        }
    }

    /// Upsert one note carrying `<img src=NAME>` in Front; returns its id.
    async fn upsert_image_note(kernel: &Kernel, front: &str) -> i64 {
        let notes = serde_json::json!([{
            "note_type": "Basic", "deck": "Default",
            "fields": {"Front": front, "Back": "b"}
        }]);
        let results: Vec<serde_json::Value> = serde_json::from_str(
            &upsert_wire(kernel, notes.to_string(), "error".into(), false).await,
        )
        .unwrap();
        // The write is fire-and-forget; settle the drain so the index/derived
        // effects are visible to the test's immediate assertions.
        kernel.settle().await;
        results[0]["id"].as_i64().unwrap()
    }

    /// Build a kernel with a SEPARATE text-primary + image-primary CLIP, sharing
    /// a media map. Returns (kernel, dir, text, clip) — the counters live on
    /// `text`/`clip`.
    async fn two_space_kernel(
        dir: &std::path::Path,
        media: std::collections::HashMap<String, Vec<u8>>,
    ) -> (Kernel, Arc<CountingText>, Arc<CountingClip>) {
        let kernel = Kernel::open(
            dir.join("c.anki2").to_str().unwrap(),
            dir.join("cache").to_str().unwrap(),
        )
        .await
        .unwrap();
        let text = Arc::new(CountingText {
            calls: AtomicUsize::new(0),
            fp: "text-primary:v1",
        });
        let clip = Arc::new(CountingClip {
            text_calls: AtomicUsize::new(0),
            image_calls: AtomicUsize::new(0),
            fp: "clip-primary:v1",
        });
        // The text-primary FIRST (so it's the per-modality text primary), then
        // the image-capable CLIP as a SEPARATE space (image primary).
        kernel.attach_embedder(Arc::clone(&text) as Arc<dyn Embedder>, None);
        let images: KernelImages = (
            Box::new(ClipImages(Arc::clone(&clip))),
            Box::new(MapResolver::new(media)),
        );
        kernel.attach_embedder_space(
            Some("clip-primary:v1".into()),
            Arc::clone(&clip) as Arc<dyn Embedder>,
            Some(images),
        );
        (kernel, text, clip)
    }

    #[test]
    fn per_space_write_routes_text_and_image_to_their_primaries() {
        // A dedicated text space + a separate CLIP image space. A note's
        // text lands in the text space; its image lands in the CLIP space's
        // ImageOnly index. The CLIP text tower is NEVER called on the write path
        // (ImageOnly), and the image space holds the image vector.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let mut media = std::collections::HashMap::new();
            media.insert("a.png".to_string(), b"image-a-bytes".to_vec());
            let (kernel, _text, clip) = two_space_kernel(&dir, media).await;
            kernel.reindex_if_needed().await.unwrap();

            let nid = upsert_image_note(&kernel, "the krebs cycle <img src=\"a.png\">").await;
            assert!(
                kernel.index().engine().contains(nid),
                "text space has the note"
            );
            let clip_orch = kernel
                .index_set()
                .orchestrator_for("clip-primary:v1")
                .expect("the CLIP image space is bound");
            assert!(
                clip_orch.engine().modality_contains("image", nid),
                "the image space holds the note's image vector"
            );
            assert!(
                clip.image_calls.load(Ordering::SeqCst) >= 1,
                "clip embedded images"
            );
            assert_eq!(
                clip.text_calls.load(Ordering::SeqCst),
                0,
                "ImageOnly never calls the CLIP text tower on the write path"
            );
            // The text space's engine holds NO image-space vector for the note
            // beyond its own (text-space image modality is unused here).
            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn per_space_hashing_only_re_embeds_the_changed_modality() {
        // The load-bearing hashing property, on the RECONCILE (drift) path — the
        // route where per-space hashing decides what re-embeds. Each space's
        // orchestrator reconciles against ITS OWN hashes: a same-text /
        // different-image input re-embeds the IMAGE space only; a
        // different-text / same-image input re-embeds the TEXT space only; an
        // unchanged input re-embeds neither. (A direct upsert always re-embeds
        // the written note in both spaces — the hashing governs DRIFT.)
        use index_orchestrator::{EmbedInput, WriteMode};

        fn input(nid: i64, text: &str, image: &str) -> EmbedInput {
            EmbedInput {
                note_id: nid,
                text: text.to_owned(),
                image_names: vec![image.to_owned()],
                ocr_texts: vec![],
            }
        }

        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let mut media = std::collections::HashMap::new();
            media.insert("a.png".to_string(), b"aaa".to_vec());
            media.insert("b.png".to_string(), b"bbb".to_vec());
            let (kernel, text, clip) = two_space_kernel(&dir, media.clone()).await;

            let text_orch = kernel.index().clone();
            let clip_orch = kernel
                .index_set()
                .orchestrator_for("clip-primary:v1")
                .unwrap();

            let text_svc = kernel.embed_service().unwrap();
            let (clip_orch2, clip_svc) = kernel.image_only_route().unwrap();
            assert!(Arc::ptr_eq(&clip_orch, &clip_orch2));
            let (ie, res) = clip_svc.images_pair().unwrap();

            // Seed both spaces (text→text orch; image→clip orch ImageOnly).
            let v1 = vec![input(1, "front text", "a.png")];
            text_orch
                .reconcile(
                    v1.clone(),
                    1,
                    text_svc.embedder.fingerprint(),
                    &*text_svc.embedder,
                    None,
                )
                .await
                .unwrap();
            clip_orch
                .reconcile_with_mode(
                    v1,
                    1,
                    clip_svc.embedder.fingerprint(),
                    &*clip_svc.embedder,
                    Some((ie, res)),
                    WriteMode::ImageOnly,
                )
                .await
                .unwrap();
            let (t0, i0) = (
                text.calls.load(Ordering::SeqCst),
                clip.image_calls.load(Ordering::SeqCst),
            );

            // ── PURE-IMAGE change: same text, a.png → b.png. ──
            let (ie, res) = clip_svc.images_pair().unwrap();
            let v_img = vec![input(1, "front text", "b.png")];
            text_orch
                .reconcile(
                    v_img.clone(),
                    2,
                    text_svc.embedder.fingerprint(),
                    &*text_svc.embedder,
                    None,
                )
                .await
                .unwrap();
            clip_orch
                .reconcile_with_mode(
                    v_img,
                    2,
                    clip_svc.embedder.fingerprint(),
                    &*clip_svc.embedder,
                    Some((ie, res)),
                    WriteMode::ImageOnly,
                )
                .await
                .unwrap();
            assert_eq!(
                text.calls.load(Ordering::SeqCst),
                t0,
                "pure-image change: the text space did NOT re-embed (text hash stable)"
            );
            assert!(
                clip.image_calls.load(Ordering::SeqCst) > i0,
                "pure-image change: the image space re-embedded (image hash moved)"
            );
            let (t1, i1) = (
                text.calls.load(Ordering::SeqCst),
                clip.image_calls.load(Ordering::SeqCst),
            );

            // ── PURE-TEXT change: edit text, keep image b.png. ──
            let (ie, res) = clip_svc.images_pair().unwrap();
            let v_txt = vec![input(1, "EDITED text", "b.png")];
            text_orch
                .reconcile(
                    v_txt.clone(),
                    3,
                    text_svc.embedder.fingerprint(),
                    &*text_svc.embedder,
                    None,
                )
                .await
                .unwrap();
            clip_orch
                .reconcile_with_mode(
                    v_txt,
                    3,
                    clip_svc.embedder.fingerprint(),
                    &*clip_svc.embedder,
                    Some((ie, res)),
                    WriteMode::ImageOnly,
                )
                .await
                .unwrap();
            assert!(
                text.calls.load(Ordering::SeqCst) > t1,
                "pure-text change: the text space re-embedded"
            );
            assert_eq!(
                clip.image_calls.load(Ordering::SeqCst),
                i1,
                "pure-text change: the image space did NOT re-embed (image hash stable)"
            );

            kernel.close().await.unwrap();
        });
    }

    // ── 0 / 1 / N graceful degradation ───────────────────────────────

    #[test]
    fn delete_fans_out_to_every_space_at_n2() {
        // A note's vectors leave EVERY space on delete: the text
        // space drops its text vector, the CLIP space drops its image vector.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let mut media = std::collections::HashMap::new();
            media.insert("a.png".to_string(), b"aaa".to_vec());
            let (kernel, _text, _clip) = two_space_kernel(&dir, media).await;
            kernel.reindex_if_needed().await.unwrap();

            let nid = upsert_image_note(&kernel, "front <img src=\"a.png\">").await;
            let clip_orch = kernel
                .index_set()
                .orchestrator_for("clip-primary:v1")
                .unwrap();
            assert!(kernel.index().engine().contains(nid), "text space has it");
            assert!(
                clip_orch.engine().modality_contains("image", nid),
                "image space has it"
            );

            // Delete → both spaces drop the note (remove_all fans out). The
            // vector removal rides the drain, so settle before asserting it's gone.
            assert_eq!(
                kernel.delete_notes(vec![nid]).await.unwrap().deleted,
                vec![nid]
            );
            kernel.settle().await;
            assert!(
                !kernel.index().engine().contains(nid),
                "gone from text space"
            );
            assert!(
                !clip_orch.engine().contains(nid),
                "gone from image space too (fan-out)"
            );
            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn delete_notes_partitions_deleted_and_not_found_in_one_op() {
        // The maintained kernel delete_notes returns {deleted, not_found}
        // in its single write job (existence partition + delete + sidecar drop),
        // so the action routes through it instead of wrapper.delete_notes +
        // a separate forget_notes. A requested id that doesn't exist is
        // not_found (never silently counted deleted), and a real id's vectors +
        // derived rows leave in the same op.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();
            let basic = kernel.notetype_id("Basic").await.unwrap();
            let CreateOutcome::Created(nid) = kernel
                .upsert_note(
                    basic,
                    1,
                    vec!["mitochondria powerhouse".into(), "b".into()],
                    vec![],
                    DuplicatePolicy::Error,
                )
                .await
                .unwrap()
            else {
                panic!("create failed")
            };
            kernel.settle().await;
            assert!(kernel.index().engine().contains(nid), "indexed");
            // Lexical row exists too (derived store ingested on upsert).
            assert!(!kernel
                .derived
                .search_substring("mitochondria", 5, None, &[])
                .unwrap()
                .unwrap()
                .is_empty());

            // Delete a mix: the real id + a bogus one. ONE op partitions them.
            let bogus = nid + 99_999;
            let resp = kernel.delete_notes(vec![nid, bogus]).await.unwrap();
            assert_eq!(resp.deleted, vec![nid], "the real id is deleted");
            assert_eq!(resp.not_found, vec![bogus], "the bogus id is not_found");
            // The sidecar removal runs off the drain (the per-op derived write
            // rides the compute pool); settle the queue before asserting on it.
            kernel.settle().await;

            // Sidecars dropped in the SAME op — no separate forget_notes needed.
            assert!(
                !kernel.index().engine().contains(nid),
                "vector dropped in the maintained op"
            );
            assert!(
                kernel
                    .derived
                    .search_substring("mitochondria", 5, None, &[])
                    .unwrap()
                    .unwrap()
                    .is_empty(),
                "derived row dropped in the maintained op"
            );
            // No drift: the watermark advanced in-op (the maintained-write tail).
            assert!(!kernel.reindex_if_needed().await.unwrap());

            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn search_with_no_embedder_degrades_to_lexical_only() {
        // 0-space search: no text-capable space → semantic is off, the
        // cross-space fan-out is empty, and search serves the LEXICAL signals
        // only (no empty-index panic). The literal hit lands via `exact`.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            // No embedder attached at all → zero spaces.
            assert_eq!(kernel.embed_space_count(), 0);
            // build_cross_space is empty + safe with zero spaces.
            assert!(kernel
                .build_cross_space(&["anything".to_string()], 10)
                .await
                .unwrap()
                .is_empty());

            let basic = kernel.notetype_id("Basic").await.unwrap();
            let CreateOutcome::Created(nid) = kernel
                .upsert_note(
                    basic,
                    1,
                    vec!["the krebs cycle in biology".into(), "b".into()],
                    vec![],
                    DuplicatePolicy::Error,
                )
                .await
                .unwrap()
            else {
                panic!("create works without an embedder")
            };
            // Even with no embedder, the lexical (FTS5) derived write rides the
            // async drain — settle before the substring search reads it back.
            kernel.settle().await;
            // Lexical-only: the literal substring hit lands; NO text signal.
            let hits = kernel.search("krebs cycle", 5).await.unwrap();
            assert_eq!(hits[0].note_id, nid);
            assert!(
                hits[0].signals.iter().all(|(s, _)| s != "text"),
                "no semantic signal with zero spaces"
            );
            assert!(
                hits[0]
                    .signals
                    .iter()
                    .any(|(s, _)| s == "exact" || s == "fuzzy"),
                "the lexical signal carried the hit"
            );
            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn upsert_neighbors_pin_to_the_primary_text_space_across_n() {
        // The dedup neighbor path pins to the PRIMARY text space's engine across
        // N spaces: adding a separate CLIP space must not change which
        // engine the text-neighbor query reads — it stays the primary text
        // engine, so the dedup signal is deterministic, never ambiguous across N.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let mut media = std::collections::HashMap::new();
            media.insert("a.png".to_string(), b"aaa".to_vec());
            let (kernel, _text, _clip) = two_space_kernel(&dir, media).await;
            kernel.reindex_if_needed().await.unwrap();

            // The neighbor path reads the PRIMARY engine (index()), and searches
            // the `text` modality — independent of the CLIP image space.
            let primary_engine = kernel.index().engine_arc();
            let tag_engine = kernel.index_set().tag_engine();
            assert!(
                Arc::ptr_eq(&primary_engine, &tag_engine),
                "the primary/text engine is the one the neighbor + tag paths read"
            );
            // The CLIP image space's engine is a DISTINCT engine (not what
            // neighbors read).
            let clip_engine = kernel
                .index_set()
                .orchestrator_for("clip-primary:v1")
                .unwrap()
                .engine_arc();
            assert!(
                !Arc::ptr_eq(&primary_engine, &clip_engine),
                "the CLIP space is a separate engine; neighbors never read it"
            );
            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn composed_kernel_serves_ops_over_injected_stores() {
        // The injection seam: a kernel assembled from PRE-BUILT stores
        // (the deployment ladder's composition point) behaves like an opened
        // one — the actor wraps the injected collection, ops serve, and the
        // index/derived paths ride the injected trait objects.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            std::fs::create_dir_all(dir.join("cache")).unwrap();
            let collection: Arc<dyn shrike_collection::Collection> =
                Arc::new(CollectionCore::open(dir.join("c.anki2").to_str().unwrap()).unwrap());
            let index: Arc<dyn VectorIndex> =
                Arc::new(MultiModalIndex::new(vec!["text".to_owned()]).unwrap());
            let derived: Arc<dyn DerivedStore> = Arc::new(
                DerivedEngine::open(
                    dir.join("cache/shrike.db").to_str().unwrap(),
                    DerivedEngine::SCHEMA_VERSION,
                )
                .unwrap(),
            );
            let kernel = Kernel::compose(
                collection,
                index,
                derived,
                dir.join("cache").to_str().unwrap(),
                None,
                None,
            )
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();

            let basic = kernel.notetype_id("Basic").await.unwrap();
            let CreateOutcome::Created(nid) = kernel
                .upsert_note(
                    basic,
                    1,
                    vec!["composed kernels serve".into(), "b".into()],
                    vec![],
                    DuplicatePolicy::Error,
                )
                .await
                .unwrap()
            else {
                panic!("create failed")
            };
            kernel.settle().await;
            assert!(kernel.index().engine().contains(nid));
            let hits = kernel.search("composed kernels", 5).await.unwrap();
            assert_eq!(hits[0].note_id, nid);

            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn media_ops_and_prune_run_as_kernel_ops() {
        // store_media's byte prepare rides the blocking
        // pool with per-item errors, the read ops round-trip, and prune
        // carries its own maintenance tail (vectors leave with the notes,
        // the watermark advances — no host-side forget/metadata calls).
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();

            // store: a good data item + a sourceless one — the bad slot
            // errors, the batch survives, indexes echo positions.
            let items = vec![
                shrike_schemas::StoreMediaItem {
                    filename: Some("pic.png".into()),
                    data: Some("UE5HREFUQQ==".into()), // b64("PNGDATA")
                    ..Default::default()
                },
                shrike_schemas::StoreMediaItem::default(),
            ];
            let stored = kernel.store_media(items, false, Vec::new()).await.unwrap();
            assert!(matches!(
                &stored[0],
                shrike_schemas::StoreMediaResult::Stored { index: 0, filename, .. }
                    if filename == "pic.png"
            ));
            assert!(matches!(
                &stored[1],
                shrike_schemas::StoreMediaResult::Error { index: 1, .. }
            ));

            // fetch/list/check round-trip on the kernel ops.
            let fetched = kernel
                .fetch_media(vec!["pic.png".into(), "ghost.png".into()])
                .await
                .unwrap();
            assert!(matches!(
                &fetched[0],
                shrike_schemas::MediaFetchResult::Found { .. }
            ));
            assert!(matches!(
                &fetched[1],
                shrike_schemas::MediaFetchResult::Missing { .. }
            ));
            assert_eq!(kernel.list_media(None, None).await.unwrap().count, 1);
            assert_eq!(kernel.media_check().await.unwrap().unused.len(), 1);

            // A note that goes blank via a raw (external-style) edit: prune
            // applies, and the kernel tail drops its vector + advances the
            // watermark — no drift on the next reindex check.
            let basic = kernel.notetype_id("Basic").await.unwrap();
            let CreateOutcome::Created(nid) = kernel
                .upsert_note(
                    basic,
                    1,
                    vec!["temp".into(), "".into()],
                    vec!["onlytag".into()],
                    DuplicatePolicy::Error,
                )
                .await
                .unwrap()
            else {
                panic!("create failed")
            };
            kernel.settle().await;
            assert!(kernel.index().engine().contains(nid));
            kernel
                .collection
                .run(move |core| core.update_note(nid, &["".into(), "".into()], None))
                .await
                .unwrap()
                .unwrap();

            let preview = kernel
                .collection_prune(true, true, true, true, true)
                .await
                .unwrap();
            assert!(preview.dry_run);
            kernel.settle().await;
            assert!(kernel.index().engine().contains(nid)); // untouched

            let applied = kernel
                .collection_prune(true, true, true, true, false)
                .await
                .unwrap();
            assert!(!applied.dry_run);
            assert_eq!(applied.empty_notes.unwrap().removed, vec![nid]);
            assert!(!kernel.index().engine().contains(nid));
            assert!(!kernel.reindex_if_needed().await.unwrap());

            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn tag_deck_ops_carry_the_metadata_tail() {
        // The metadata tail: a real change advances the
        // index watermark to the new col_mod (no drift on the next check);
        // a no-op batch leaves the watermark EXACTLY where it was — the
        // `changed` guard skips the tail, not merely "no drift afterwards".
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();
            let basic = kernel.notetype_id("Basic").await.unwrap();
            let CreateOutcome::Created(nid) = kernel
                .upsert_note(
                    basic,
                    1,
                    vec!["front".into(), "back".into()],
                    vec!["t".into()],
                    DuplicatePolicy::Error,
                )
                .await
                .unwrap()
            else {
                panic!("create failed")
            };

            // Settle the upsert's drain so `before` reflects its watermark.
            kernel.settle().await;
            let watermark = |k: &Kernel| k.index().status().col_mod;
            let before = watermark(&kernel);

            // No-op: a non-existent note modifies nothing → the tail is
            // skipped and the watermark doesn't move at all.
            let miss = kernel
                .update_note_tags(vec![999_999], None, vec!["x".into()], Vec::new())
                .await
                .unwrap();
            assert_eq!(miss.notes_modified, 0);
            kernel.settle().await;
            assert_eq!(watermark(&kernel), before);

            // All-error deck batch: same skip.
            let errs = kernel
                .upsert_decks(vec![shrike_schemas::DeckInput {
                    id: Some(999_999),
                    name: "X".into(),
                }])
                .await
                .unwrap();
            assert!(matches!(
                errs[0],
                shrike_schemas::UpsertDeckResult::Error { .. }
            ));
            kernel.settle().await;
            assert_eq!(watermark(&kernel), before);

            // A real tag edit: the tail advances the watermark to the new
            // col_mod, so the bump never reads as drift.
            let hit = kernel
                .update_note_tags(vec![nid], None, vec!["fresh".into()], Vec::new())
                .await
                .unwrap();
            assert_eq!(hit.notes_modified, 1);
            kernel.settle().await;
            assert_eq!(watermark(&kernel), Some(kernel.col_mod().await.unwrap()));
            assert!(!kernel.reindex_if_needed().await.unwrap());

            // Deck create + empty-delete: both tails fire.
            kernel
                .upsert_decks(vec![shrike_schemas::DeckInput {
                    id: None,
                    name: "Temp".into(),
                }])
                .await
                .unwrap();
            let del = kernel.delete_decks(vec!["Temp".into()]).await.unwrap();
            assert_eq!(del.deleted, vec!["Temp"]);
            kernel.settle().await;
            assert_eq!(watermark(&kernel), Some(kernel.col_mod().await.unwrap()));
            assert!(!kernel.reindex_if_needed().await.unwrap());

            kernel.close().await.unwrap();
        });
    }

    /// A METADATA-ONLY op (deck rename) must NOT fire a tag-centroid
    /// recompute (no centroid input moved); a TAG-membership op must. The
    /// GREEN-proof: mutate tags OUT OF BAND (straight through the
    /// collection actor, so the mutation itself fires no `request()`), then —
    /// because `recompute` re-reads `note_tag_rows` from the live collection —
    /// a fresh out-of-band tag appears in the centroid key map ONLY IF a
    /// recompute actually ran. So a deck op leaving it ABSENT proves no
    /// recompute; a tag op making it APPEAR proves the relevant refresh still
    /// fires.
    #[test]
    fn metadata_only_op_does_not_recompute_tag_centroids() {
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();
            let basic = kernel.notetype_id("Basic").await.unwrap();

            // 6 UNTAGGED notes first (so total_notes is settled), then tag two of
            // them "alpha" in one op. Coverage 2/6 < 0.5 → a real centroid (if we
            // tagged during creation, the first refresh would fire when only the
            // 2 tagged notes existed → coverage 1.0 > 0.5 → alpha excluded).
            let mut alpha_ids = Vec::new();
            for i in 0..6 {
                let CreateOutcome::Created(id) = kernel
                    .upsert_note(
                        basic,
                        1,
                        vec![format!("note {i} text"), "back".into()],
                        vec![],
                        DuplicatePolicy::Error,
                    )
                    .await
                    .unwrap()
                else {
                    panic!("create failed")
                };
                if i < 2 {
                    alpha_ids.push(id);
                }
            }
            // Tag two notes "alpha" in one op (a tag-membership change → refresh).
            kernel
                .update_note_tags(alpha_ids.clone(), None, vec!["alpha".into()], Vec::new())
                .await
                .unwrap();
            // Wait for "alpha" to land so we have a known-good baseline.
            wait_for_tags(&kernel, |k| {
                k.lookup(tag_centroids::tag_key("alpha")).is_some()
            })
            .await;
            // Drain step 1's background refresh to QUIESCENCE before the
            // out-of-band beta write. The recompute reads the collection BEFORE
            // taking its exclusion lock, so a not-yet-started background run could
            // otherwise read the collection AFTER the beta write and pull beta in
            // with no tag op — beta appearing spuriously under a starved scheduler
            // (#880). Awaiting idle fences that: once it returns, no step-1
            // refresh can still read a later collection state.
            kernel.await_tag_quiesce().await;
            let beta = tag_centroids::tag_key("beta");
            assert!(
                kernel.tag_keys().lookup(beta).is_none(),
                "precondition: no beta centroid yet"
            );

            // OUT-OF-BAND: add "beta" to the alpha notes straight through the
            // collection actor — this changes membership in the collection but
            // does NOT call tag_refresh.request() (it bypasses
            // kernel.update_note_tags). The centroid map can only learn "beta"
            // via a fresh recompute that re-reads note_tag_rows.
            let oob_ids = alpha_ids.clone();
            kernel
                .collection()
                .run(move |core| core.update_note_tags(&oob_ids, None, &["beta".to_string()], &[]))
                .await
                .unwrap()
                .unwrap();

            // A METADATA-ONLY op: a deck create. With the fix it passes
            // membership_may_have_changed=false → no recompute → "beta" stays
            // invisible to the centroid map.
            kernel
                .upsert_decks(vec![shrike_schemas::DeckInput {
                    id: None,
                    name: "SomeDeck".into(),
                }])
                .await
                .unwrap();
            // Deterministic barriers instead of a fixed sleep window: settle()
            // drains the deck op's ingest tail (where a wrongly-triggered
            // `tag_refresh.request()` would fire), and await_tag_quiesce() then
            // lets any such refresh RUN TO COMPLETION. So a deck op that wrongly
            // recomputed would have pulled beta in by now and the assertion fires;
            // a correct one leaves beta absent. A fixed sleep raced a late refresh
            // under starvation and spuriously populated beta (#880); this can't.
            kernel.settle().await;
            kernel.await_tag_quiesce().await;
            assert!(
                kernel.tag_keys().lookup(beta).is_none(),
                "DEFECT (#600): a deck op fired a full tag-centroid recompute \
                 with no relevance probe (beta appeared without a tag op)"
            );

            // A TAG-membership op: update_note_tags (adds a throwaway tag) passes
            // membership_may_have_changed=true → the recompute runs, re-reads the
            // collection, and now SEES the out-of-band beta membership.
            kernel
                .update_note_tags(alpha_ids.clone(), None, vec!["gamma".into()], Vec::new())
                .await
                .unwrap();
            wait_for_tags(&kernel, |k| k.lookup(beta).is_some()).await;

            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn note_type_ops_run_as_kernel_ops() {
        // The metadata-tail ops advance
        // the watermark inside the kernel (field metadata unconditionally, a
        // template/CSS replace only when something matched), and an applied
        // migration re-embeds the changed notes — no host-side
        // metadata_changed/reindex_notes calls anywhere.
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// HashEmbedder vectors + an embed-call counter, so "migrate-apply
        /// re-embeds" is a direct observation, not a watermark inference.
        struct CountingEmbedder(AtomicUsize);

        impl Embedder for CountingEmbedder {
            fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Box::pin(async move { Ok(HashEmbedder::embed_sync(&texts)) })
            }

            fn fingerprint(&self) -> Option<String> {
                Some("hash-embedder:v1".to_string())
            }

            fn dim(&self) -> Option<usize> {
                Some(64)
            }
        }

        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            let embedder = Arc::new(CountingEmbedder(AtomicUsize::new(0)));
            kernel.attach_embedder(Arc::clone(&embedder) as Arc<dyn Embedder>, None);
            kernel.reindex_if_needed().await.unwrap();

            // Field metadata: the unconditional watermark tail — the persist
            // bumps col.mod, yet the next drift check sees nothing.
            let resp = kernel
                .update_note_type_field_metadata(
                    "Basic".into(),
                    vec![shrike_schemas::FieldMetadataInput {
                        name: "Front".into(),
                        font: None,
                        size: Some(28),
                        description: None,
                    }],
                )
                .await
                .unwrap();
            assert_eq!(resp.fields_updated, vec!["Front"]);
            assert!(!kernel.reindex_if_needed().await.unwrap());

            // find_replace: a real replace advances the watermark too.
            let basic_css = kernel
                .find_replace_note_types(
                    "Basic".into(),
                    "text-align: center".into(),
                    "text-align: left".into(),
                    false,
                    true,
                    false,
                    false,
                    true,
                )
                .await
                .unwrap();
            assert!(basic_css.replacements > 0);
            assert!(!kernel.reindex_if_needed().await.unwrap());

            // Migrate-apply: field content moves under an unchanged id, so
            // the kernel tail re-embeds it (an extra embed call lands, the
            // vector stays present, no drift remains).
            let basic = kernel.notetype_id("Basic").await.unwrap();
            let CreateOutcome::Created(nid) = kernel
                .upsert_note(
                    basic,
                    1,
                    vec!["hello front".into(), "back".into()],
                    Vec::new(),
                    DuplicatePolicy::Error,
                )
                .await
                .unwrap()
            else {
                panic!("create failed")
            };
            kernel.settle().await;
            let embeds_before = embedder.0.load(Ordering::SeqCst);

            let field_map: std::collections::BTreeMap<String, String> = [
                ("Front".to_string(), "Text".to_string()),
                ("Back".to_string(), "Back Extra".to_string()),
            ]
            .into();
            let dry = kernel
                .migrate_note_type(
                    vec![nid],
                    "Cloze".into(),
                    field_map.clone(),
                    Default::default(),
                    true,
                )
                .await
                .unwrap();
            assert!(dry.dry_run);
            assert_eq!(embedder.0.load(Ordering::SeqCst), embeds_before); // untouched

            let applied = kernel
                .migrate_note_type(
                    vec![nid],
                    "Cloze".into(),
                    field_map,
                    Default::default(),
                    false,
                )
                .await
                .unwrap();
            assert_eq!(applied.changed, vec![nid]);
            kernel.settle().await;
            assert!(embedder.0.load(Ordering::SeqCst) > embeds_before);
            kernel.settle().await;
            assert!(kernel.index().engine().contains(nid));
            assert!(!kernel.reindex_if_needed().await.unwrap());

            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn kernel_ops_reopen_after_cooperative_release() {
        // Open-on-demand, kernel-side: an idle release between ops
        // (or between one op's jobs) self-heals on the next serialized job
        // instead of erroring CollectionNotOpen.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
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
            let results = upsert_wire(
                &kernel,
                r#"[{"note_type": "Basic", "deck": "Default",
                         "fields": {"Front": "second", "Back": "b"}}]"#
                    .to_string(),
                "error".to_string(),
                false,
            )
            .await;
            assert!(results.contains("\"created\""), "got: {results}");

            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn tag_centroids_build_and_never_leak_into_note_search() {
        // The tag-centroid layer end to end: tagged upserts → centroids in the
        // tag.text space (hygiene-filtered) → note searches structurally
        // blind to tag keys.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
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
            upsert_wire(
                &kernel,
                serde_json::json!(notes).to_string(),
                "error".to_string(),
                false,
            )
            .await;

            wait_for_tags(&kernel, |k| !k.is_empty()).await;
            let keys = kernel.tag_keys();
            let cell_key = tag_centroids::tag_key("bio::cell");
            assert_eq!(keys.lookup(cell_key).as_deref(), Some("bio::cell"));
            assert_eq!(
                keys.lookup(tag_centroids::tag_key("bio")).as_deref(),
                Some("bio"),
                "hierarchy rolls up"
            );
            let index = kernel.index();
            assert!(index
                .engine()
                .modality_get(TAG_TEXT_SPACE, cell_key)
                .is_some());

            // A note search never surfaces a tag key.
            let hits = kernel.search("note number", 20).await.unwrap();
            assert!(hits.iter().all(|h| keys.lookup(h.note_id).is_none()));

            // A tag-only edit (rename) rides metadata_changed, not an index
            // op — it must also schedule a refresh so the key map tracks
            // the new name.
            kernel
                .collection()
                .run(|core| core.rename_tag("bio::cell", "bio::organelle", &[]))
                .await
                .unwrap()
                .unwrap();
            // A rename is a membership-relevant change → request the refresh.
            kernel.metadata_changed(true).await.unwrap();
            wait_for_tags(&kernel, |k| {
                k.lookup(tag_centroids::tag_key("bio::organelle")).is_some()
                    && k.lookup(tag_centroids::tag_key("bio::cell")).is_none()
            })
            .await;
            let cell_key = tag_centroids::tag_key("bio::organelle");

            // Deleting a member triggers the refresh through the delete
            // tail's membership probe: the tag falls below
            // min_members and its centroid retires.
            let victim = kernel.tag_keys().members(cell_key)[0];
            kernel.delete_notes(vec![victim]).await.unwrap();
            wait_for_tags(&kernel, |k| k.lookup(cell_key).is_none()).await;

            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn tag_signal_surfaces_off_topic_members() {
        // The tag-signal payoff: a member whose own text doesn't match the query
        // still surfaces because its TAG's centroid (dominated by on-topic
        // siblings) activates and expands.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
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
                &upsert_wire(
                    &kernel,
                    serde_json::json!(notes).to_string(),
                    "error".to_string(),
                    false,
                )
                .await,
            )
            .unwrap();
            let mnemonic_id = results[2]["id"].as_i64().unwrap();

            let key = tag_centroids::tag_key("krebs");
            wait_for_tags(&kernel, |k| k.members(key).len() == 3).await;
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
        });
    }

    struct StubRecognizer;

    impl Recognizer for StubRecognizer {
        fn recognize(
            &self,
            items: Vec<MediaItem>,
        ) -> BoxFuture<'_, NativeResult<Vec<Recognition>>> {
            Box::pin(async move {
                Ok(items
                    .iter()
                    .map(|b| {
                        let text = String::from_utf8_lossy(&b.bytes).to_string();
                        Recognition {
                            text: text.clone(),
                            confidence: 0.95,
                            segments: vec![Segment {
                                text,
                                confidence: 0.95,
                                locator: Some(Locator::Bbox([0.0, 0.0, 1.0, 0.1])),
                            }],
                        }
                    })
                    .collect())
            })
        }

        fn fingerprint(&self) -> Option<String> {
            Some("stub:v1".to_string())
        }
    }

    struct MapResolver {
        files: std::collections::HashMap<String, Vec<u8>>,
        /// Names whose `read` fails even though `exists` reports them
        /// present — the TOCTOU-delete / transient-read shape. A
        /// Mutex so a test can "heal" the read mid-flight.
        unreadable: std::sync::Mutex<std::collections::HashSet<String>>,
    }

    impl MapResolver {
        fn new(files: std::collections::HashMap<String, Vec<u8>>) -> Self {
            Self {
                files,
                unreadable: std::sync::Mutex::new(std::collections::HashSet::new()),
            }
        }
    }

    impl ImageResolver for MapResolver {
        fn read(&self, name: &str) -> Option<Vec<u8>> {
            if self.unreadable.lock().unwrap().contains(name) {
                return None;
            }
            self.files.get(name).cloned()
        }

        fn exists(&self, name: &str) -> bool {
            self.files.contains_key(name)
        }
    }

    #[test]
    fn recognition_pipeline_mints_lexical_and_vector_consumers() {
        // Recognition end-to-end: one recognition pass per image feeds the
        // lexical store (rows + segments) AND the text-space vector, so a
        // query matching only the text INSIDE an image surfaces the note.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();

            let notes = vec![
                serde_json::json!({"note_type": "Basic", "deck": "Default",
                    "fields": {"Front": "See the diagram <img src=\"krebs.png\">", "Back": "b"}}),
                serde_json::json!({"note_type": "Basic", "deck": "Default",
                    "fields": {"Front": "zzz filler mnemonic", "Back": "b"}}),
            ];
            let results: Vec<serde_json::Value> = serde_json::from_str(
                &upsert_wire(
                    &kernel,
                    serde_json::json!(notes).to_string(),
                    "error".to_string(),
                    false,
                )
                .await,
            )
            .unwrap();
            let diagram_id = results[0]["id"].as_i64().unwrap();

            // No recognizer attached → the sweep reports unavailable.
            let off = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(off, recognize::SweepReport::Unavailable);

            let mut media = std::collections::HashMap::new();
            media.insert(
                "krebs.png".to_string(),
                b"the krebs cycle produces energy carriers".to_vec(),
            );
            kernel.attach_recognizer(Arc::new(StubRecognizer), Arc::new(MapResolver::new(media)));

            let report = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(
                report,
                recognize::SweepReport::Ran {
                    recognized: 1,
                    stored: 1,
                    remaining: 0
                }
            );

            // Lexical consumer: a literal phrase living ONLY inside the image
            // hits, with ocr provenance riding the existing seam.
            let hits = kernel.search("energy carriers", 5).await.unwrap();
            let hit = hits
                .iter()
                .find(|h| h.note_id == diagram_id)
                .expect("ocr text searchable");
            assert!(hit.signals.iter().any(|(sig, _)| sig == "exact"));

            // Vector consumer: a non-literal query (no substring) still
            // surfaces the note through its OCR vector in the text space.
            let sem = kernel.search("carriers energy", 5).await.unwrap();
            let sem_hit = sem
                .iter()
                .find(|h| h.note_id == diagram_id)
                .expect("ocr vector ranks");
            assert!(sem_hit.signals.iter().any(|(sig, _)| sig == "text"));

            // The sweep is idempotent: a second call has nothing to do.
            let again = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(again, recognize::SweepReport::Idle);

            // Persistence + reconcile==rebuild: a fresh kernel over the same
            // cache sees no drift (the hash folds the OCR text) and still
            // serves both consumers.
            kernel.close().await.unwrap();
            let kernel2 = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel2.attach_embedder(Arc::new(HashEmbedder), None);
            kernel2.reindex_if_needed().await.unwrap();
            let sem2 = kernel2.search("carriers energy", 5).await.unwrap();
            assert!(sem2.iter().any(|h| h.note_id == diagram_id));

            kernel2.close().await.unwrap();
        });
    }

    #[test]
    fn recognition_does_not_advance_the_index_or_derived_col_mod_watermark() {
        // C1: recognition stores media-derived text, which never changes
        // col.mod, so a recognition sweep must NOT advance the index/derived
        // col.mod watermark — advancing it would over-certify a concurrent,
        // still-queued user write whose col.mod the sweep happened to read, and
        // recognition never bumps col.mod, so drift could never re-detect it.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);

            let notes = vec![serde_json::json!({"note_type": "Basic", "deck": "Default",
                "fields": {"Front": "diagram <img src=\"krebs.png\">", "Back": "b"}})];
            upsert_wire(
                &kernel,
                serde_json::json!(notes).to_string(),
                "error".to_string(),
                false,
            )
            .await;
            kernel.reindex_if_needed().await.unwrap();

            // The watermarks the reindex stamped — recognition must leave them
            // exactly here (it certifies no text indexing).
            let index_before = kernel.index().status().col_mod;
            let derived_before = kernel.derived.get_col_mod();

            let mut media = std::collections::HashMap::new();
            media.insert(
                "krebs.png".to_string(),
                b"the krebs cycle produces energy carriers".to_vec(),
            );
            kernel.attach_recognizer(Arc::new(StubRecognizer), Arc::new(MapResolver::new(media)));
            let report = kernel.recognize_pending(10).await.unwrap();
            assert!(
                matches!(report, recognize::SweepReport::Ran { stored: 1, .. }),
                "the sweep stored the recognized text + vector"
            );
            // Settle the coalesced tag refresh the re-embed requested.
            kernel.settle().await;

            assert_eq!(
                kernel.index().status().col_mod,
                index_before,
                "recognition must NOT advance the index col.mod watermark"
            );
            assert_eq!(
                kernel.derived.get_col_mod(),
                derived_before,
                "recognition must NOT advance the derived col.mod watermark"
            );
            // The recognized vector still landed (orthogonal to the watermark):
            // a non-literal query surfaces the note through its OCR vector.
            let sem = kernel.search("carriers energy", 5).await.unwrap();
            assert!(
                !sem.is_empty(),
                "the OCR vector is searchable despite the unchanged watermark"
            );

            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn one_sweep_call_drains_a_multi_chunk_backlog_enumerating_once() {
        // A single recognize_pending(batch) call DRAINS the whole pending
        // backlog in internal bounded chunks — so the O(collection) image-ref
        // enumeration + done-set load is paid ONCE per drain, not once per
        // batch. With 5 pending images and batch=2 the drain runs 3 internal
        // chunks (2+2+1) but ONE enumeration; the call returns
        // recognized=5/remaining=0.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();

            // 5 notes, each referencing one distinct image.
            let mut media = std::collections::HashMap::new();
            let notes: Vec<serde_json::Value> = (0..5)
                .map(|i| {
                    let name = format!("img{i}.png");
                    media.insert(name.clone(), format!("recognized text {i}").into_bytes());
                    serde_json::json!({"note_type": "Basic", "deck": "Default",
                        "fields": {"Front": format!("see <img src=\"{name}\">"), "Back": "b"}})
                })
                .collect();
            upsert_wire(
                &kernel,
                serde_json::json!(notes).to_string(),
                "error".to_string(),
                false,
            )
            .await;

            kernel.attach_recognizer(Arc::new(StubRecognizer), Arc::new(MapResolver::new(media)));

            // ONE call, batch=2 < 5 pending → the internal drain processes all
            // five and reports nothing remaining.
            let report = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(
                report,
                recognize::SweepReport::Ran {
                    recognized: 5,
                    stored: 5,
                    remaining: 0,
                },
                "one call must drain the whole backlog (enumerate once), not just one batch"
            );
            // A second call is Idle — everything is done.
            assert_eq!(
                kernel.recognize_pending(2).await.unwrap(),
                recognize::SweepReport::Idle
            );

            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn recognition_skips_unreadable_media_and_keeps_it_pending() {
        // exists() says present but read() returns None (TOCTOU
        // delete, transient error) — the item must be SKIPPED, never
        // recognized over empty bytes, and must stay pending so a later
        // sweep (once the read heals) recognizes the real bytes.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();

            let notes = vec![serde_json::json!({"note_type": "Basic", "deck": "Default",
                "fields": {"Front":
                    "Two diagrams <img src=\"good.png\"> <img src=\"flaky.png\">",
                    "Back": "b"}})];
            upsert_wire(
                &kernel,
                serde_json::json!(notes).to_string(),
                "error".to_string(),
                false,
            )
            .await;

            let mut media = std::collections::HashMap::new();
            media.insert("good.png".to_string(), b"readable diagram text".to_vec());
            media.insert("flaky.png".to_string(), b"flaky secret payload".to_vec());
            let resolver = Arc::new(MapResolver::new(media));
            resolver
                .unreadable
                .lock()
                .unwrap()
                .insert("flaky.png".to_string());
            kernel.attach_recognizer(Arc::new(StubRecognizer), resolver.clone());

            // Sweep 1: only the readable item is sent and stored; the
            // unreadable one is skipped (not recognized as empty bytes) and
            // counted against the batch so the harness loop can't spin.
            let report = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(
                report,
                recognize::SweepReport::Ran {
                    recognized: 1,
                    stored: 1,
                    remaining: 0
                }
            );
            let hits = kernel.search("flaky secret", 5).await.unwrap();
            assert!(
                !hits
                    .iter()
                    .any(|h| h.signals.iter().any(|(sig, _)| sig == "exact")),
                "nothing stored lexically for the unreadable item"
            );

            // Sweep 2: still unreadable → still pending, still skipped — a
            // sweep ran (it was offered again), but nothing was sent.
            let again = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(
                again,
                recognize::SweepReport::Ran {
                    recognized: 0,
                    stored: 0,
                    remaining: 0
                }
            );

            // Heal the read: the next sweep recognizes the REAL bytes.
            resolver.unreadable.lock().unwrap().clear();
            let healed = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(
                healed,
                recognize::SweepReport::Ran {
                    recognized: 1,
                    stored: 1,
                    remaining: 0
                }
            );
            let hits = kernel.search("flaky secret", 5).await.unwrap();
            assert!(
                hits.iter()
                    .any(|h| h.signals.iter().any(|(sig, _)| sig == "exact")),
                "healed item recognized and stored"
            );

            // And now everything is done.
            let idle = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(idle, recognize::SweepReport::Idle);

            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn unreadable_prefix_reports_no_progress_for_the_driver() {
        // Livelock shape: total_pending > max_items with a permanently
        // unreadable PREFIX of the pending order. Skipped items stay pending,
        // so each call re-takes the same window — the kernel can't drain it.
        // The report must let a driver detect the no-progress batch
        // (recognized == 0 with remaining > 0) and stop instead of spinning.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();

            let notes = vec![serde_json::json!({"note_type": "Basic", "deck": "Default",
                "fields": {"Front":
                    "<img src=\"u1.png\"> <img src=\"u2.png\"> <img src=\"ok.png\">",
                    "Back": "b"}})];
            upsert_wire(
                &kernel,
                serde_json::json!(notes).to_string(),
                "error".to_string(),
                false,
            )
            .await;

            let mut media = std::collections::HashMap::new();
            media.insert("u1.png".to_string(), b"unreadable one".to_vec());
            media.insert("u2.png".to_string(), b"unreadable two".to_vec());
            media.insert("ok.png".to_string(), b"readable tail text".to_vec());
            let resolver = Arc::new(MapResolver::new(media));
            {
                let mut unreadable = resolver.unreadable.lock().unwrap();
                unreadable.insert("u1.png".to_string());
                unreadable.insert("u2.png".to_string());
            }
            kernel.attach_recognizer(Arc::new(StubRecognizer), resolver.clone());

            // The FIRST drain chunk covers only the unreadable prefix [u1,u2]:
            // nothing is sent, nothing stored, so the internal drain STOPS on
            // that no-progress chunk (the same halt the harness used to
            // apply between calls, now applied between chunks) and the readable
            // `ok.png` tail stays pending. The next call would re-take the
            // identical prefix, so the report signals no progress.
            let report = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(
                report,
                recognize::SweepReport::Ran {
                    recognized: 0,
                    stored: 0,
                    remaining: 1
                }
            );

            // And it really would: a second call is identical.
            let again = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(again, report);

            // Healed reads: ONE call now drains the WHOLE backlog — the
            // [u1,u2] chunk (now readable) AND the `ok.png` chunk — instead of
            // one batch per call. recognized=3, nothing remaining.
            resolver.unreadable.lock().unwrap().clear();
            let healed = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(
                healed,
                recognize::SweepReport::Ran {
                    recognized: 3,
                    stored: 3,
                    remaining: 0
                }
            );
            let idle = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(idle, recognize::SweepReport::Idle);

            kernel.close().await.unwrap();
        });
    }

    /// A [`StubRecognizer`] that counts every item it is handed and whose
    /// fingerprint a test can swap mid-flight (the OS/model-upgrade shape).
    struct CountingRecognizer {
        items_seen: std::sync::atomic::AtomicUsize,
        fingerprint: std::sync::Mutex<String>,
    }

    impl CountingRecognizer {
        fn new(fingerprint: &str) -> Self {
            Self {
                items_seen: std::sync::atomic::AtomicUsize::new(0),
                fingerprint: std::sync::Mutex::new(fingerprint.to_string()),
            }
        }
    }

    impl Recognizer for CountingRecognizer {
        fn recognize(
            &self,
            items: Vec<MediaItem>,
        ) -> BoxFuture<'_, NativeResult<Vec<Recognition>>> {
            self.items_seen
                .fetch_add(items.len(), std::sync::atomic::Ordering::SeqCst);
            Box::pin(async move {
                Ok(items
                    .iter()
                    .map(|b| Recognition {
                        text: String::from_utf8_lossy(&b.bytes).to_string(),
                        confidence: 0.95,
                        segments: Vec::new(),
                    })
                    .collect())
            })
        }

        fn fingerprint(&self) -> Option<String> {
            Some(self.fingerprint.lock().unwrap().clone())
        }
    }

    #[test]
    fn gated_items_are_judged_once_and_rederive_on_fingerprint_change() {
        // An item the gate drops gets a below-gate marker, so it is
        // recognized ONCE — not re-OCR'd every sweep — and only a recognizer
        // fingerprint change (engine upgrade) puts it back in the pending
        // set, exactly like stored rows re-derive.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();

            let notes = vec![serde_json::json!({"note_type": "Basic", "deck": "Default",
                "fields": {"Front":
                    "<img src=\"tiny.png\"> <img src=\"good.png\">", "Back": "b"}})];
            upsert_wire(
                &kernel,
                serde_json::json!(notes).to_string(),
                "error".to_string(),
                false,
            )
            .await;

            let mut media = std::collections::HashMap::new();
            // "ok" is 2 trimmed chars < min_chars_lexical → GateOutcome::Drop.
            media.insert("tiny.png".to_string(), b"ok".to_vec());
            media.insert(
                "good.png".to_string(),
                b"substantive recognized diagram text".to_vec(),
            );
            let recognizer = Arc::new(CountingRecognizer::new("stub:v1"));
            kernel.attach_recognizer(recognizer.clone(), Arc::new(MapResolver::new(media)));

            // Sweep 1: both items recognized, one stored, the gated one
            // marked done — nothing remains.
            let report = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(
                report,
                recognize::SweepReport::Ran {
                    recognized: 2,
                    stored: 1,
                    remaining: 0
                }
            );
            let seen = || {
                recognizer
                    .items_seen
                    .load(std::sync::atomic::Ordering::SeqCst)
            };
            assert_eq!(seen(), 2);

            // Sweep 2: idle — the gated item is DONE, not re-judged (the
            // recognizer is never called again).
            let again = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(again, recognize::SweepReport::Idle);
            assert_eq!(seen(), 2, "the gated item must not re-OCR");

            // Engine upgrade: a fingerprint change invalidates rows AND
            // markers, so BOTH items re-derive.
            *recognizer.fingerprint.lock().unwrap() = "stub:v2".to_string();
            let rederived = kernel.recognize_pending(10).await.unwrap();
            assert!(matches!(
                rederived,
                recognize::SweepReport::Ran {
                    recognized: 2,
                    stored: 1,
                    ..
                }
            ));
            assert_eq!(seen(), 4);

            // And the new outcome sticks: idle again, no further calls.
            let idle = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(idle, recognize::SweepReport::Idle);
            assert_eq!(seen(), 4);

            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn all_gated_window_converges_across_batches() {
        // The residual livelock shape: a permanently-gated prefix wider
        // than the batch window. Markers make each batch's gated items DONE,
        // so successive windows advance and the sweep converges to idle —
        // instead of re-taking the identical window forever (which the
        // no-progress stop, keyed on recognized == 0, deliberately does not
        // terminate).
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();

            let notes = vec![serde_json::json!({"note_type": "Basic", "deck": "Default",
                "fields": {"Front":
                    "<img src=\"t1.png\"> <img src=\"t2.png\"> <img src=\"t3.png\">",
                    "Back": "b"}})];
            upsert_wire(
                &kernel,
                serde_json::json!(notes).to_string(),
                "error".to_string(),
                false,
            )
            .await;

            let mut media = std::collections::HashMap::new();
            for name in ["t1.png", "t2.png", "t3.png"] {
                media.insert(name.to_string(), b"no".to_vec()); // all below-gate
            }
            let recognizer = Arc::new(CountingRecognizer::new("stub:v1"));
            kernel.attach_recognizer(recognizer.clone(), Arc::new(MapResolver::new(media)));

            // ONE drain: chunk [t1,t2] then chunk [t3] — each recognizes
            // (recognized > 0, so the drain does NOT stop on a no-progress halt)
            // but all gate out (stored=0). Markers make each item DONE, so the
            // chunks ADVANCE rather than re-taking the same window forever (the
            // gated-window convergence), and the whole gated backlog drains in this call.
            let first = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(
                first,
                recognize::SweepReport::Ran {
                    recognized: 3,
                    stored: 0,
                    remaining: 0
                }
            );

            // Convergence: idle, with each item judged exactly once (markers
            // make the gated items DONE — never re-recognized).
            let idle = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(idle, recognize::SweepReport::Idle);
            assert_eq!(
                recognizer
                    .items_seen
                    .load(std::sync::atomic::Ordering::SeqCst),
                3
            );

            kernel.close().await.unwrap();
        });
    }

    /// A recognizer that emits FIXED prose regardless of the input bytes,
    /// with a controllable fingerprint — the VLM-describe shape (the output
    /// is generated, not a transcription of visible text), so a lexical query
    /// for its words must NOT hit while a vector query (and provenance) does.
    struct ProseRecognizer {
        text: String,
        fingerprint: std::sync::Mutex<String>,
    }

    impl ProseRecognizer {
        fn new(text: &str, fp: &str) -> Self {
            Self {
                text: text.to_string(),
                fingerprint: std::sync::Mutex::new(fp.to_string()),
            }
        }
    }

    impl Recognizer for ProseRecognizer {
        fn recognize(
            &self,
            items: Vec<MediaItem>,
        ) -> BoxFuture<'_, NativeResult<Vec<Recognition>>> {
            let text = self.text.clone();
            Box::pin(async move {
                Ok(items
                    .iter()
                    .map(|_| Recognition {
                        text: text.clone(),
                        confidence: 0.95,
                        segments: Vec::new(),
                    })
                    .collect())
            })
        }

        fn fingerprint(&self) -> Option<String> {
            Some(self.fingerprint.lock().unwrap().clone())
        }
    }

    #[test]
    fn multi_engine_routing_keeps_describe_vector_only_and_ocr_unchanged() {
        // OCR and VLM-describe attach as INDEPENDENT purposes over one
        // image. OCR lands in source "ocr" (lexical + vector, bit-identical
        // to the single-slot sweep). Describe lands in source "vlm",
        // VECTOR-ONLY: a vector mints (a non-literal query surfaces the note)
        // but the describe prose is NEVER reachable via substring/fuzzy.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();

            let notes = vec![serde_json::json!({"note_type": "Basic", "deck": "Default",
                "fields": {"Front": "photo card <img src=\"photo.png\">", "Back": "b"}})];
            let results: Vec<serde_json::Value> = serde_json::from_str(
                &upsert_wire(
                    &kernel,
                    serde_json::json!(notes).to_string(),
                    "error".to_string(),
                    false,
                )
                .await,
            )
            .unwrap();
            let photo_id = results[0]["id"].as_i64().unwrap();

            let mut media = std::collections::HashMap::new();
            media.insert("photo.png".to_string(), b"opaque image bytes".to_vec());
            let resolver = Arc::new(MapResolver::new(media));

            // OCR: a recognizer reading visible text (echoes the bytes).
            kernel.attach_recognizer_with(
                recognize::RecognitionPurpose::Ocr,
                Arc::new(StubRecognizer),
                resolver.clone(),
            );
            // Describe: distinctive generated prose under the vlm source.
            kernel.attach_recognizer_with(
                recognize::RecognitionPurpose::Describe,
                Arc::new(ProseRecognizer::new(
                    "a photograph of a sunlit mountain valley with grazing cattle",
                    "describe:test:v1",
                )),
                resolver.clone(),
            );
            assert_eq!(
                kernel.attached_recognition_purposes(),
                vec![
                    recognize::RecognitionPurpose::Ocr,
                    recognize::RecognitionPurpose::Describe
                ]
            );

            // One sweep runs BOTH purposes over the one image: 2 recognized,
            // 2 stored (the aggregate report).
            let report = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(
                report,
                recognize::SweepReport::Ran {
                    recognized: 2,
                    stored: 2,
                    remaining: 0
                }
            );

            // OCR is unchanged: its visible text is lexically searchable.
            let ocr_hits = kernel.search("opaque image bytes", 5).await.unwrap();
            let ocr_hit = ocr_hits
                .iter()
                .find(|h| h.note_id == photo_id)
                .expect("ocr text searchable");
            assert!(ocr_hit.signals.iter().any(|(s, _)| s == "exact"));

            // The describe prose mints a VECTOR (a non-literal token-bag query
            // surfaces the note via the text space) ...
            let sem = kernel
                .search("valley cattle grazing mountain", 5)
                .await
                .unwrap();
            let sem_hit = sem
                .iter()
                .find(|h| h.note_id == photo_id)
                .expect("describe vector ranks");
            assert!(sem_hit.signals.iter().any(|(s, _)| s == "text"));

            // ... but is NEVER reachable via LEXICAL search: a literal phrase
            // that lives ONLY in the describe prose must not hit on exact or
            // fuzzy (the VectorOnly destination — docs/dev/decisions.md).
            let lex = kernel.search("sunlit mountain valley", 5).await.unwrap();
            let lex_hit = lex.iter().find(|h| h.note_id == photo_id);
            assert!(
                lex_hit.is_none_or(|h| h.signals.iter().all(|(s, _)| s != "exact" && s != "fuzzy")),
                "describe prose must be hidden from substring/fuzzy"
            );

            // Per-purpose fingerprint independence: bumping ONLY the describe
            // recognizer's fingerprint re-derives the vlm rows, never the ocr
            // ones. Attach a new describe engine with a changed fingerprint;
            // OCR's stored rows + idle state are untouched.
            kernel.attach_recognizer_with(
                recognize::RecognitionPurpose::Describe,
                Arc::new(ProseRecognizer::new(
                    "a different generated caption entirely",
                    "describe:test:v2",
                )),
                resolver.clone(),
            );
            // OCR alone is idle (its fingerprint didn't change).
            let ocr_idle = kernel
                .recognize_pending_for(recognize::RecognitionPurpose::Ocr, 10)
                .await
                .unwrap();
            assert_eq!(ocr_idle, recognize::SweepReport::Idle);
            // Describe re-derives exactly one row (the changed fingerprint
            // invalidated its prior vlm row, not the ocr one).
            let desc = kernel
                .recognize_pending_for(recognize::RecognitionPurpose::Describe, 10)
                .await
                .unwrap();
            assert_eq!(
                desc,
                recognize::SweepReport::Ran {
                    recognized: 1,
                    stored: 1,
                    remaining: 0
                }
            );
            // OCR text still searchable after the describe re-derive.
            let still = kernel.search("opaque image bytes", 5).await.unwrap();
            assert!(still.iter().any(|h| h.note_id == photo_id));

            kernel.close().await.unwrap();
        });
    }

    /// An ASR recognizer over audio bytes: transcribes the bytes and carries a
    /// single time-`Span` segment (the audio locator, vs OCR's bbox).
    struct AsrRecognizer;

    impl Recognizer for AsrRecognizer {
        fn recognize(
            &self,
            items: Vec<MediaItem>,
        ) -> BoxFuture<'_, NativeResult<Vec<Recognition>>> {
            Box::pin(async move {
                Ok(items
                    .iter()
                    .map(|m| {
                        let text = String::from_utf8_lossy(&m.bytes).to_string();
                        Recognition {
                            text: text.clone(),
                            confidence: 0.9,
                            segments: vec![Segment {
                                text,
                                confidence: 0.9,
                                // A time range, not a bbox — the audio locator.
                                locator: Some(Locator::Span([0.0, 2.5])),
                            }],
                        }
                    })
                    .collect())
            })
        }

        fn fingerprint(&self) -> Option<String> {
            Some("asr:stub:v1".to_string())
        }
    }

    #[test]
    fn asr_sweep_enumerates_sound_refs_and_mints_lexical_and_vector() {
        // The AUDIO path end-to-end. A note referencing [sound:…]
        // audio is enumerated by note_sound_refs (the audio twin), the ASR
        // purpose recognizes it (source "asr", LexicalAndVector like OCR), and
        // both consumers light up: the transcript is lexically searchable AND
        // mints a text-space vector. Span segments persist. OCR is untouched.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();

            // One audio note, one image note — to prove the AUDIO purpose reads
            // ONLY [sound:] refs (never the image), and vice versa.
            let notes = vec![
                serde_json::json!({"note_type": "Basic", "deck": "Default",
                    "fields": {"Front": "Listen [sound:lecture.mp3]", "Back": "b"}}),
                serde_json::json!({"note_type": "Basic", "deck": "Default",
                    "fields": {"Front": "zzz filler card qqq", "Back": "b"}}),
            ];
            let results: Vec<serde_json::Value> = serde_json::from_str(
                &upsert_wire(
                    &kernel,
                    serde_json::json!(notes).to_string(),
                    "error".to_string(),
                    false,
                )
                .await,
            )
            .unwrap();
            let audio_id = results[0]["id"].as_i64().unwrap();

            let mut media = std::collections::HashMap::new();
            media.insert(
                "lecture.mp3".to_string(),
                b"mitochondria are the powerhouse of the cell".to_vec(),
            );
            let resolver = Arc::new(MapResolver::new(media));

            kernel.attach_recognizer_with(
                recognize::RecognitionPurpose::Asr,
                Arc::new(AsrRecognizer),
                resolver.clone(),
            );
            assert_eq!(
                kernel.attached_recognition_purposes(),
                vec![recognize::RecognitionPurpose::Asr]
            );

            // The sweep enumerates the [sound:] ref and transcribes it.
            let report = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(
                report,
                recognize::SweepReport::Ran {
                    recognized: 1,
                    stored: 1,
                    remaining: 0
                }
            );

            // LexicalAndVector — both consumers: the transcript is lexically
            // searchable (exact) ...
            let lex = kernel.search("powerhouse of the cell", 5).await.unwrap();
            let lex_hit = lex
                .iter()
                .find(|h| h.note_id == audio_id)
                .expect("asr transcript lexically searchable");
            assert!(lex_hit.signals.iter().any(|(s, _)| s == "exact"));

            // ... AND mints a vector (a non-literal token-bag query surfaces it
            // via the text space).
            let sem = kernel
                .search("cell powerhouse mitochondria", 5)
                .await
                .unwrap();
            assert!(
                sem.iter()
                    .find(|h| h.note_id == audio_id)
                    .is_some_and(|h| h.signals.iter().any(|(s, _)| s == "text")),
                "asr transcript mints a text vector"
            );

            // A second sweep is idle — the item is DONE (not re-transcribed).
            let idle = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(idle, recognize::SweepReport::Idle);

            kernel.close().await.unwrap();
        });
    }

    /// A recognizer that records the PEAK number of `recognize()` calls
    /// in-flight at once — the probe for the slow-recognition concurrency bound.
    ///
    /// Each call rendezvous on a barrier sized to the bound: it blocks until
    /// exactly `SLOW_RECOGNITION_CONCURRENCY` calls are in flight together, then
    /// all release. This makes the observed overlap DETERMINISTIC rather than a
    /// bet that one fixed sleep window outlasts io-thread scheduling delay — the
    /// permitted batch is held open until the full batch arrives, so peak reaches
    /// the bound on every run. If a regression dropped the limit BELOW the
    /// barrier width, the batch could never fill and the wait would block; the
    /// timeout turns that into a clean test failure instead of a hang.
    struct ConcurrencyProbeRecognizer {
        in_flight: Arc<std::sync::atomic::AtomicUsize>,
        peak: Arc<std::sync::atomic::AtomicUsize>,
        rendezvous: Arc<tokio::sync::Barrier>,
    }

    impl Recognizer for ConcurrencyProbeRecognizer {
        fn recognize(
            &self,
            items: Vec<MediaItem>,
        ) -> BoxFuture<'_, NativeResult<Vec<Recognition>>> {
            let in_flight = Arc::clone(&self.in_flight);
            let peak = Arc::clone(&self.peak);
            let rendezvous = Arc::clone(&self.rendezvous);
            Box::pin(async move {
                let now = in_flight.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                peak.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
                // Hold the permitted batch open until the whole batch is present:
                // the barrier RELEASE is the structural proof that exactly the
                // bound's worth overlapped. No timeout — if the limit regressed
                // below the barrier width the batch can't fill and the test hangs
                // (Bazel's per-test timeout catches that), never a flaky window.
                rendezvous.wait().await;
                in_flight.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                Ok(items
                    .iter()
                    .map(|m| Recognition {
                        text: String::from_utf8_lossy(&m.bytes).to_string(),
                        confidence: 0.9,
                        segments: Vec::new(),
                    })
                    .collect())
            })
        }

        fn fingerprint(&self) -> Option<String> {
            Some("probe:v1".to_string())
        }
    }

    #[test]
    fn slow_recognition_concurrency_is_bounded() {
        // A long-running purpose's recognize() holds a concurrency
        // permit, so no more than SLOW_RECOGNITION_CONCURRENCY run at once even
        // when several single-item sweeps are driven CONCURRENTLY. (One audio
        // note per item; max_items=1 makes each concurrent sweep one
        // recognize() dispatch.) The probe records the peak overlap.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();

            // Several audio notes so concurrent single-item sweeps pick
            // distinct items (the pending diff hands each a different ref). A
            // whole multiple of the bound, so every permitted batch exactly
            // fills the rendezvous barrier (no half-batch left waiting).
            let n = SLOW_RECOGNITION_CONCURRENCY * 3;
            let mut media = std::collections::HashMap::new();
            let notes: Vec<serde_json::Value> = (0..n)
                .map(|i| {
                    let name = format!("clip{i}.mp3");
                    media.insert(
                        name.clone(),
                        format!("transcript number {i} here").into_bytes(),
                    );
                    serde_json::json!({"note_type": "Basic", "deck": "Default",
                        "fields": {"Front": format!("Listen [sound:{name}]"), "Back": "b"}})
                })
                .collect();
            upsert_wire(
                &kernel,
                serde_json::json!(notes).to_string(),
                "error".to_string(),
                false,
            )
            .await;

            let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            kernel.attach_recognizer_with(
                recognize::RecognitionPurpose::Asr,
                Arc::new(ConcurrencyProbeRecognizer {
                    in_flight: Arc::clone(&in_flight),
                    peak: Arc::clone(&peak),
                    rendezvous: Arc::new(tokio::sync::Barrier::new(SLOW_RECOGNITION_CONCURRENCY)),
                }),
                Arc::new(MapResolver::new(media)),
            );

            // Drive `n` concurrent single-item ASR sweeps. Without the bound,
            // peak would reach up to `n`; with it, peak ≤ the ceiling.
            let kernel = Arc::new(kernel);
            let mut handles = Vec::new();
            for _ in 0..n {
                let k = Arc::clone(&kernel);
                handles.push(tokio::spawn(async move {
                    k.recognize_pending_for(recognize::RecognitionPurpose::Asr, 1)
                        .await
                }));
            }
            for h in handles {
                h.await.unwrap().unwrap();
            }

            // The rendezvous holds each permitted batch open until it is full, so
            // the peak is exactly the bound: a regression to a HIGHER limit lets
            // more than the bound overlap (peak > bound), and a regression to a
            // LOWER limit can't fill the barrier (the timeout fires, peak < bound).
            let observed_peak = peak.load(std::sync::atomic::Ordering::SeqCst);
            assert_eq!(
                observed_peak, SLOW_RECOGNITION_CONCURRENCY,
                "slow-recognition concurrency must be bounded at exactly \
                 SLOW_RECOGNITION_CONCURRENCY={SLOW_RECOGNITION_CONCURRENCY}, saw peak \
                 {observed_peak}"
            );

            Arc::try_unwrap(kernel)
                .unwrap_or_else(|_| panic!("kernel still shared"))
                .close()
                .await
                .unwrap();
        });
    }

    /// A describe recognizer that returns the EMPTY recognition (`"", 0.0`) for
    /// any item whose bytes contain a sentinel marker — the per-item PERMANENT-
    /// failure shape (a 4xx the engine converts to the empty recognition) — and
    /// real prose for any other. (MediaItem carries no name, so the stub keys
    /// on the bytes, exactly as a real endpoint keys on the response for THAT
    /// image.) Counts items seen so a test can prove the failed item is NOT
    /// re-offered. Controllable fingerprint (the engine-upgrade retry).
    struct PerItemFailingDescribe {
        fail_marker: Vec<u8>,
        prose: String,
        fingerprint: std::sync::Mutex<String>,
        items_seen: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Recognizer for PerItemFailingDescribe {
        fn recognize(
            &self,
            items: Vec<MediaItem>,
        ) -> BoxFuture<'_, NativeResult<Vec<Recognition>>> {
            self.items_seen
                .fetch_add(items.len(), std::sync::atomic::Ordering::SeqCst);
            let fail = self.fail_marker.clone();
            let prose = self.prose.clone();
            Box::pin(async move {
                Ok(items
                    .iter()
                    .map(|m| {
                        // The condemned item yields the empty recognition
                        // exactly like a 4xx in shrike-describe-remote.
                        if m.bytes.windows(fail.len()).any(|w| w == fail.as_slice()) {
                            Recognition {
                                text: String::new(),
                                confidence: 0.0,
                                segments: Vec::new(),
                            }
                        } else {
                            Recognition {
                                text: prose.clone(),
                                confidence: 1.0,
                                segments: Vec::new(),
                            }
                        }
                    })
                    .collect())
            })
        }

        fn fingerprint(&self) -> Option<String> {
            Some(self.fingerprint.lock().unwrap().clone())
        }
    }

    #[test]
    fn per_item_permanent_failure_is_negative_cached_and_retried_on_fingerprint_change() {
        // A per-item PERMANENT failure (a 4xx the engine turns
        // into the empty recognition) is judged ONCE — gated under the vlm
        // source so it is not re-offered every sweep (expensive against a paid
        // endpoint) — and re-tried only after a describe fingerprint change
        // (clear_gated(vlm)), exactly like a stored row re-derives. Pins the
        // 4xx negative-cache path explicitly (the existing gated test covers
        // only the below-substance drop).
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();

            // One note, two images: one the endpoint 4xx-rejects, one it
            // describes fine — so the failed item and a healthy item share a
            // sweep (proving the failure doesn't sink the batch).
            let notes = vec![serde_json::json!({"note_type": "Basic", "deck": "Default",
                "fields": {"Front":
                    "<img src=\"bad.png\"> <img src=\"good.png\">", "Back": "b"}})];
            upsert_wire(
                &kernel,
                serde_json::json!(notes).to_string(),
                "error".to_string(),
                false,
            )
            .await;

            let mut media = std::collections::HashMap::new();
            // The condemned image's bytes carry the "oversized" sentinel; the
            // healthy one does not.
            media.insert("bad.png".to_string(), b"opaque oversized bytes".to_vec());
            media.insert("good.png".to_string(), b"opaque ok bytes".to_vec());
            let items_seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let recognizer = Arc::new(PerItemFailingDescribe {
                fail_marker: b"oversized".to_vec(),
                prose: "a clear photograph of rolling green hills under a blue sky".to_string(),
                fingerprint: std::sync::Mutex::new("describe:v1".to_string()),
                items_seen: Arc::clone(&items_seen),
            });
            kernel.attach_recognizer_with(
                recognize::RecognitionPurpose::Describe,
                recognizer.clone(),
                Arc::new(MapResolver::new(media)),
            );
            let seen = || items_seen.load(std::sync::atomic::Ordering::SeqCst);

            // Sweep 1: both images recognized; the good one stored, the failed
            // one gated (Drop) — nothing remains.
            let report = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(
                report,
                recognize::SweepReport::Ran {
                    recognized: 2,
                    stored: 1,
                    remaining: 0
                }
            );
            assert_eq!(seen(), 2);

            // Sweep 2: idle — the 4xx'd item is DONE (negative-cached), NOT
            // re-offered. The recognizer is never called again.
            let again = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(again, recognize::SweepReport::Idle);
            assert_eq!(
                seen(),
                2,
                "the 4xx'd item must not be re-offered each sweep"
            );

            // The describe prose still mints a VectorOnly vector (the healthy
            // item) — the failure didn't regress describe routing.
            let sem = kernel
                .search("rolling green hills blue sky", 5)
                .await
                .unwrap();
            assert!(
                sem.iter()
                    .any(|h| h.signals.iter().any(|(s, _)| s == "text")),
                "the good describe vector is searchable"
            );

            // Engine upgrade: a fingerprint change clears the vlm markers, so
            // the previously-failed item RE-ENTERS the pending set (the new
            // engine may handle what the old rejected) — both re-offered.
            *recognizer.fingerprint.lock().unwrap() = "describe:v2".to_string();
            let retried = kernel.recognize_pending(10).await.unwrap();
            assert!(
                matches!(retried, recognize::SweepReport::Ran { recognized: 2, .. }),
                "a fingerprint change re-tries the negative-cached item: {retried:?}"
            );
            assert_eq!(
                seen(),
                4,
                "both items re-offered after the fingerprint change"
            );

            kernel.close().await.unwrap();
        });
    }

    /// A describe recognizer whose CHUNK fails (an endpoint/transport error) —
    /// the load-bearing contrast with the per-item failure: the whole chunk
    /// Err's, so nothing is persisted or gated and the backlog stays pending.
    struct ChunkErroringDescribe;

    impl Recognizer for ChunkErroringDescribe {
        fn recognize(
            &self,
            _items: Vec<MediaItem>,
        ) -> BoxFuture<'_, NativeResult<Vec<Recognition>>> {
            Box::pin(async move {
                Err(NativeError::unavailable(
                    "describe request failed: connection refused".to_string(),
                ))
            })
        }

        fn fingerprint(&self) -> Option<String> {
            Some("describe:down:v1".to_string())
        }
    }

    #[test]
    fn endpoint_level_failure_does_not_negative_cache_and_leaves_backlog_pending() {
        // An ENDPOINT-level failure (transport/auth/
        // exhausted retries) must NOT gate the item — it Err's the chunk before
        // any persist or fingerprint advance, so the backlog stays pending and
        // a later sweep (once the endpoint is up) retries. The negative cache
        // is for per-item PERMANENT failures only.
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            kernel.reindex_if_needed().await.unwrap();

            let notes = vec![serde_json::json!({"note_type": "Basic", "deck": "Default",
                "fields": {"Front": "<img src=\"photo.png\">", "Back": "b"}})];
            upsert_wire(
                &kernel,
                serde_json::json!(notes).to_string(),
                "error".to_string(),
                false,
            )
            .await;

            let mut media = std::collections::HashMap::new();
            media.insert("photo.png".to_string(), b"opaque bytes".to_vec());
            kernel.attach_recognizer_with(
                recognize::RecognitionPurpose::Describe,
                Arc::new(ChunkErroringDescribe),
                Arc::new(MapResolver::new(media)),
            );

            // The sweep aborts with the chunk Err — nothing gated, nothing
            // stored. The describe fingerprint meta is NOT advanced.
            let err = kernel.recognize_pending(10).await.unwrap_err();
            assert!(
                err.to_string().contains("describe request failed"),
                "the endpoint error propagates: {err}"
            );

            // Swap in a working engine with the SAME fingerprint (the endpoint
            // came back up): the item is STILL pending — it was never gated —
            // so it recognizes now. A negative cache would have wrongly skipped
            // it.
            kernel.attach_recognizer_with(
                recognize::RecognitionPurpose::Describe,
                Arc::new(ProseRecognizer::new(
                    "a recovered description of the photo",
                    "describe:down:v1",
                )),
                Arc::new(MapResolver::new({
                    let mut m = std::collections::HashMap::new();
                    m.insert("photo.png".to_string(), b"opaque bytes".to_vec());
                    m
                })),
            );
            let report = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(
                report,
                recognize::SweepReport::Ran {
                    recognized: 1,
                    stored: 1,
                    remaining: 0
                },
                "the un-gated item was still pending and recognized on recovery"
            );

            kernel.close().await.unwrap();
        });
    }

    #[test]
    fn open_upsert_search_close_without_python() {
        // The harness picks the runtime: here futures' minimal block_on —
        // no tokio, nothing owned by the kernel.
        crate::runtime::testing::run_with_sync(assert_send(smoke()));
    }

    async fn smoke() {
        let dir = temp_dir();
        let col = dir.join("collection.anki2");
        let cache = dir.join("cache");
        // The harness assembles the scheduling: here the thread-free
        let kernel = Kernel::open(col.to_str().unwrap(), cache.to_str().unwrap())
            .await
            .unwrap();
        // The harness attaches the embedding service (the registry slot).
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
        kernel.settle().await;
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
        assert_eq!(
            kernel.delete_notes(vec![mito]).await.unwrap().deleted,
            vec![mito]
        );
        kernel.settle().await;
        let after = kernel.search("mitochondria powerhouse", 5).await.unwrap();
        assert!(after.iter().all(|h| h.note_id != mito));

        // Restart: close() drains the ingest queue and flushes the index savers
        // durably (the shutdown ordering), so the on-disk index already reflects
        // every write — the fresh kernel finds NO drift and serves the persisted
        // vectors directly.
        kernel.close().await.unwrap();
        let kernel2 = Kernel::open(col.to_str().unwrap(), cache.to_str().unwrap())
            .await
            .unwrap();
        kernel2.attach_embedder(Arc::new(HashEmbedder), None);
        assert!(!kernel2.reindex_if_needed().await.unwrap()); // durable close → already current
        let hits = kernel2.search("newton laws of motion", 5).await.unwrap();
        assert!(!hits.is_empty());
        kernel2.close().await.unwrap();
    }
    /// The derived rebuild's collect→commit window vs concurrent
    /// writes. The first half DEMONSTRATES the hazard mechanically (a build
    /// over a stale snapshot erases a newer note's derived rows — why the
    /// verify exists); the second half pins that `rebuild_derived` converges
    /// (post-commit col_mod equals the committed snapshot) and restores them.
    #[test]
    fn derived_rebuild_verifies_against_mid_build_writes() {
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            let basic = kernel.notetype_id("Basic").await.unwrap();
            kernel
                .upsert_note(
                    basic,
                    1,
                    vec!["the krebs cycle".into(), "a".into()],
                    vec![],
                    DuplicatePolicy::Error,
                )
                .await
                .unwrap();
            let (_, settled) = kernel.rebuild_derived().await.unwrap();
            assert_eq!(kernel.col_mod().await.unwrap(), settled);

            // Snapshot rows NOW (note A only), then land note B…
            let stale = kernel
                .collection
                .run(|core| -> shrike_error::NativeResult<_> {
                    let ids = core.find_notes("")?;
                    let rows = core.derived_field_rows(&ids)?;
                    let dmod = core.col_mod()?;
                    Ok((rows, dmod))
                })
                .await
                .unwrap()
                .unwrap();
            kernel
                .upsert_note(
                    basic,
                    1,
                    vec!["oxaloacetate condenses".into(), "b".into()],
                    vec![],
                    DuplicatePolicy::Error,
                )
                .await
                .unwrap();
            kernel.settle().await;
            assert!(
                !kernel
                    .derived
                    .search_substring("oxaloacetate", 5, None, &[])
                    .unwrap()
                    .unwrap()
                    .is_empty(),
                "B's ingest landed"
            );
            // …and commit the stale build: B's FIELD rows are erased (the
            // hazard). The stale snapshot's own note ids are its live set — this
            // is the field-row erase the converge loop heals, distinct from the
            // recognition-row prune that keys off the live collection.
            let stale_live: Vec<i64> = stale.0.iter().map(|r| r.0).collect();
            kernel
                .derived
                .build(&stale.0, &stale_live, stale.1)
                .unwrap();
            assert!(
                kernel
                    .derived
                    .search_substring("oxaloacetate", 5, None, &[])
                    .unwrap()
                    .unwrap()
                    .is_empty(),
                "a stale-snapshot commit erases newer rows — the #471 hazard"
            );

            // The public op converges: verify catches movement, re-collects,
            // and the final watermark equals the live col_mod.
            let (_, dmod) = kernel.rebuild_derived().await.unwrap();
            assert_eq!(kernel.col_mod().await.unwrap(), dmod);
            assert!(
                !kernel
                    .derived
                    .search_substring("oxaloacetate", 5, None, &[])
                    .unwrap()
                    .unwrap()
                    .is_empty(),
                "the rebuild restored B's rows"
            );
            kernel.close().await.unwrap();
        });
    }

    // ── watermark over-certification + best-effort tail ──────────

    /// An embedder that, on the FIRST text containing "bravo", PARKS (so a
    /// concurrent op can interleave while it is in flight) and then FAILS that
    /// embed when released; EVERY LATER bravo embed succeeds (so a heal path —
    /// `reindex_if_needed` — can reconcile bravo in). All non-bravo texts embed
    /// normally. The probe of the watermark guard + best-effort tail.
    struct GatedEmbedder {
        gate: tokio::sync::Notify,
        parked: tokio::sync::Notify,
        bravo_seen: std::sync::atomic::AtomicBool,
        first_bravo_done: std::sync::atomic::AtomicBool,
    }
    impl GatedEmbedder {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                gate: tokio::sync::Notify::new(),
                parked: tokio::sync::Notify::new(),
                bravo_seen: std::sync::atomic::AtomicBool::new(false),
                first_bravo_done: std::sync::atomic::AtomicBool::new(false),
            })
        }
    }
    impl Embedder for GatedEmbedder {
        fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            Box::pin(async move {
                use std::sync::atomic::Ordering::SeqCst;
                let is_bravo = texts.iter().any(|t| t.contains("bravo"));
                // Only the FIRST bravo embed parks+fails; later ones succeed so
                // the heal path can reconcile. `swap` makes the gate single-shot.
                if is_bravo && !self.first_bravo_done.swap(true, SeqCst) {
                    self.bravo_seen.store(true, SeqCst);
                    self.parked.notify_one();
                    self.gate.notified().await;
                    return Err(NativeError::internal("bravo embed deliberately failed"));
                }
                Ok(HashEmbedder::embed_sync(&texts))
            })
        }
        fn fingerprint(&self) -> Option<String> {
            Some("gated-embedder:v1".to_string())
        }
        fn dim(&self) -> Option<usize> {
            Some(64)
        }
    }

    /// The keystone. A concurrent op B writes "bravo" and parks
    /// in embed (in flight, its index tail not yet run); op A writes "alpha" and
    /// runs its full tail. With a naive advance, op A's `advance_watermarks` read
    /// the LIVE `col.mod` (already reflecting bravo) and stamped it →
    /// `check_drift` saw no drift → bravo was PERMANENTLY missing from the index.
    /// Fixed: op A may not certify a `col.mod` covering B's still-in-flight write,
    /// so the watermark stays behind and drift heals bravo.
    #[test]
    fn s5_interleaved_upsert_does_not_falsely_advance_the_index_watermark() {
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Arc::new(
                Kernel::open(
                    dir.join("c.anki2").to_str().unwrap(),
                    dir.join("cache").to_str().unwrap(),
                )
                .await
                .unwrap(),
            );
            let embedder = GatedEmbedder::new();
            kernel.attach_embedder(embedder.clone(), None);
            kernel.reindex_if_needed().await.unwrap();
            let basic = kernel.notetype_id("Basic").await.unwrap();

            // Op B: writes "bravo", then PARKS in embed (in flight). Its
            // collection write completes before it parks, so col.mod already
            // reflects bravo.
            let kb = Arc::clone(&kernel);
            let op_b = crate::spawn_op(async move {
                kb.upsert_note(
                    basic,
                    1,
                    vec!["bravo bravo bravo".into(), "b".into()],
                    vec![],
                    DuplicatePolicy::Error,
                )
                .await
                .map(|_| ())
            });
            embedder.parked.notified().await;
            assert!(embedder
                .bravo_seen
                .load(std::sync::atomic::Ordering::SeqCst));

            // Op A: writes "alpha" + runs its full index tail.
            let alpha = kernel
                .upsert_note(
                    basic,
                    1,
                    vec!["alpha alpha alpha".into(), "a".into()],
                    vec![],
                    DuplicatePolicy::Error,
                )
                .await
                .unwrap();
            let CreateOutcome::Created(alpha_id) = alpha else {
                panic!("alpha create")
            };

            embedder.gate.notify_one(); // release op B → it ERRORS, bravo never indexed
            let _ = op_b.await;

            // Settle to the deterministic barrier before reading index state: op
            // A's index tail (embed → index add) runs asynchronously after its
            // upsert returns, so without this the "alpha indexed" sanity check
            // races the tail and spuriously fails when the runtime is starved
            // (#880). settle() returns once every enqueued tail — A's success and
            // B's failure — is fully processed.
            kernel.settle().await;

            let all_ids = kernel
                .collection()
                .run(|core| core.find_notes(""))
                .await
                .unwrap()
                .unwrap();
            let bravo_id = *all_ids.iter().find(|id| **id != alpha_id).unwrap();
            let primary = kernel.index_set().primary();
            let col_mod = kernel.col_mod().await.unwrap();

            assert!(
                primary.engine().contains(alpha_id),
                "alpha indexed (sanity)"
            );
            assert!(
                !primary.engine().contains(bravo_id),
                "PRECONDITION: bravo's embed failed, not indexed"
            );
            assert_ne!(
                primary.col_mod(),
                Some(col_mod),
                "FIX (#585): the watermark was NOT advanced to the live col.mod, \
                 so drift stays armed"
            );
            let drift_will_heal = kernel.reindex_if_needed().await.unwrap();
            assert!(
                drift_will_heal,
                "FIX (#585): watermark < col.mod, reindex sees drift and reconciles"
            );
            assert!(
                primary.engine().contains(bravo_id),
                "FIX (#585): the drift reconcile healed bravo into the index"
            );

            kernel.close().await.unwrap();
        });
    }

    /// The derived/FTS5 twin. The SAME interleave over the lexical
    /// surface: while op B is in flight (parked in embed, its derived ingest not
    /// yet run, so bravo is NOT in FTS5), op A completes its full tail. With a
    /// naive advance, op A's `advance_watermarks` advanced the DERIVED watermark
    /// to the live col.mod too (an UNCONDITIONAL `self.derived.set_col_mod` that
    /// fired even with no embedder) → `rebuild_derived`'s drift
    /// gate went quiet → bravo invisible to substring/fuzzy forever. Fixed: op A
    /// may not certify a derived watermark covering B's still-in-flight
    /// (un-ingested) write, so the derived watermark stays behind and the drift
    /// gate remains armed. Assert at the parked moment — that IS the
    /// over-certification point.
    #[test]
    fn s6_2_interleaved_upsert_does_not_falsely_advance_the_derived_watermark() {
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Arc::new(
                Kernel::open(
                    dir.join("c.anki2").to_str().unwrap(),
                    dir.join("cache").to_str().unwrap(),
                )
                .await
                .unwrap(),
            );
            let embedder = GatedEmbedder::new();
            kernel.attach_embedder(embedder.clone(), None);
            kernel.reindex_if_needed().await.unwrap();
            // Build the derived store so its watermark exists and tracks col.mod.
            kernel.rebuild_derived().await.unwrap();
            let basic = kernel.notetype_id("Basic").await.unwrap();

            // Op B writes "bravo" and PARKS in embed — its derived ingest has
            // NOT run, so bravo is not in FTS5 and its derived watermark token
            // is still in flight.
            let kb = Arc::clone(&kernel);
            let op_b = crate::spawn_op(async move {
                kb.upsert_note(
                    basic,
                    1,
                    vec!["bravo bravo bravo".into(), "b".into()],
                    vec![],
                    DuplicatePolicy::Error,
                )
                .await
                .map(|_| ())
            });
            embedder.parked.notified().await;

            // Op A writes "alpha" and runs its FULL tail (its derived ingest
            // succeeds). Its derived-watermark advance must be BLOCKED by op B's
            // still-in-flight derived token.
            let alpha = kernel
                .upsert_note(
                    basic,
                    1,
                    vec!["alpha alpha alpha".into(), "a".into()],
                    vec![],
                    DuplicatePolicy::Error,
                )
                .await
                .unwrap();
            let CreateOutcome::Created(_alpha_id) = alpha else {
                panic!("alpha create")
            };

            // While op B is still parked: the live col.mod reflects both writes,
            // but bravo is NOT in FTS5 (B parked before ingest).
            let col_mod = kernel.col_mod().await.unwrap();
            assert!(
                kernel
                    .derived
                    .search_substring("bravo", 5, None, &[])
                    .unwrap()
                    .unwrap()
                    .is_empty(),
                "PRECONDITION: bravo not yet ingested into FTS5 (op B parked)"
            );
            assert_ne!(
                kernel.derived.get_col_mod(),
                Some(col_mod),
                "FIX (#585/S6-2): op A did NOT advance the derived watermark to \
                 the live col.mod over op B's un-ingested write → the derived \
                 drift gate stays armed (pre-#585 it equalled col_mod → silent loss)"
            );

            // Release op B; its derived ingest now lands bravo. The watermark
            // catches up legitimately, and a rebuild keeps both searchable.
            embedder.gate.notify_one();
            let _ = op_b.await;
            kernel.rebuild_derived().await.unwrap();
            assert!(
                !kernel
                    .derived
                    .search_substring("bravo", 5, None, &[])
                    .unwrap()
                    .unwrap()
                    .is_empty(),
                "bravo is searchable after op B's ingest + the rebuild heal"
            );

            kernel.close().await.unwrap();
        });
    }

    /// The in-crate companion to `tests/s6_repro.rs`: a single
    /// upsert whose embed tail fails returns Ok with the per-item result (the
    /// committed note), AND leaves the index watermark behind so drift heals it.
    #[test]
    fn s6_best_effort_tail_returns_results_and_leaves_the_watermark_behind() {
        crate::runtime::testing::run_with_sync(async {
            let dir = temp_dir();
            let kernel = Kernel::open(
                dir.join("c.anki2").to_str().unwrap(),
                dir.join("cache").to_str().unwrap(),
            )
            .await
            .unwrap();
            let embedder = GatedEmbedder::new();
            kernel.attach_embedder(embedder.clone(), None);
            kernel.reindex_if_needed().await.unwrap();
            let basic = kernel.notetype_id("Basic").await.unwrap();

            // Release the gate immediately so the bravo embed fails without
            // parking (no concurrency needed for the best-effort assertion).
            embedder.gate.notify_one();
            let outcome = kernel
                .upsert_note(
                    basic,
                    1,
                    vec!["bravo bravo bravo".into(), "b".into()],
                    vec![],
                    DuplicatePolicy::Error,
                )
                .await;

            // The committed write returns Ok despite the failed
            // embed (a prior version `?`-propagated → Err).
            let CreateOutcome::Created(bravo_id) =
                outcome.expect("committed write returns Ok despite a failed embed tail")
            else {
                panic!("bravo create")
            };
            let primary = kernel.index_set().primary();
            let col_mod = kernel.col_mod().await.unwrap();
            assert!(
                !primary.engine().contains(bravo_id),
                "the embed failed, so no vector was written"
            );
            assert_ne!(
                primary.col_mod(),
                Some(col_mod),
                "FIX (#585/#590): a failed tail leaves the watermark behind"
            );
            // The drift gate stays armed (watermark < col.mod) — a future boot
            // with a working embedder would reconcile bravo in. We assert the
            // gate directly rather than driving the reconcile, since this gated
            // embedder would just fail bravo again.
            let model_id = embedder.fingerprint();
            assert!(
                primary.check_drift(col_mod, model_id.as_deref(), false),
                "FIX (#585/#590): drift is still armed — never falsely certified"
            );

            kernel.close().await.unwrap();
        });
    }

    /// A `rebuild_derived` overlapping concurrent searches must not deadlock.
    ///
    /// The streaming rebuild pulls field-row chunks through the collection actor
    /// (the single drive_sync thread); a concurrent `search` runs `search_notes`
    /// through that SAME actor and takes the derived connection lock
    /// (search_substring/search_fuzzy). If the rebuild held that lock across its
    /// chunk pulls, the search would wedge the actor on the lock while the
    /// rebuild waited on the actor for its next chunk — a circular wait that
    /// hangs the whole kernel. The rebuild drops the lock around every pull, so
    /// the two interleave.
    ///
    /// The deadlock guard is STRUCTURAL: the test awaits the real completion of
    /// BOTH the rebuild and the search loop (`futures::join!`). A regression that
    /// wedges them never completes the join, so the test hangs and Bazel's
    /// per-test timeout fails it — no in-test wall-clock watchdog whose budget
    /// could trip on a slow-but-live run (and whose `process::abort` would kill
    /// the whole 143-test binary).
    #[test]
    fn rebuild_derived_does_not_deadlock_against_concurrent_search() {
        crate::runtime::testing::run_with_sync(async move {
            let dir = temp_dir();
            let kernel = Arc::new(
                Kernel::open(
                    dir.join("c.anki2").to_str().unwrap(),
                    dir.join("cache").to_str().unwrap(),
                )
                .await
                .unwrap(),
            );
            kernel.attach_embedder(Arc::new(HashEmbedder), None);
            let basic = kernel.notetype_id("Basic").await.unwrap();
            let n = index_orchestrator::STREAM_CHUNK * 3;
            let notes: Vec<NoteSpec> = (0..n)
                .map(|i| NoteSpec {
                    notetype_id: basic,
                    deck_id: 1,
                    fields: vec![format!("note body {i} mitochondria"), format!("back {i}")],
                    tags: vec![],
                })
                .collect();
            kernel
                .upsert_notes(notes, DuplicatePolicy::Error)
                .await
                .unwrap();
            // Settle so the field rows are durably in the derived store BEFORE
            // the concurrent rebuild — the searches below assert those rows stay
            // findable THROUGHOUT the rebuild (no full-recall cliff), so they
            // must be present to begin with.
            kernel.settle().await;

            let searcher = Arc::clone(&kernel);
            let search_loop = async move {
                let mut min_hits = usize::MAX;
                for _ in 0..200 {
                    let hits = searcher.search("mitochondria", 5).await.unwrap();
                    min_hits = min_hits.min(hits.len());
                }
                min_hits
            };
            let (rebuilt, min_hits) = futures::join!(kernel.rebuild_derived(), search_loop);
            rebuilt.unwrap();
            // Build-and-swap: the rebuild builds the new index into a shadow and
            // swaps it over the live one atomically, so the live field index is
            // never emptied mid-rebuild — EVERY search still finds the notes. A
            // delete-all-then-insert rebuild would let a search land in the empty
            // window and return zero (the full-recall cliff this guards against).
            assert!(
                min_hits > 0,
                "a search during the rebuild returned no hits — recall cliff"
            );

            kernel.close().await.unwrap();
        });
    }
}
