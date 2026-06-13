//! The pure-Rust kernel (#279, slice 2 — PR 1: the no-CPython keystone).
//!
//! This crate composes the native compute plane into the embedded-host shape
//! #224 specs: it owns the collection core (anki via its protobuf service
//! layer, #278), the vector index engine, the derived-text store, the
//! fusion — and, since the tokio pivot (#374), **its own runtime**
//! ([`runtime`]). The kernel is idiomatic async Rust: every op is an
//! `async fn` composing with ordinary awaits (embed → index add → derived
//! ingest), collection access serializes through a task-actor
//! ([`SerializedCollection`]), and hosts adapt the *action exchange* — an op
//! in, a completion-backed future out via [`spawn_op`] — never scheduling.
//! (anki keeps its own runtime for sync; the kernel guarantees sync ops never
//! run on a runtime worker thread, not that only one runtime exists — see
//! [`runtime`] for the `spawn_blocking` discipline and its #503 panic-repro
//! gate.)
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
pub mod cache_layout;
pub mod fusion;
pub mod index_orchestrator;
pub mod media_fetch;
pub mod recognize;
pub mod tag_centroids;
pub mod textsim;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};

use futures::channel::oneshot;
#[cfg(test)]
use futures::future::BoxFuture;
use tracing::Instrument;

use shrike_collection::CollectionCore;
use shrike_derived::DerivedEngine;
use shrike_ffi::{NativeError, NativeResult};
use shrike_index::MultiModalIndex;
use shrike_store_api::{Collection, CreateOutcome, DerivedStore, DuplicatePolicy, VectorIndex};

pub mod runtime;
pub use runtime::{block_on, init_runtime, spawn_op};

// The multi-engine routing key (#485): re-exported so the pyo3 binding maps
// the harness's purpose string onto it as `shrike_kernel::RecognitionPurpose`
// without reaching into the `recognize` module.
pub use recognize::RecognitionPurpose;

// The engine contract (#342): traits live in shrike-engine-api — the kernel
// consumes them and re-exports for downstream paths; it names no engine.
pub use shrike_engine_api::{
    Embedder, ImageEmbedder, ImageResolver, Locator, MediaItem, Recognition, Recognizer, Segment,
};

// The export request/scope/format types (#71): the store contract's, re-exported
// so the pyo3 binding constructs them as `shrike_kernel::*` without reaching
// past the kernel into the store-api crate.
pub use shrike_store_api::{ExportOutcome, ExportRequest, ExportScope, PackageFormat};

/// The collection as a task-actor (#374): every access is one job sent to a
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
/// (pinned in shrike-collection); when client sync (#33/#362) lands, those
/// sync ops MUST ride `spawn_blocking` rather than an inline job — a
/// blocking-pool thread is a legal `block_on` site. The discipline and its
/// panic-repro gate live in [`runtime`] (#503).
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
    pub async fn open(collection_path: String) -> NativeResult<Self> {
        // The open IS the actor's first job: the core is created inside the
        // task, so no collection access ever happens outside it.
        let (opened_tx, opened_rx) = oneshot::channel();
        let (jobs_tx, mut jobs_rx) = tokio::sync::mpsc::unbounded_channel::<Job>();
        let actor = runtime::handle().spawn(async move {
            let core = match CollectionCore::open(&collection_path) {
                Ok(core) => Arc::new(core),
                Err(e) => {
                    let _ = opened_tx.send(Err(e));
                    return;
                }
            };
            let _ = opened_tx.send(Ok(Arc::clone(&core)));
            while let Some(job) = jobs_rx.recv().await {
                job();
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

    /// The actor around a PRE-BUILT store (#389 compose): same loop, same
    /// serialization discipline — only construction differs (an injected
    /// impl is built by the host's assembly, not on the actor task; the
    /// path-opening convenience above keeps anki construction inside it).
    pub fn from_store(core: Arc<dyn Collection>) -> Self {
        let (jobs_tx, mut jobs_rx) = tokio::sync::mpsc::unbounded_channel::<Job>();
        let actor = runtime::handle().spawn(async move {
            while let Some(job) = jobs_rx.recv().await {
                job();
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
    /// idle-released (#64's open-on-demand, kernel-side — so a cooperative
    /// release between any two jobs self-heals; contention surfaces as the
    /// BUSY tier from `ensure_open`).
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
    /// rather than string-matched by the caller (#382). Mirrors `run` except
    /// for that one rule.
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
/// maintains the engine) + the derived store + fusion, every op an idiomatic
/// async fn on the kernel's own runtime (#374). No transport, no Python.
/// Index maintenance is **kernel-internal** (#332 S3d):
/// upserts/deletes keep the orchestrator's vectors, fingerprints, and
/// watermarks current, and the debounced saver (tokio::time, #374 B2)
/// bounds what a crash can discard.
pub struct Kernel {
    /// Arc so the tag refresher's background task can read through the
    /// actor without holding the kernel itself (#445).
    collection: Arc<SerializedCollection>,
    orchestrator: Arc<index_orchestrator::IndexOrchestrator>,
    saver: Arc<index_orchestrator::DebouncedSaver>,
    derived: Arc<dyn DerivedStore>,
    /// The attachable embedding service (#342's first registry slot):
    /// swappable at runtime — the harness attaches on embedding start,
    /// detaches on stop, and a model swap is detach + attach. Ops that need
    /// embedding degrade (lexical-only search, unindexed-but-created upserts)
    /// when the slot is empty, mirroring the Python host's gating. Arc so
    /// the tag refresher shares the gate (#445).
    embed: Arc<RwLock<Option<Arc<EmbedService>>>>,
    /// Tag-centroid state (#178/#179): the live key→tag map for the engine's
    /// `tag.text` space + the hygiene knobs. Centroids refresh in the
    /// background after membership-relevant index ops, coalesced under
    /// write bursts (#445); boot/rebuild paths refresh synchronously.
    tag_keys: Arc<tag_centroids::TagKeyMap>,
    tag_config: tag_centroids::TagCentroidConfig,
    tag_refresh: Arc<tag_centroids::TagRefresher>,
    /// The recognition services (#228/#342/#485, the second registry slot):
    /// OCR/ASR/describe engines the harness attaches at runtime, exactly like
    /// the embed slot — but **keyed by purpose** (#485) so OCR, ASR, and VLM
    /// describe can be attached independently, each sweeping its own pending
    /// set / source / fingerprint / destination. The kernel runs the pipeline
    /// over whatever is registered; recognition for a purpose is simply off
    /// when its slot is empty.
    recognize: RwLock<BTreeMap<recognize::RecognitionPurpose, Arc<RecognizeService>>>,
    recognition_gate: recognize::RecognitionGate,
}

/// One attached recognition capability: the engine + the media resolver it
/// reads bytes through (independent of the embed slot — an OCR-only
/// deployment has no image embedder).
pub struct RecognizeService {
    pub recognizer: Arc<dyn Recognizer>,
    pub resolver: Arc<dyn ImageResolver>,
}

/// The derived-store source recognized image text lands under (#199/#228).
pub const OCR_SOURCE: &str = "ocr";
/// The derived-store meta key holding the recognizer fingerprint.
pub const RECOGNIZER_FINGERPRINT_KEY: &str = "recognizer_fingerprint";

/// One attached embedding capability: the text embedder + optionally its
/// image half (present only for an image-advertising backend with a media
/// resolver).
pub struct EmbedService {
    pub embedder: Arc<dyn Embedder>,
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

/// A shared, pre-enumerated `(note_id, media_names)` set for one media kind
/// (#485): the multi-purpose recognition sweep enumerates each kind once and
/// hands this Arc to every same-kind purpose's sweep, so the collection is
/// never re-scanned per purpose.
type MediaRefs = Arc<[(i64, Vec<String>)]>;

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
    /// Scheduling AND timing are the kernel's own (#374): the owned runtime
    /// spawns the collection actor, and the debounced index flush rides
    /// tokio::time unconditionally. Saver defaults apply; a host with tuning
    /// flags uses [`Self::open_with`].
    pub async fn open(collection_path: &str, cache_dir: &str) -> NativeResult<Self> {
        Self::open_with(collection_path, cache_dir, None, None).await
    }

    /// [`Self::open`] with the index-flush tuning the host's
    /// `--index-save-*` flags carry (#355 item 2): `save_delay` is the idle
    /// debounce in seconds, `save_threshold` the unsaved-change count that
    /// forces an immediate flush. `None` = the built-in default.
    pub async fn open_with(
        collection_path: &str,
        cache_dir: &str,
        save_delay: Option<f64>,
        save_threshold: Option<u64>,
    ) -> NativeResult<Self> {
        // The derived store opens its file under cache_dir before assemble
        // runs, so the dir must exist first (assemble re-creates idempotently
        // for composed callers).
        std::fs::create_dir_all(cache_dir)
            .map_err(|e| NativeError::internal(format!("cache dir: {e}")))?;
        let collection = Arc::new(SerializedCollection::open(collection_path.to_string()).await?);
        let engine: Arc<dyn VectorIndex> = Arc::new(MultiModalIndex::new(
            NOTE_MODALITIES
                .iter()
                .map(|m| m.to_string())
                .chain(std::iter::once(TAG_TEXT_SPACE.to_string()))
                .collect(),
        )?);
        // The derived store is namespaced per collection (#547), mirroring the
        // index (#67): `<cache_dir>/derived/<namespace>/shrike.db`, so a daemon
        // serving several collections never shares one `shrike.db` (which would
        // cross-contaminate substring/fuzzy/OCR search). Migrate an existing
        // flat `<cache_dir>/shrike.db` into this collection's namespace first,
        // so the single-collection user keeps their built derived data.
        cache_layout::migrate_flat_derived(cache_dir, collection_path);
        let derived_path = cache_layout::derived_db_path(cache_dir, collection_path);
        if let Some(parent) = derived_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| NativeError::internal(format!("derived dir: {e}")))?;
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
        // The vector index is namespaced per collection (#67): each collection
        // gets its own `<cache_dir>/index/<path-derived-id>/` so a daemon
        // serving several collections never collides their indexes. The layout
        // also migrates an existing flat (single-collection) layout into this
        // collection's namespace, losslessly, so the long-standing text-only
        // user never pays a spurious rebuild on upgrade.
        let layout = cache_layout::IndexLayout::for_collection(cache_dir, collection_path);
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

    /// The injection seam (#389): a kernel over PRE-BUILT stores — the
    /// deployment ladder's composition point (remote/platform impls swap in
    /// here; [`Self::open`] is the all-local convenience over it). The
    /// collection arrives as the bare store; the kernel wraps it in its own
    /// task-actor (serialization is kernel policy, not the store's).
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
        // so it can't derive a per-collection index namespace (#67): the index
        // stays flat under `cache_dir`. The composing caller owns multiplexing
        // if it ever serves several collections through this seam.
        let layout = cache_layout::IndexLayout::flat(cache_dir);
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
        index_layout: cache_layout::IndexLayout,
        save_delay: Option<f64>,
        save_threshold: Option<u64>,
    ) -> NativeResult<Self> {
        std::fs::create_dir_all(cache_dir)
            .map_err(|e| NativeError::internal(format!("cache dir: {e}")))?;
        std::fs::create_dir_all(&index_layout.dir)
            .map_err(|e| NativeError::internal(format!("index dir: {e}")))?;
        let orchestrator = Arc::new(index_orchestrator::IndexOrchestrator::open_owned(
            index_layout.dir,
            engine,
            index_layout.owner,
        ));
        let saver = index_orchestrator::DebouncedSaver::new(
            Arc::clone(&orchestrator),
            save_delay.unwrap_or(index_orchestrator::DEFAULT_SAVE_DELAY),
            save_threshold.unwrap_or(index_orchestrator::DEFAULT_SAVE_THRESHOLD),
        );
        let embed: Arc<RwLock<Option<Arc<EmbedService>>>> = Arc::new(RwLock::new(None));
        let tag_keys = Arc::new(tag_centroids::TagKeyMap::default());
        let tag_config = tag_centroids::TagCentroidConfig::default();
        let tag_refresh = tag_centroids::TagRefresher::new(
            Arc::clone(&collection),
            orchestrator.engine_arc(),
            Arc::clone(&tag_keys),
            tag_config.clone(),
            Arc::clone(&saver),
            Arc::clone(&embed),
        );
        Ok(Self {
            collection,
            orchestrator,
            saver,
            derived,
            embed,
            tag_keys,
            tag_config,
            tag_refresh,
            recognize: RwLock::new(BTreeMap::new()),
            recognition_gate: recognize::RecognitionGate::default(),
        })
    }

    /// Attach (or swap) the OCR recognition service — the #342 slot pattern,
    /// second instance. The OCR-defaulting convenience over
    /// [`attach_recognizer_with`] (#485): existing hosts and kernel tests keep
    /// the single-arg shape and target the OCR purpose. The harness follows up
    /// by driving the pending sweep.
    pub fn attach_recognizer(
        &self,
        recognizer: Arc<dyn Recognizer>,
        resolver: Arc<dyn ImageResolver>,
    ) {
        self.attach_recognizer_with(recognize::RecognitionPurpose::Ocr, recognizer, resolver);
    }

    /// Attach (or swap) the recognition service for a specific purpose (#485)
    /// — OCR, ASR, or VLM describe, each routed to its own pending set /
    /// source / fingerprint / destination by the sweep.
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

    /// Detach the recognition service for a specific purpose (#485).
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

    /// The recognition service for a specific purpose, if attached (#485).
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

    /// The derived `source` strings whose rows are vector-minting (#485): the
    /// union of every purpose's source (all recognition purposes mint
    /// vectors — only the LEXICAL surfaces differ). `compose_embed_inputs`
    /// reads OCR/ASR/VLM recognized text from these.
    fn vector_minting_sources() -> Vec<&'static str> {
        Self::ALL_PURPOSES.iter().map(|p| p.source()).collect()
    }

    /// The derived `source` strings HIDDEN from the lexical (substring/fuzzy)
    /// surfaces (#485) — every [`recognize::Destination::VectorOnly`]
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
                let total = core.note_count()?;
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
        self.saver.request_save();
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
    #[must_use = "whether reindexing ran is the caller's signal to refresh status"]
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
        let inputs = self.compose_embed_inputs(raw, None);
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
        let inputs = self.compose_embed_inputs(raw, None);
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

    /// Compose orchestrator inputs from collection rows + the derived
    /// store's recognized texts across EVERY vector-minting source
    /// (#199/#228/#485): the index derives from collection text + OCR + ASR +
    /// VLM-describe, so reconcile == rebuild keeps holding after recognition,
    /// and a note's recognized text mints vectors on any (re-)embed path.
    /// Vector-worthiness re-judges from the stored text (confidence already
    /// gated at ingest). The destination axis is lexical-visibility only — a
    /// VectorOnly (VLM) source still feeds this embed path; it is merely
    /// hidden from the substring/fuzzy surfaces.
    fn compose_embed_inputs(
        &self,
        raw: Vec<(i64, String, Vec<String>)>,
        only_notes: Option<&[i64]>,
    ) -> Vec<index_orchestrator::EmbedInput> {
        let mut recognized_map: std::collections::HashMap<i64, Vec<String>> =
            std::collections::HashMap::new();
        // Union every recognition source's vector-worthy text under the note
        // key. Per-op callers scope the read to the written notes (#445); the
        // full-set read is for rebuild/reconcile, which consume everything.
        // One query per source (a small fixed set — ocr/vlm/asr), each bounded
        // by rows that EXIST for that source (most notes have none), so this
        // is proportional to recognized-content volume, not 3× the
        // collection. (A single `source IN (…)` read could collapse it if a
        // profile ever flags it.)
        for source in Self::vector_minting_sources() {
            let texts = match only_notes {
                Some(ids) => self.derived.texts_for_source_for_notes(source, ids),
                None => self.derived.texts_for_source(source),
            };
            match texts {
                Ok(rows) => {
                    for (nid, _r, text) in rows {
                        if self.recognition_gate.vector_worthy(&text) {
                            recognized_map.entry(nid).or_default().push(text);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(error = ?e, %source, "reading recognized texts failed; embedding without them");
                }
            }
        }
        raw.into_iter()
            .map(
                |(note_id, text, image_names)| index_orchestrator::EmbedInput {
                    note_id,
                    text,
                    image_names,
                    ocr_texts: recognized_map.remove(&note_id).unwrap_or_default(),
                },
            )
            .collect()
    }

    /// Full derived-text (FTS5) rebuild, entirely kernel-side (#445): one
    /// collection job collects the field rows, the build runs on the blocking
    /// pool against the kernel's own engine — the rows never cross the FFI.
    /// (The Python path round-tripped the whole collection's text
    /// Rust→Python→Rust: ~150-250MB transient at 100k notes, on every drifted
    /// boot, /reload, and cooperative re-acquire.) Returns
    /// `(row_count, col_mod)` — the col_mod is the BUILD's own snapshot (read
    /// in the same collection job as the rows), so the host's watermark can't
    /// mask a write that landed mid-build.
    pub async fn rebuild_derived(&self) -> NativeResult<(usize, i64)> {
        // Commit-then-verify (#471): the collect runs in one actor job but
        // the build commits OFF the actor, so concurrent jobs (an upsert's
        // ingest, a sweep's OCR store for a new note) can land in the window
        // — and the build's field-replace + dead-note prune would erase
        // them. Every lossy interleave bumps col.mod (note writes do; OCR
        // stores touch only existing notes, which the prune keeps), so
        // re-reading col_mod after the commit and re-collecting on movement
        // converges. After the cap the watermark stays at the last snapshot
        // — visible drift, healed by the next rebuild plus the sweep
        // re-pending any pruned OCR rows.
        const REBUILD_ATTEMPTS: usize = 3;
        let mut last = (0usize, 0i64);
        for attempt in 1..=REBUILD_ATTEMPTS {
            let (n, dmod, now) = self.rebuild_derived_once().await?;
            last = (n, dmod);
            if now == dmod {
                return Ok(last);
            }
            tracing::debug!(
                attempt,
                "collection moved during derived build; re-collecting"
            );
        }
        tracing::warn!(
            "derived rebuild raced sustained writes; watermark left at the last snapshot"
        );
        Ok(last)
    }

    /// One collect → build → verify pass: returns the row count, the
    /// snapshot `col_mod` the build committed under, and the post-commit
    /// `col_mod` (equal means nothing interleaved).
    async fn rebuild_derived_once(&self) -> NativeResult<(usize, i64, i64)> {
        let (rows, dmod) = self
            .collection
            .run(|core| -> NativeResult<_> {
                let ids = core.find_notes("")?;
                let rows = core.derived_field_rows(&ids)?;
                let dmod = core.col_mod()?;
                Ok((rows, dmod))
            })
            .await??;
        let n = rows.len();
        let derived = Arc::clone(&self.derived);
        tokio::task::spawn_blocking(move || derived.build(&rows, dmod))
            .await
            .map_err(|e| NativeError::internal(format!("derived build task: {e}")))??;
        let now = self.collection.run(|core| core.col_mod()).await??;
        Ok((n, dmod, now))
    }

    /// One bounded recognition sweep across EVERY attached purpose
    /// (#228/#485): for each of OCR / ASR / VLM-describe that has a recognizer
    /// attached, recognize up to `max_items` of ITS pending media, persist
    /// gated text + segments per its destination, and re-embed the affected
    /// notes so recognition vectors mint. Each purpose sweeps independently
    /// (its own pending set / source / fingerprint key / destination), and
    /// the per-purpose chunk-Err-aborts-before-persist contract holds per
    /// sweep — a down describe endpoint leaves its backlog intact without
    /// touching OCR. Returns an AGGREGATED report (summed counts, max
    /// remaining) so the harness's existing `remaining > 0` driver loop is
    /// unchanged; `Unavailable` only when NO purpose is attached.
    pub async fn recognize_pending(
        &self,
        max_items: usize,
    ) -> NativeResult<recognize::SweepReport> {
        let purposes = self.attached_recognition_purposes();
        if purposes.is_empty() {
            return Ok(recognize::SweepReport::Unavailable);
        }
        // Enumerate each NEEDED media kind ONCE (#445: no per-purpose
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

    /// One bounded recognition sweep for a SINGLE purpose (#485) — the
    /// per-engine routing of the original #228 sweep. Pending = a resolvable
    /// media ref of this purpose's media kind with no row for THIS source AND
    /// no below-gate marker (#416) — or all of them after this purpose's
    /// recognizer-fingerprint changes (its own meta key, so an OCR upgrade
    /// never re-derives ASR/VLM and vice versa). Persists per the purpose's
    /// destination: a VectorOnly (VLM) row is still stored (for provenance +
    /// reconcile) but the kernel excludes its source from the lexical
    /// surfaces. On a recognizer chunk `Err`, the sweep aborts BEFORE
    /// persisting anything or advancing this purpose's fingerprint meta —
    /// everything stays pending and the next sweep retries (the load-bearing
    /// describe-engine contract, preserved per purpose).
    pub async fn recognize_pending_for(
        &self,
        purpose: recognize::RecognitionPurpose,
        max_items: usize,
    ) -> NativeResult<recognize::SweepReport> {
        // Enumerate this purpose's media kind once, then delegate. The
        // multi-purpose driver shares one enumeration across same-kind
        // purposes (#445); this single-purpose entry point is the
        // test/binding convenience.
        if self.recognize_service_for(purpose).is_none() {
            return Ok(recognize::SweepReport::Unavailable);
        }
        let refs: MediaRefs = self.note_media_refs(purpose.media_kind()).await?.into();
        self.recognize_pending_for_refs(purpose, max_items, &refs)
            .await
    }

    /// [`recognize_pending_for`] over a PRE-ENUMERATED media-ref set (#445):
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
            // Below-gate markers ride the same invalidation (#416): the new
            // engine may read what the old one couldn't, so gated items
            // re-enter the pending set exactly like stored rows re-derive.
            self.derived.clear_gated(source)?;
        }

        // Pending set: resolvable media of this purpose's kind without a row
        // for THIS source — and without a below-gate marker (#416): an item
        // the gate dropped is DONE (its outcome can't change until the
        // fingerprint does), not pending, so it is never re-recognized and an
        // all-gated window converges instead of re-taking itself forever.
        // `raw` is the pre-enumerated (note_id, names) set shared across
        // same-kind purposes (#445): the pending diff needs only the names.
        let mut done: std::collections::HashSet<(i64, String)> =
            self.derived.refs_for_source(source)?.into_iter().collect();
        done.extend(self.derived.gated_refs_for_source(source)?);
        let mut pending: Vec<(i64, String)> = Vec::new();
        for (note_id, names) in raw {
            for name in names {
                if !done.contains(&(*note_id, name.clone())) && svc.resolver.exists(name) {
                    pending.push((*note_id, name.clone()));
                }
            }
        }
        let total_pending = pending.len();
        pending.truncate(max_items);
        if pending.is_empty() {
            self.derived.meta_set(fingerprint_key, &fingerprint)?;
            return Ok(recognize::SweepReport::Idle);
        }

        // One pass, many consumers: recognize the batch, keep text AND
        // segments. A read that fails after the exists() check (TOCTOU
        // delete, transient error, resolver bug) SKIPS the item rather than
        // recognizing empty bytes (#386) — nothing is stored for it, so the
        // next sweep re-offers it. `sent` and `items` are built together so
        // the recognizer's output stays aligned with what was actually sent.
        let mut sent: Vec<(i64, String)> = Vec::with_capacity(pending.len());
        let mut items: Vec<MediaItem> = Vec::with_capacity(pending.len());
        for (note_id, name) in &pending {
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
        let recognitions = if items.is_empty() {
            Vec::new()
        } else {
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
        // a below-gate marker (#416), so the next sweep's pending diff counts
        // it done instead of re-recognizing it forever. Markers are cleared
        // with the rows on a fingerprint change (above), so an engine upgrade
        // re-judges them like everything else.
        // Scoped to the batch's notes (#445): the merge previously read the
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
                            .map_err(|e| NativeError::internal(format!("segments: {e}")))?;
                        segments.push((*note_id, name.clone(), json));
                    }
                    stored_count += 1;
                }
            }
        }
        let affected: Vec<i64> = touched.keys().copied().collect();
        for (note_id, refs_text) in &touched {
            self.derived.ingest(*note_id, source, refs_text)?;
        }
        for (note_id, name, json) in &segments {
            self.derived.put_segments(*note_id, source, name, json)?;
        }
        self.derived.mark_gated(source, &gated)?;
        self.derived.meta_set(fingerprint_key, &fingerprint)?;

        // Re-embed the affected notes: their hash now folds the recognized
        // text, so this mints the vectors and the next reconcile sees them
        // current. (A VectorOnly source's rows are excluded from the lexical
        // surfaces, but still mint a vector here.)
        if !affected.is_empty() {
            self.index_written(&affected).await?;
        }

        // `recognized` counts what was actually sent. A skipped (unreadable)
        // item stores nothing, so it STAYS PENDING and the next call re-takes
        // the same window — with an unreadable prefix of the pending order
        // and more items beyond it, `remaining` stays > 0 indefinitely. The
        // kernel does not terminate that loop; the HARNESS's driver stops on
        // a no-progress batch (`recognized == 0`) and the next sweep trigger
        // (boot, /reload, cooperative re-acquire) retries the read.
        Ok(recognize::SweepReport::Ran {
            recognized: sent.len(),
            stored: stored_count,
            remaining: total_pending.saturating_sub(pending.len()),
        })
    }

    /// `(note_id, media_names)` for every note referencing media of `kind` —
    /// the sweep's pending-set source, routed by media kind (#485). Image
    /// refs come from `note_image_refs` (the `<img src>` extractor); audio
    /// enumeration (`note_audio_refs`, the `[sound:…]` extractor) lands in
    /// Slice 2 (#485) — until then an Audio purpose has an empty pending set.
    async fn note_media_refs(
        &self,
        kind: recognize::MediaKind,
    ) -> NativeResult<Vec<(i64, Vec<String>)>> {
        match kind {
            recognize::MediaKind::Image => {
                self.collection.run(|core| core.note_image_refs()).await?
            }
            // Slice 2 wires `core.note_audio_refs()` here.
            recognize::MediaKind::Audio => Ok(Vec::new()),
        }
    }

    async fn index_written(&self, written: &[i64]) -> NativeResult<()> {
        if written.is_empty() {
            return Ok(());
        }
        // First-occurrence dedupe (#445): a batch that wrote the same note
        // twice must not index it twice — and the collection readers' move-
        // out assembly hands a repeated id its content on the first
        // occurrence only.
        let mut seen = std::collections::BTreeSet::new();
        let written: Vec<i64> = written
            .iter()
            .copied()
            .filter(|id| seen.insert(*id))
            .collect();
        let written = &written[..];
        let ids = written.to_vec();
        let (raw_inputs, rows, tagged) = self
            .collection
            .run(move |core| -> NativeResult<_> {
                let inputs = core.note_embed_inputs(&ids)?;
                let rows = core.derived_field_rows(&ids)?;
                let tagged = core.any_tagged(&ids)?;
                Ok((inputs, rows, tagged))
            })
            .await??;
        let svc = self.embed_service();
        if let Some(svc) = &svc {
            let inputs = self.compose_embed_inputs(raw_inputs, Some(written));
            self.orchestrator
                .add(&inputs, &*svc.embedder, svc.images_pair())
                .await?;
        }
        // Group the derived rows per note; ONE transaction for the whole
        // batch (#445 — one commit per note was journal churn × batch size).
        let mut refs: BTreeMap<i64, Vec<(String, String)>> = BTreeMap::new();
        for (nid, _source, name, value) in rows {
            refs.entry(nid).or_default().push((name, value));
        }
        let batch: Vec<(i64, Vec<(String, String)>)> = written
            .iter()
            .map(|note_id| (*note_id, refs.remove(note_id).unwrap_or_default()))
            .collect();
        self.derived.ingest_many(&batch, FIELD_SOURCE)?;
        self.advance_watermarks(svc.is_some()).await?;
        // Tag centroids derive from the text vectors just written (#179) —
        // refreshed off the op tail, and only when the op could have changed
        // membership: a written note carries tags now, or was a member
        // before (an update that removed them) (#445).
        if tagged || self.tag_keys.any_member_of(written) {
            self.tag_refresh.request();
        }
        Ok(())
    }

    /// The wire-shaped bulk upsert (#77): the collection core's NAMED upsert
    /// — `id`?/`note_type`/`deck`/`fields` map/`tags`, create AND update,
    /// `dry_run`, typed per-item results — run as ONE collection job, then
    /// the kernel-internal index/derived maintenance over everything written.
    /// This is the op the MCP `upsert_notes` action rides (S3d-2).
    pub async fn upsert_notes_wire(
        &self,
        notes: Vec<shrike_schemas::NoteInput>,
        policy: DuplicatePolicy,
        dry_run: bool,
    ) -> NativeResult<Vec<shrike_schemas::UpsertNoteResult>> {
        let span = tracing::debug_span!("kernel.upsert_notes_wire", dry_run);
        async move {
            let results = self
                .collection
                .run(move |core| core.upsert_notes(&notes, policy, dry_run))
                .await??;
            if !dry_run {
                // Typed outcomes (#391): written ids come straight off the
                // variants — the parse-own-output round-trip is gone.
                let written: Vec<i64> = results
                    .iter()
                    .filter_map(|r| match r {
                        shrike_schemas::UpsertNoteResult::Created { id, .. }
                        | shrike_schemas::UpsertNoteResult::Updated { id, .. } => Some(*id),
                        _ => None,
                    })
                    .collect();
                self.index_written(&written).await?;
            }
            Ok(results)
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
            self.saver.request_save();
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
        self.drop_note_sidecars(&note_ids).await
    }

    /// The sidecar tail shared by every note-deletion shape (#382): drop the
    /// notes' vectors + derived rows, advance the watermarks, refresh tags.
    async fn drop_note_sidecars(&self, note_ids: &[i64]) -> NativeResult<()> {
        self.orchestrator.remove(note_ids)?;
        self.derived.remove(note_ids, None)?;
        self.advance_watermarks(self.embed_service().is_some())
            .await?;
        // A deletion changes membership only if a deleted note was IN it —
        // the in-memory probe alone decides (the rows are already gone), and
        // the refresh runs off the op tail (#445).
        if self.tag_keys.any_member_of(note_ids) {
            self.tag_refresh.request();
        }
        Ok(())
    }

    /// A metadata-only collection change (tags/decks/templates/field metadata
    /// — nothing that feeds embedding text or derived rows): advance the
    /// watermarks so the col_mod bump doesn't read as drift on next boot.
    pub async fn metadata_changed(&self) -> NativeResult<()> {
        self.advance_watermarks(self.embed_service().is_some())
            .await?;
        // Tag-only ops (update_note_tags / rename_tag) ride this path and
        // change MEMBERSHIP without an index op — centroids previously went
        // stale until the next upsert/delete or reboot. The coalescing
        // refresher makes the over-trigger on deck/template/field-metadata
        // edits cheap, so request unconditionally (#445).
        self.tag_refresh.request();
        Ok(())
    }

    pub async fn delete_notes(&self, note_ids: Vec<i64>) -> NativeResult<usize> {
        let ids = note_ids.clone();
        let removed = self
            .collection
            .run(move |core| core.delete_notes(&ids))
            .await??;
        self.drop_note_sidecars(&note_ids).await?;
        Ok(removed)
    }

    /// Import an `.apkg`/`.colpkg` package, then bring the index in line (#72).
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
        // line in one op, exactly as `harness.reload` does (the host coordination
        // the old `derived is the harness's job` comment hand-waved but never
        // wired). The col.mod bump is the drift signal for both; we never advance
        // a watermark before reconciling (that would suppress it).
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

    // ── media + maintenance ops (#391 re-home, decision 3) ──────────────────
    // The #70 media tools and the #89 prune as maintained kernel ops: the
    // host keeps only the tool signatures (and the serving-URL fill, which is
    // host config). Media never touches embedding text, so none of these do
    // index work — except prune, whose deletions carry their own sidecar
    // tail below.

    /// The full store_media batch: each item's byte source (base64 decode /
    /// SSRF-guarded URL download) prepares CONCURRENTLY on the blocking pool
    /// — the host facade's gather-over-to_thread, re-homed — then the batch
    /// writes as ONE collection job (`path` items run their containment
    /// gates under that job; they carry no prepare work).
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
                tokio::task::spawn_blocking(move || {
                    crate::media_fetch::prepare_media_item(i as i64, item, allow_private_fetch)
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
                    .map_err(|e| NativeError::internal(format!("media prepare task: {e}")))?,
            );
        }
        self.collection
            .run(move |core| core.store_prepared_media(&prepared, &path_roots))
            .await?
    }

    /// Resolve filenames to where their bytes live (never the bytes; the
    /// host fills the serving `url`).
    pub async fn fetch_media(
        &self,
        filenames: Vec<String>,
    ) -> NativeResult<Vec<shrike_schemas::MediaFetchResult>> {
        self.collection
            .run(move |core| core.fetch_media(&filenames))
            .await?
    }

    /// List media files (sorted, optional glob + limit).
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
    pub async fn delete_media(
        &self,
        filenames: Vec<String>,
    ) -> NativeResult<shrike_schemas::DeleteMediaResponse> {
        self.collection
            .run(move |core| core.delete_media(&filenames))
            .await?
    }

    /// Read-only media diagnostics.
    pub async fn media_check(&self) -> NativeResult<shrike_schemas::CollectionCheckResponse> {
        self.collection.run(move |core| core.media_check()).await?
    }

    /// Export the collection (or a scope of it) to an Anki package (#71).
    ///
    /// Read-only on the collection's data, but it holds the collection for the
    /// whole package write — so it rides the collection task-actor (serializing
    /// against other ops on this collection, exactly like a write; export is
    /// exclusive by nature). The host has already gated `out_path` (the
    /// path-safety check); the kernel trusts it and performs the anki export.
    /// `out_path` extension picks the format isn't done here — the host passes
    /// an explicit [`PackageFormat`] so the kernel never guesses.
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

    /// The #89 prune with its maintenance tail (the host's old post-apply
    /// block, re-homed): deletions drop their sidecars like `delete_notes`;
    /// a tags-only prune is a metadata-only watermark advance. The tail is
    /// best-effort — a failure logs and the response still returns (the next
    /// boot's drift check repairs).
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
                self.metadata_changed().await
            } else {
                Ok(())
            };
            if let Err(e) = tail {
                tracing::warn!(error = %e, "index maintenance after prune failed");
            }
        }
        Ok(response)
    }

    // ── tag + deck ops (#391 re-home, long-tail group 2) ────────────────────
    // Tags and deck names are not embedding text: each op is one collection
    // job plus the best-effort metadata watermark tail when anything changed
    // (mirroring the host's _bump_col_mod_after_metadata_change, whose
    // remaining callers retire with the note-type re-home).

    /// The shared metadata tail: advance the watermarks so a vectors-
    /// unchanged col_mod bump doesn't read as drift on next boot. Best-effort
    /// — cache bookkeeping never fails an op already committed.
    async fn metadata_tail(&self, changed: bool) {
        if !changed {
            return;
        }
        if let Err(e) = self.metadata_changed().await {
            tracing::warn!(error = %e, "watermark bump after metadata change failed");
        }
    }

    /// Edit tags on a note set (`set` full-replace XOR add/remove — the
    /// exclusivity is the host's input validation).
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
        self.metadata_tail(response.notes_modified > 0).await;
        Ok(response)
    }

    /// Rename a tag collection-wide (empty `note_ids`) or exactly on a set.
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
        self.metadata_tail(response.notes_modified > 0).await;
        Ok(response)
    }

    /// Create or rename decks in bulk (id present = rename; never merges).
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
        self.metadata_tail(changed).await;
        Ok(results)
    }

    /// Delete decks by reference — only if empty.
    pub async fn delete_decks(
        &self,
        refs: Vec<String>,
    ) -> NativeResult<shrike_schemas::DeleteDecksResponse> {
        let response = self
            .collection
            .run(move |core| core.delete_decks(&refs))
            .await??;
        self.metadata_tail(!response.deleted.is_empty()).await;
        Ok(response)
    }

    // ── note-type ops (#391 re-home, long-tail group 3) ─────────────────────
    // The #76/#119/#75 note-type tools as maintained kernel ops. The
    // structural edits (upsert/fields/templates/delete) carry NO tail: a
    // field-list change can alter embedding text, so their col.mod bump
    // deliberately reads as drift on the next boot (a removed field makes a
    // rebuild correct; for the rest it's conservative — the pre-re-home
    // contract). Template/CSS text and field metadata never feed embedding
    // text, so those two advance the watermarks; migration moves note fields,
    // so an apply re-embeds the changed notes.

    /// Create/update note-type definitions in bulk (the position-keyed
    /// replace with the #76 unsound-move rejection), per-item results.
    pub async fn upsert_note_types(
        &self,
        note_types: Vec<shrike_schemas::NoteTypeInput>,
    ) -> NativeResult<Vec<shrike_schemas::NoteTypeResult>> {
        self.collection
            .run(move |core| core.upsert_note_types(&note_types))
            .await?
    }

    /// Identity-based field ops (add/remove/rename/reposition), atomic.
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
            if let Err(e) = self.metadata_changed().await {
                tracing::warn!(error = %e, "watermark advance after find_replace_note_types failed");
            }
        }
        Ok(response)
    }

    /// Per-field editor metadata (font/size/description), with the
    /// unconditional watermark tail: editor cosmetics never touch embedding
    /// text, but the persist bumps col.mod. Best-effort, like the tag/deck
    /// metadata ops.
    pub async fn update_note_type_field_metadata(
        &self,
        note_type_name: String,
        updates: Vec<shrike_schemas::FieldMetadataInput>,
    ) -> NativeResult<shrike_schemas::UpdateNoteTypeFieldMetadataResponse> {
        let response = self
            .collection
            .run(move |core| core.update_note_type_field_metadata(&note_type_name, &updates))
            .await??;
        if let Err(e) = self.metadata_changed().await {
            tracing::warn!(error = %e, "watermark advance after field-metadata update failed");
        }
        Ok(response)
    }

    /// Change notes' note type via name maps (#75). Migration moves field
    /// content — embedding text — under unchanged ids, so an apply re-embeds
    /// and re-ingests the changed notes (best-effort: the migration is
    /// committed either way; the next boot's drift check repairs). Dry-run
    /// touches nothing.
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
    pub async fn delete_note_types(
        &self,
        ids: Vec<i64>,
    ) -> NativeResult<Vec<shrike_schemas::DeleteNoteTypeResult>> {
        self.collection
            .run(move |core| core.delete_note_types(&ids))
            .await?
    }

    /// Fused search with the kernel's default arguments — a thin delegate to
    /// [`actions::search_notes`], the ONE fused-search spine (#388): no
    /// scope, no threshold, no image floor, the canonical `fusion` weights.
    /// The query embeds here when an embedder is attached; otherwise the
    /// action degrades to the lexical signals. `score` carries the wire
    /// contract's semantic similarity when present (the raw RRF magnitude is
    /// deliberately not exposed, matching the host surface).
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
            let args = actions::SearchArgs {
                top_k,
                threshold: 0.0,
                weights: BTreeMap::new(), // empty = the canonical fusion set
                semantic,
                index_size: self.orchestrator.engine().size(),
                hidden_lexical_sources: Self::hidden_lexical_sources()
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
                ..Default::default()
            };
            let engine = self.orchestrator.engine_arc();
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

    /// Close the collection and drain the actor — close returns with nothing
    /// in flight (the interpreter-teardown guard, #374 design 7). Works
    /// through `&self` so shared handles (the binding's `Arc<Kernel>`) can
    /// close; idempotent (`SerializedCollection::close` treats a drained
    /// actor as already-closed, #382).
    pub async fn close(&self) -> NativeResult<()> {
        // A sleeping coalesced tag refresh has nothing to read once the
        // actor drains — abort it first (#445).
        self.tag_refresh.shutdown();
        let result = self.collection.close().await;
        self.collection.shutdown().await;
        result
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

    /// Test shim (#391): the wire-shaped upsert with the pre-typed-seam call
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

    /// The op-tail tag refresh is a coalesced background task since #445 —
    /// poll until the tag state reaches the expected shape (an isolated
    /// op's first fire runs immediately, so this resolves in milliseconds).
    async fn wait_for_tags(kernel: &Kernel, pred: impl Fn(&tag_centroids::TagKeyMap) -> bool) {
        for _ in 0..500 {
            if pred(kernel.tag_keys()) {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("tag state never reached the expected shape");
    }

    /// Compile-time pin: every kernel future is Send, so a harness may spawn
    /// kernel ops on any multithreaded runtime (the #310 contract — a !Send
    /// regression, e.g. an entered span guard held across an await, fails
    /// here instead of downstream).
    /// #374 design 2: dropping the edge wrapper DETACHES observation — the
    /// spawned op runs to completion (never an abort; a half-applied
    /// collection write would be corruption). The regression guard against
    /// a JoinHandle-shaped wrapper.
    #[test]
    fn dropping_the_op_wrapper_never_aborts_the_work() {
        crate::runtime::block_on(async {
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
            let _ = std::fs::remove_dir_all(dir);
        });
    }

    /// The actor serializes FIFO: jobs sent in order run in order, even with
    /// every completion awaited concurrently.
    #[test]
    fn collection_jobs_run_fifo() {
        crate::runtime::block_on(async {
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
            let _ = std::fs::remove_dir_all(dir);
        });
    }

    fn assert_send<F: std::future::Future + Send>(f: F) -> F {
        f
    }

    #[test]
    fn detach_degrades_and_reattach_recovers() {
        // The embed slot is runtime-swappable (#342): detached, the kernel
        // still creates notes and serves lexical search; re-attached, the
        // stale index watermark makes reindex catch up on what it missed.
        crate::runtime::block_on(async {
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
            // Idempotent (#382): a second close after the actor drained is
            // already-closed, not an actor-gone error.
            kernel.close().await.unwrap();
            std::fs::remove_dir_all(dir).ok();
        });
    }

    #[test]
    fn composed_kernel_serves_ops_over_injected_stores() {
        // The #389 injection seam: a kernel assembled from PRE-BUILT stores
        // (the deployment ladder's composition point) behaves like an opened
        // one — the actor wraps the injected collection, ops serve, and the
        // index/derived paths ride the injected trait objects.
        crate::runtime::block_on(async {
            let dir = temp_dir();
            std::fs::create_dir_all(dir.join("cache")).unwrap();
            let collection: Arc<dyn shrike_store_api::Collection> =
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
            assert!(kernel.index().engine().contains(nid));
            let hits = kernel.search("composed kernels", 5).await.unwrap();
            assert_eq!(hits[0].note_id, nid);

            kernel.close().await.unwrap();
            std::fs::remove_dir_all(dir).ok();
        });
    }

    #[test]
    fn media_ops_and_prune_run_as_kernel_ops() {
        // The #391 re-home: store_media's byte prepare rides the blocking
        // pool with per-item errors, the read ops round-trip, and prune
        // carries its own maintenance tail (vectors leave with the notes,
        // the watermark advances — no host-side forget/metadata calls).
        crate::runtime::block_on(async {
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
            std::fs::remove_dir_all(dir).ok();
        });
    }

    #[test]
    fn tag_deck_ops_carry_the_metadata_tail() {
        // The #391 group-2 tail at its home: a real change advances the
        // index watermark to the new col_mod (no drift on the next check);
        // a no-op batch leaves the watermark EXACTLY where it was — the
        // `changed` guard skips the tail, not merely "no drift afterwards"
        // (the host-side spies this replaces asserted the skip directly).
        crate::runtime::block_on(async {
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

            let watermark = |k: &Kernel| k.index().status().col_mod;
            let before = watermark(&kernel);

            // No-op: a non-existent note modifies nothing → the tail is
            // skipped and the watermark doesn't move at all.
            let miss = kernel
                .update_note_tags(vec![999_999], None, vec!["x".into()], Vec::new())
                .await
                .unwrap();
            assert_eq!(miss.notes_modified, 0);
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
            assert_eq!(watermark(&kernel), before);

            // A real tag edit: the tail advances the watermark to the new
            // col_mod, so the bump never reads as drift.
            let hit = kernel
                .update_note_tags(vec![nid], None, vec!["fresh".into()], Vec::new())
                .await
                .unwrap();
            assert_eq!(hit.notes_modified, 1);
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
            assert_eq!(watermark(&kernel), Some(kernel.col_mod().await.unwrap()));
            assert!(!kernel.reindex_if_needed().await.unwrap());

            kernel.close().await.unwrap();
            std::fs::remove_dir_all(dir).ok();
        });
    }

    #[test]
    fn note_type_ops_run_as_kernel_ops() {
        // The #391 re-home, long-tail group 3: the metadata-tail ops advance
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

        crate::runtime::block_on(async {
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
            assert!(embedder.0.load(Ordering::SeqCst) > embeds_before);
            assert!(kernel.index().engine().contains(nid));
            assert!(!kernel.reindex_if_needed().await.unwrap());

            kernel.close().await.unwrap();
            std::fs::remove_dir_all(dir).ok();
        });
    }

    #[test]
    fn kernel_ops_reopen_after_cooperative_release() {
        // The #64 open-on-demand, kernel-side: an idle release between ops
        // (or between one op's jobs) self-heals on the next serialized job
        // instead of erroring CollectionNotOpen.
        crate::runtime::block_on(async {
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
            std::fs::remove_dir_all(dir).ok();
        });
    }

    #[test]
    fn tag_centroids_build_and_never_leak_into_note_search() {
        // The #178/#179 layer end to end: tagged upserts → centroids in the
        // tag.text space (hygiene-filtered) → note searches structurally
        // blind to tag keys.
        crate::runtime::block_on(async {
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
            let engine = kernel.index().engine();
            assert!(engine.modality_get(TAG_TEXT_SPACE, cell_key).is_some());

            // A note search never surfaces a tag key.
            let hits = kernel.search("note number", 20).await.unwrap();
            assert!(hits.iter().all(|h| keys.lookup(h.note_id).is_none()));

            // A tag-only edit (rename) rides metadata_changed, not an index
            // op — it must also schedule a refresh so the key map tracks
            // the new name (#445).
            kernel
                .collection()
                .run(|core| core.rename_tag("bio::cell", "bio::organelle", &[]))
                .await
                .unwrap()
                .unwrap();
            kernel.metadata_changed().await.unwrap();
            wait_for_tags(&kernel, |k| {
                k.lookup(tag_centroids::tag_key("bio::organelle")).is_some()
                    && k.lookup(tag_centroids::tag_key("bio::cell")).is_none()
            })
            .await;
            let cell_key = tag_centroids::tag_key("bio::organelle");

            // Deleting a member triggers the refresh through the delete
            // tail's membership probe (#445): the tag falls below
            // min_members and its centroid retires.
            let victim = kernel.tag_keys().members(cell_key)[0];
            kernel.delete_notes(vec![victim]).await.unwrap();
            wait_for_tags(&kernel, |k| k.lookup(cell_key).is_none()).await;

            kernel.close().await.unwrap();
            std::fs::remove_dir_all(dir).ok();
        });
    }

    #[test]
    fn tag_signal_surfaces_off_topic_members() {
        // The #179 payoff: a member whose own text doesn't match the query
        // still surfaces because its TAG's centroid (dominated by on-topic
        // siblings) activates and expands.
        crate::runtime::block_on(async {
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
            std::fs::remove_dir_all(dir).ok();
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
        /// present — the #386 TOCTOU-delete / transient-read shape. A
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
        // The #228 end-to-end: one recognition pass per image feeds the
        // lexical store (rows + segments) AND the text-space vector, so a
        // query matching only the text INSIDE an image surfaces the note.
        crate::runtime::block_on(async {
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
            std::fs::remove_dir_all(dir).ok();
        });
    }

    #[test]
    fn recognition_skips_unreadable_media_and_keeps_it_pending() {
        // #386: exists() says present but read() returns None (TOCTOU
        // delete, transient error) — the item must be SKIPPED, never
        // recognized over empty bytes, and must stay pending so a later
        // sweep (once the read heals) recognizes the real bytes.
        crate::runtime::block_on(async {
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
            std::fs::remove_dir_all(dir).ok();
        });
    }

    #[test]
    fn unreadable_prefix_reports_no_progress_for_the_driver() {
        // #386 livelock shape: total_pending > max_items with a permanently
        // unreadable PREFIX of the pending order. Skipped items stay pending,
        // so each call re-takes the same window — the kernel can't drain it.
        // The report must let a driver detect the no-progress batch
        // (recognized == 0 with remaining > 0) and stop instead of spinning.
        crate::runtime::block_on(async {
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

            // The batch window covers only the unreadable prefix: nothing is
            // sent, nothing stored, and the readable tail stays beyond the
            // window — the exact shape a driver must stop on, since the next
            // call would re-take the identical window.
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

            // Healed reads drain normally across batches, then go idle.
            resolver.unreadable.lock().unwrap().clear();
            let healed = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(
                healed,
                recognize::SweepReport::Ran {
                    recognized: 2,
                    stored: 2,
                    remaining: 1
                }
            );
            let tail = kernel.recognize_pending(2).await.unwrap();
            assert!(matches!(
                tail,
                recognize::SweepReport::Ran {
                    recognized: 1,
                    remaining: 0,
                    ..
                }
            ));
            let idle = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(idle, recognize::SweepReport::Idle);

            kernel.close().await.unwrap();
            std::fs::remove_dir_all(dir).ok();
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
        // #416: an item the gate drops gets a below-gate marker, so it is
        // recognized ONCE — not re-OCR'd every sweep — and only a recognizer
        // fingerprint change (engine upgrade) puts it back in the pending
        // set, exactly like stored rows re-derive.
        crate::runtime::block_on(async {
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
            std::fs::remove_dir_all(dir).ok();
        });
    }

    #[test]
    fn all_gated_window_converges_across_batches() {
        // The #416 residual livelock shape: a permanently-gated prefix wider
        // than the batch window. Markers make each batch's gated items DONE,
        // so successive windows advance and the sweep converges to idle —
        // instead of re-taking the identical window forever (which the #413
        // no-progress stop, keyed on recognized == 0, deliberately does not
        // terminate).
        crate::runtime::block_on(async {
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

            // Window 1: all recognized, all gated — real work (recognized > 0,
            // so the driver keeps going), and the window is now done.
            let first = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(
                first,
                recognize::SweepReport::Ran {
                    recognized: 2,
                    stored: 0,
                    remaining: 1
                }
            );

            // Window 2 ADVANCES past the marked items and drains the tail.
            let second = kernel.recognize_pending(2).await.unwrap();
            assert!(matches!(
                second,
                recognize::SweepReport::Ran {
                    recognized: 1,
                    remaining: 0,
                    ..
                }
            ));

            // Convergence: idle, with each item judged exactly once.
            let idle = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(idle, recognize::SweepReport::Idle);
            assert_eq!(
                recognizer
                    .items_seen
                    .load(std::sync::atomic::Ordering::SeqCst),
                3
            );

            kernel.close().await.unwrap();
            std::fs::remove_dir_all(dir).ok();
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
        // #485: OCR and VLM-describe attach as INDEPENDENT purposes over one
        // image. OCR lands in source "ocr" (lexical + vector, bit-identical
        // to the single-slot sweep). Describe lands in source "vlm",
        // VECTOR-ONLY: a vector mints (a non-literal query surfaces the note)
        // but the describe prose is NEVER reachable via substring/fuzzy.
        crate::runtime::block_on(async {
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
            // fuzzy (the VectorOnly destination — docs/decisions.md).
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
            std::fs::remove_dir_all(dir).ok();
        });
    }

    #[test]
    fn open_upsert_search_close_without_python() {
        // The harness picks the runtime: here futures' minimal block_on —
        // no tokio, nothing owned by the kernel.
        crate::runtime::block_on(assert_send(smoke()));
    }

    async fn smoke() {
        let dir = temp_dir();
        let col = dir.join("collection.anki2");
        let cache = dir.join("cache");
        // The harness assembles the scheduling: here the thread-free
        let kernel = Kernel::open(col.to_str().unwrap(), cache.to_str().unwrap())
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
        let kernel2 = Kernel::open(col.to_str().unwrap(), cache.to_str().unwrap())
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
    /// #471: the derived rebuild's collect→commit window vs concurrent
    /// writes. The first half DEMONSTRATES the hazard mechanically (a build
    /// over a stale snapshot erases a newer note's derived rows — why the
    /// verify exists); the second half pins that `rebuild_derived` converges
    /// (post-commit col_mod equals the committed snapshot) and restores them.
    #[test]
    fn derived_rebuild_verifies_against_mid_build_writes() {
        crate::runtime::block_on(async {
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
                .run(|core| -> shrike_ffi::NativeResult<_> {
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
            assert!(
                !kernel
                    .derived
                    .search_substring("oxaloacetate", 5, None, &[])
                    .unwrap()
                    .unwrap()
                    .is_empty(),
                "B's ingest landed"
            );
            // …and commit the stale build: B's rows are erased (the hazard).
            kernel.derived.build(&stale.0, stale.1).unwrap();
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
}
