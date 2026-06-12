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
//! (anki's own lazy runtime is never instantiated on Shrike's call paths;
//! see [`runtime`].)
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
pub mod fusion;
pub mod index_orchestrator;
pub mod recognize;
pub mod tag_centroids;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};

use futures::channel::oneshot;
#[cfg(test)]
use futures::future::BoxFuture;
use tracing::Instrument;

use shrike_collection::{CollectionCore, CreateOutcome, DuplicatePolicy};
use shrike_derived::DerivedEngine;
use shrike_ffi::{NativeError, NativeResult};
use shrike_index::MultiModalIndex;

pub mod runtime;
pub use runtime::{block_on, init_runtime, spawn_op};

// The engine contract (#342): traits live in shrike-engine-api — the kernel
// consumes them and re-exports for downstream paths; it names no engine.
pub use shrike_engine_api::{
    Embedder, ImageEmbedder, ImageResolver, Locator, MediaItem, Recognition, Recognizer, Segment,
};

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
/// compete. anki's internal `block_on` exists only on sync/AnkiWeb service
/// paths Shrike never calls (pinned in shrike-collection), so no
/// nested-runtime hazard exists on our call paths.
pub struct SerializedCollection {
    core: Arc<CollectionCore>,
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
        job: impl FnOnce(&CollectionCore) -> T + Send + 'static,
    ) -> NativeResult<T> {
        let core = Arc::clone(&self.core);
        let (tx, rx) = oneshot::channel();
        self.sender()?
            .send(Box::new(move || {
                let _ = tx.send(core.ensure_open().map(|_| job(&core)));
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
    derived: Arc<DerivedEngine>,
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
    /// The recognition service (#228/#342, the second registry slot):
    /// OCR/ASR engines the harness attaches at runtime, exactly like the
    /// embed slot — the kernel runs the pipeline over whatever is registered
    /// and recognition is simply off when nothing is.
    recognize: RwLock<Option<Arc<RecognizeService>>>,
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
        std::fs::create_dir_all(cache_dir)
            .map_err(|e| NativeError::internal(format!("cache dir: {e}")))?;
        let collection = Arc::new(SerializedCollection::open(collection_path.to_string()).await?);
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
        let saver = index_orchestrator::DebouncedSaver::new(
            Arc::clone(&orchestrator),
            save_delay.unwrap_or(index_orchestrator::DEFAULT_SAVE_DELAY),
            save_threshold.unwrap_or(index_orchestrator::DEFAULT_SAVE_THRESHOLD),
        );
        let derived = Arc::new(DerivedEngine::open(
            &format!("{}/shrike.db", cache_dir.trim_end_matches('/')),
            DerivedEngine::SCHEMA_VERSION,
        )?);
        tracing::debug!(collection = collection_path, "kernel opened");
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
            recognize: RwLock::new(None),
            recognition_gate: recognize::RecognitionGate::default(),
        })
    }

    /// Attach (or swap) the recognition service — the #342 slot pattern,
    /// second instance. The harness follows up by driving the pending sweep.
    pub fn attach_recognizer(
        &self,
        recognizer: Arc<dyn Recognizer>,
        resolver: Arc<dyn ImageResolver>,
    ) {
        *self.recognize.write().expect("recognize slot poisoned") =
            Some(Arc::new(RecognizeService {
                recognizer,
                resolver,
            }));
    }

    /// Detach the recognition service. Already-derived text stays (it remains
    /// valid output of the engine that produced it); only new recognition
    /// stops.
    pub fn detach_recognizer(&self) {
        *self.recognize.write().expect("recognize slot poisoned") = None;
    }

    /// The currently attached recognition service, if any.
    pub fn recognize_service(&self) -> Option<Arc<RecognizeService>> {
        self.recognize
            .read()
            .expect("recognize slot poisoned")
            .clone()
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
    /// store's gated recognized texts (#199/#228): the index derives from
    /// BOTH, so reconcile == rebuild keeps holding after recognition, and a
    /// note's OCR text mints vectors on any (re-)embed path. Vector-worthiness
    /// re-judges from the stored text (confidence already gated at ingest).
    fn compose_embed_inputs(
        &self,
        raw: Vec<(i64, String, Vec<String>)>,
        only_notes: Option<&[i64]>,
    ) -> Vec<index_orchestrator::EmbedInput> {
        let mut ocr_map: std::collections::HashMap<i64, Vec<String>> =
            std::collections::HashMap::new();
        // Per-op callers scope the read to the written notes (#445) — the
        // full-set read is for rebuild/reconcile, which consume everything.
        let texts = match only_notes {
            Some(ids) => self.derived.texts_for_source_for_notes(OCR_SOURCE, ids),
            None => self.derived.texts_for_source(OCR_SOURCE),
        };
        match texts {
            Ok(rows) => {
                for (nid, _r, text) in rows {
                    if self.recognition_gate.vector_worthy(&text) {
                        ocr_map.entry(nid).or_default().push(text);
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = ?e, "reading recognized texts failed; embedding without them");
            }
        }
        raw.into_iter()
            .map(
                |(note_id, text, image_names)| index_orchestrator::EmbedInput {
                    note_id,
                    text,
                    image_names,
                    ocr_texts: ocr_map.remove(&note_id).unwrap_or_default(),
                },
            )
            .collect()
    }

    /// One bounded recognition sweep (#228): recognize up to `max_items`
    /// pending (note, image) pairs, persist gated text + segments to the
    /// derived store, and re-embed the affected notes so OCR vectors mint.
    /// Pending = a resolvable image with no OCR row AND no below-gate marker
    /// (#416) — or all of them, after a recognizer-fingerprint change
    /// invalidates the prior text and the markers together. Returns
    /// `{status, recognized, stored, remaining}` — the harness loops while
    /// `remaining > 0` and the batch made progress (`recognized > 0`), so
    /// one call never occupies the executor for long and a permanently
    /// unreadable window can't spin the driver.
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

    pub async fn recognize_pending(&self, max_items: usize) -> NativeResult<serde_json::Value> {
        let Some(svc) = self.recognize_service() else {
            return Ok(serde_json::json!({"status": "unavailable"}));
        };

        // Fingerprint drift: a changed engine invalidates ALL recognized text
        // (the analog of the embedder's model_id rebuild).
        let fingerprint = svc.recognizer.fingerprint().unwrap_or_default();
        let stored = self
            .derived
            .meta_get(RECOGNIZER_FINGERPRINT_KEY)?
            .unwrap_or_default();
        if !stored.is_empty() && stored != fingerprint {
            let stale: Vec<i64> = self
                .derived
                .refs_for_source(OCR_SOURCE)?
                .into_iter()
                .map(|(nid, _)| nid)
                .collect();
            if !stale.is_empty() {
                tracing::info!(
                    notes = stale.len(),
                    "recognizer fingerprint changed; invalidating recognized text"
                );
                self.derived.remove(&stale, Some(OCR_SOURCE))?;
            }
            // Below-gate markers ride the same invalidation (#416): the new
            // engine may read what the old one couldn't, so gated items
            // re-enter the pending set exactly like stored rows re-derive.
            self.derived.clear_gated(OCR_SOURCE)?;
        }

        // Pending set: resolvable images without an OCR row — and without a
        // below-gate marker (#416): an item the gate dropped is DONE (its
        // outcome can't change until the fingerprint does), not pending, so
        // it is never re-recognized and an all-gated window converges instead
        // of re-taking itself forever.
        let raw = self
            .collection
            .run(|core| -> NativeResult<_> {
                let ids = core.find_notes("")?;
                core.note_embed_inputs(&ids)
            })
            .await??;
        let mut done: std::collections::HashSet<(i64, String)> = self
            .derived
            .refs_for_source(OCR_SOURCE)?
            .into_iter()
            .collect();
        done.extend(self.derived.gated_refs_for_source(OCR_SOURCE)?);
        let mut pending: Vec<(i64, String)> = Vec::new();
        for (note_id, _text, image_names) in &raw {
            for name in image_names {
                if !done.contains(&(*note_id, name.clone())) && svc.resolver.exists(name) {
                    pending.push((*note_id, name.clone()));
                }
            }
        }
        let total_pending = pending.len();
        pending.truncate(max_items);
        if pending.is_empty() {
            self.derived
                .meta_set(RECOGNIZER_FINGERPRINT_KEY, &fingerprint)?;
            return Ok(serde_json::json!({
                "status": "idle", "recognized": 0, "stored": 0, "remaining": 0
            }));
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
                        image = %name,
                        note_id,
                        "media read failed after exists(); skipping until the next sweep"
                    );
                }
            }
        }
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
        let mut existing: std::collections::HashMap<i64, Vec<(String, String)>> =
            std::collections::HashMap::new();
        for (nid, r, text) in self.derived.texts_for_source(OCR_SOURCE)? {
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
            self.derived.ingest(*note_id, OCR_SOURCE, refs_text)?;
        }
        for (note_id, name, json) in &segments {
            self.derived
                .put_segments(*note_id, OCR_SOURCE, name, json)?;
        }
        self.derived.mark_gated(OCR_SOURCE, &gated)?;
        self.derived
            .meta_set(RECOGNIZER_FINGERPRINT_KEY, &fingerprint)?;

        // Re-embed the affected notes: their hash now folds the OCR text, so
        // this mints the vectors and the next reconcile sees them current.
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
        Ok(serde_json::json!({
            "status": "ran",
            "recognized": sent.len(),
            "stored": stored_count,
            "remaining": total_pending.saturating_sub(pending.len()),
        }))
    }

    async fn index_written(&self, written: &[i64]) -> NativeResult<()> {
        if written.is_empty() {
            return Ok(());
        }
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
                        Some(&engine),
                        Some(&derived),
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
            kernel
                .upsert_notes_json(
                    serde_json::json!(notes).to_string(),
                    "error".to_string(),
                    false,
                )
                .await
                .unwrap();

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
            let diagram_id = results[0]["id"].as_i64().unwrap();

            // No recognizer attached → the sweep reports unavailable.
            let off = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(off["status"], "unavailable");

            let mut media = std::collections::HashMap::new();
            media.insert(
                "krebs.png".to_string(),
                b"the krebs cycle produces energy carriers".to_vec(),
            );
            kernel.attach_recognizer(Arc::new(StubRecognizer), Arc::new(MapResolver::new(media)));

            let report = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(report["status"], "ran");
            assert_eq!(report["stored"], 1);
            assert_eq!(report["remaining"], 0);

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
            assert_eq!(again["status"], "idle");

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
            kernel
                .upsert_notes_json(
                    serde_json::json!(notes).to_string(),
                    "error".to_string(),
                    false,
                )
                .await
                .unwrap();

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
            assert_eq!(report["status"], "ran");
            assert_eq!(report["recognized"], 1);
            assert_eq!(report["stored"], 1);
            assert_eq!(report["remaining"], 0);
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
            assert_eq!(again["status"], "ran");
            assert_eq!(again["recognized"], 0);
            assert_eq!(again["stored"], 0);

            // Heal the read: the next sweep recognizes the REAL bytes.
            resolver.unreadable.lock().unwrap().clear();
            let healed = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(healed["status"], "ran");
            assert_eq!(healed["recognized"], 1);
            assert_eq!(healed["stored"], 1);
            let hits = kernel.search("flaky secret", 5).await.unwrap();
            assert!(
                hits.iter()
                    .any(|h| h.signals.iter().any(|(sig, _)| sig == "exact")),
                "healed item recognized and stored"
            );

            // And now everything is done.
            let idle = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(idle["status"], "idle");

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
            kernel
                .upsert_notes_json(
                    serde_json::json!(notes).to_string(),
                    "error".to_string(),
                    false,
                )
                .await
                .unwrap();

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
            assert_eq!(report["status"], "ran");
            assert_eq!(report["recognized"], 0);
            assert_eq!(report["stored"], 0);
            assert_eq!(report["remaining"], 1);

            // And it really would: a second call is byte-identical.
            let again = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(again, report);

            // Healed reads drain normally across batches, then go idle.
            resolver.unreadable.lock().unwrap().clear();
            let healed = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(healed["recognized"], 2);
            assert_eq!(healed["stored"], 2);
            assert_eq!(healed["remaining"], 1);
            let tail = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(tail["recognized"], 1);
            assert_eq!(tail["remaining"], 0);
            let idle = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(idle["status"], "idle");

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
            kernel
                .upsert_notes_json(
                    serde_json::json!(notes).to_string(),
                    "error".to_string(),
                    false,
                )
                .await
                .unwrap();

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
            assert_eq!(report["status"], "ran");
            assert_eq!(report["recognized"], 2);
            assert_eq!(report["stored"], 1);
            assert_eq!(report["remaining"], 0);
            let seen = || {
                recognizer
                    .items_seen
                    .load(std::sync::atomic::Ordering::SeqCst)
            };
            assert_eq!(seen(), 2);

            // Sweep 2: idle — the gated item is DONE, not re-judged (the
            // recognizer is never called again).
            let again = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(again["status"], "idle");
            assert_eq!(seen(), 2, "the gated item must not re-OCR");

            // Engine upgrade: a fingerprint change invalidates rows AND
            // markers, so BOTH items re-derive.
            *recognizer.fingerprint.lock().unwrap() = "stub:v2".to_string();
            let rederived = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(rederived["status"], "ran");
            assert_eq!(rederived["recognized"], 2);
            assert_eq!(rederived["stored"], 1);
            assert_eq!(seen(), 4);

            // And the new outcome sticks: idle again, no further calls.
            let idle = kernel.recognize_pending(10).await.unwrap();
            assert_eq!(idle["status"], "idle");
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
            kernel
                .upsert_notes_json(
                    serde_json::json!(notes).to_string(),
                    "error".to_string(),
                    false,
                )
                .await
                .unwrap();

            let mut media = std::collections::HashMap::new();
            for name in ["t1.png", "t2.png", "t3.png"] {
                media.insert(name.to_string(), b"no".to_vec()); // all below-gate
            }
            let recognizer = Arc::new(CountingRecognizer::new("stub:v1"));
            kernel.attach_recognizer(recognizer.clone(), Arc::new(MapResolver::new(media)));

            // Window 1: all recognized, all gated — real work (recognized > 0,
            // so the driver keeps going), and the window is now done.
            let first = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(first["status"], "ran");
            assert_eq!(first["recognized"], 2);
            assert_eq!(first["stored"], 0);
            assert_eq!(first["remaining"], 1);

            // Window 2 ADVANCES past the marked items and drains the tail.
            let second = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(second["status"], "ran");
            assert_eq!(second["recognized"], 1);
            assert_eq!(second["remaining"], 0);

            // Convergence: idle, with each item judged exactly once.
            let idle = kernel.recognize_pending(2).await.unwrap();
            assert_eq!(idle["status"], "idle");
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
                    .search_substring("oxaloacetate", 5, None)
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
                    .search_substring("oxaloacetate", 5, None)
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
                    .search_substring("oxaloacetate", 5, None)
                    .unwrap()
                    .unwrap()
                    .is_empty(),
                "the rebuild restored B's rows"
            );
            kernel.close().await.unwrap();
        });
    }
}
