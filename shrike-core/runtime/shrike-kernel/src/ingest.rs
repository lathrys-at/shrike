//! The persistent ingest actor: the single writer of the vector index, the
//! derived store, and the watermarks.
//!
//! # One consumer, one total order
//!
//! Under the single-user assumption (write-vs-write overlap is rare, intra-op
//! parallelism on bulk ops is the target), **every** index/derived mutation
//! funnels through one persistent task ([`drain_loop`]) over one FIFO channel.
//! The task is the *only* code that touches the stores, so serialization is
//! structural — there is no lock, because there is no second writer to exclude:
//!
//! - the rebuild's snapshot→build→prune can never interleave with a concurrent
//!   sweep's ingest (#828);
//! - a recognition vector add can never interleave with a reindex
//!   (#650/#628/#644);
//! - the kernel `DerivedEngine` is the file's sole writer (the two-writer
//!   `SQLITE_BUSY` class).
//!
//! The race family dissolves by construction — not by retrofitted
//! reconciliation (the old in-flight-token machinery is gone, [`watermark`]).
//!
//! # The embed queue decouples the fast write from the slow embed
//!
//! A maintained collection write commits, enqueues `{ids, captured col.mod,
//! kind}` INSIDE its collection-actor job ([`IngestHandle::enqueue`]), and
//! returns immediately — it never waits on a multi-second remote/multimodal
//! embed, so a rare concurrent writer never blocks. The consumer drains queued
//! items in batches: re-read note content → embed on `drive_compute` → atomic
//! per-note index add → derived ingest → advance the watermark.
//!
//! Bulk ops (reindex / rebuild / recognition store) ride the SAME channel as
//! awaited [`IngestMsg::Job`]s, so they serialize with the hot path in one FIFO
//! order — the property Theme C's readiness barrier ("queue drained + no
//! rebuild pending") stands on.
//!
//! # The load-bearing ordering invariant
//!
//! The enqueue happens **inside the collection-write job**, so queue order ==
//! `col.mod` order (`col.mod` is monotonic; the collection actor is FIFO). The
//! FIFO consumer then processes items in `col.mod` order, so each drained batch
//! is a contiguous `col.mod` prefix and the watermark advance is a linear pass
//! (see [`watermark`]). An enqueue that slipped into a post-`await`
//! *continuation* would decouple queue order from `col.mod` order and silently
//! re-open the over-certification bug. State and keep it: **enqueue in the write
//! job, never after.**
//!
//! # Drain-merge
//!
//! Re-read-at-drain reflects a note's state *at drain*: two edits coalesce to
//! one embed (last-writer-wins); an id queued but **absent** at re-read ⇒ remove
//! its vectors. Deletes carry their ids as [`MaintKind::Remove`] and skip the
//! embed entirely. Within one drained batch the FIFO last-touching kind per id
//! wins, so {upsert, reindex, migrate} × {delete} resolves to one add set + one
//! remove set.
//!
//! # No new deadlock
//!
//! The actor orchestrates on `drive_io` (async) and awaits collection reads
//! (`drive_sync`) and embeds (`drive_compute`) as leaves — the existing
//! read→compute→write pattern, so the leaf-invariant ([`runtime`]) holds. A job
//! running on the actor never enqueues-and-awaits further ingest work (the
//! ingest analog of the pool leaf-invariant).
//!
//! # Durability + shutdown
//!
//! The durable index/FTS write strictly precedes the durable watermark advance
//! (`save()` orders engine → hashes → meta; `set_col_mod` is a synchronous
//! SQLite write the actor issues only after the in-memory add). Shutdown drains
//! the queue (channel close ⇒ `recv()` returns `None` only once the buffer is
//! empty), then flushes the index savers durably, before the kernel closes the
//! collection.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, RwLock};

use futures::future::BoxFuture;
use futures::FutureExt;
use shrike_error::{NativeError, NativeResult};
use shrike_store::DerivedStore;
use tokio::sync::{mpsc, oneshot};

use crate::embed_set::EmbedSpaces;
use crate::index_orchestrator::{self, EmbedInput};
use crate::index_set::IndexSet;
use crate::recognize::RecognitionGate;
use crate::runtime;
use crate::tag_centroids::{self, TagCentroidConfig, TagKeyMap, TagRefresher};
use crate::watermark::WatermarkFloors;
use crate::{EmbedService, SerializedCollection, FIELD_SOURCE};
use shrike_engine_api::{Embedder, ImageEmbedder, ImageResolver};

/// What a maintenance item asks the writer to do for its ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaintKind {
    /// Re-read the ids' content and index them: present ⇒ atomic per-note add +
    /// derived ingest, absent ⇒ remove. The upsert / reindex / migrate path.
    Maintain,
    /// Remove the ids' vectors + derived rows directly (no embed). The delete /
    /// forget / prune path.
    Remove,
    /// No ids — advance the watermark only (a metadata-only `col.mod` bump that
    /// touches no embedding text or derived row).
    AdvanceOnly,
}

/// One unit of maintenance work, enqueued INSIDE the collection-write job so
/// queue order == `col.mod` order.
#[derive(Debug, Clone)]
pub struct IngestItem {
    /// The created/updated/deleted note ids (empty for [`MaintKind::AdvanceOnly`]).
    pub ids: Vec<i64>,
    /// The `col.mod` captured with the collection write that produced this item.
    pub col_mod: i64,
    /// What to do with `ids`.
    pub kind: MaintKind,
    /// Whether tag membership could have changed (drives the coalesced tag
    /// refresh; see [`Ingestor::process_batch`]).
    pub membership_may_have_changed: bool,
}

/// The recognition-store payload: a sweep's gated text rows + segments + below-
/// gate markers, plus the notes to re-embed. Routed through the actor so the
/// derived write + re-embed serialize with rebuild (closing #828).
#[derive(Debug, Clone, Default)]
pub struct RecognitionWrite {
    /// The derived `source` these rows land under (`ocr`/`asr`/…).
    pub source: String,
    /// `note_id → (ref, text)` rows to ingest (already merged with existing rows).
    pub touched: Vec<(i64, Vec<(String, String)>)>,
    /// `(note_id, ref, segments-json)` per-segment recognition structure.
    pub segments: Vec<(i64, String, String)>,
    /// Below-gate `(note_id, ref)` markers (judged-once bookkeeping).
    pub gated: Vec<(i64, String)>,
    /// The notes whose vectors must re-mint (their hash now folds recognized text).
    pub affected: Vec<i64>,
}

/// A message on the actor channel.
enum IngestMsg {
    /// A fire-and-forget maintenance item (the hot path).
    Item(IngestItem),
    /// An awaited bulk op: a closure run on the actor task with the writer.
    Job(Box<dyn FnOnce(Arc<Ingestor>) -> BoxFuture<'static, ()> + Send>),
    /// Resolve once every message enqueued before this barrier has been
    /// processed (boot readiness, tests).
    Flush(oneshot::Sender<()>),
}

/// The kernel-held handle to the ingest actor: the FIFO sender + the drain task.
/// Cloned senders ride collection-write job closures (sync `send` on an
/// unbounded channel never blocks).
pub struct IngestHandle {
    tx: Mutex<Option<mpsc::UnboundedSender<IngestMsg>>>,
    task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// Shared with the [`Ingestor`] the drain task owns: the count of contained
    /// drain panics, so the kernel (which holds only this handle) can report a
    /// degraded sole writer on `/status`.
    drain_panics: Arc<std::sync::atomic::AtomicU64>,
}

/// A cheap, cloneable enqueue handle captured INTO a collection-write job, so
/// the enqueue happens inside the job (queue order == `col.mod` order — the
/// load-bearing invariant). Build one with [`IngestHandle::enqueuer`] *before*
/// the job and move it in; call [`IngestEnqueuer::enqueue`] after the write
/// commits and `col.mod` is read, still inside the same job closure.
#[derive(Clone)]
pub struct IngestEnqueuer(Option<mpsc::UnboundedSender<IngestMsg>>);

impl IngestEnqueuer {
    /// Enqueue a maintenance item (no-op after shutdown).
    pub fn enqueue(&self, item: IngestItem) {
        if let Some(tx) = &self.0 {
            let _ = tx.send(IngestMsg::Item(item));
        }
    }
}

impl IngestHandle {
    /// A live sender clone, or `None` after shutdown.
    fn sender(&self) -> Option<mpsc::UnboundedSender<IngestMsg>> {
        self.tx.lock().expect("ingest sender poisoned").clone()
    }

    /// A cloneable enqueue handle to move into a collection-write job.
    ///
    /// # Panics
    ///
    /// Panics if the sender mutex is poisoned (a prior holder panicked).
    pub fn enqueuer(&self) -> IngestEnqueuer {
        IngestEnqueuer(self.tx.lock().expect("ingest sender poisoned").clone())
    }

    /// Enqueue a maintenance item. Fire-and-forget: returns immediately, the
    /// actor processes it. A dropped item (post-shutdown) is the already-closed
    /// outcome — the next boot's drift reconcile heals.
    pub fn enqueue(&self, item: IngestItem) {
        if let Some(tx) = self.sender() {
            let _ = tx.send(IngestMsg::Item(item));
        }
    }

    /// Send an awaited bulk job to the actor and return its result. The closure
    /// runs on the actor task (so it serializes with the hot path) and sends its
    /// typed result back over the per-call oneshot.
    async fn job<T, F>(&self, build: F) -> NativeResult<T>
    where
        T: Send + 'static,
        F: FnOnce(Arc<Ingestor>) -> BoxFuture<'static, NativeResult<T>> + Send + 'static,
    {
        let (done_tx, done_rx) = oneshot::channel();
        let msg = IngestMsg::Job(Box::new(move |ing: Arc<Ingestor>| {
            Box::pin(async move {
                let r = build(ing).await;
                let _ = done_tx.send(r);
            })
        }));
        self.sender()
            .ok_or_else(|| NativeError::internal("the ingest actor is gone"))?
            .send(msg)
            .map_err(|_| NativeError::internal("the ingest actor is gone"))?;
        done_rx
            .await
            .map_err(|_| NativeError::internal("the ingest actor dropped a job"))?
    }

    /// Whole-collection drift reconcile of the index. See
    /// [`Ingestor::reindex_if_needed`].
    ///
    /// # Errors
    ///
    /// Returns an error if the actor is gone or the reconcile fails.
    pub async fn reindex_if_needed(&self) -> NativeResult<bool> {
        self.job(|ing| Box::pin(async move { ing.reindex_if_needed().await }))
            .await
    }

    /// Explicit full index rebuild. See [`Ingestor::rebuild_index`].
    ///
    /// # Errors
    ///
    /// Returns an error if the actor is gone, no embedder is attached, or the
    /// rebuild fails.
    pub async fn rebuild_index(&self) -> NativeResult<usize> {
        self.job(|ing| Box::pin(async move { ing.rebuild_index().await }))
            .await
    }

    /// Full derived (FTS5) rebuild. See [`Ingestor::rebuild_derived`].
    ///
    /// # Errors
    ///
    /// Returns an error if the actor is gone or the rebuild fails.
    pub async fn rebuild_derived(&self) -> NativeResult<(usize, i64)> {
        self.job(|ing| Box::pin(async move { ing.rebuild_derived().await }))
            .await
    }

    /// Refresh tag centroids synchronously on the actor (the attach path's
    /// follow-up). See [`Ingestor::refresh_tag_centroids`].
    ///
    /// # Errors
    ///
    /// Returns an error if the actor is gone or the recompute fails.
    pub async fn refresh_tag_centroids(&self) -> NativeResult<usize> {
        self.job(|ing| Box::pin(async move { ing.refresh_tag_centroids().await }))
            .await
    }

    /// Persist a recognition sweep's text + segments and re-embed the affected
    /// notes, all on the actor. See [`Ingestor::store_recognition`].
    ///
    /// # Errors
    ///
    /// Returns an error if the actor is gone or a derived/index write fails.
    pub async fn store_recognition(&self, write: RecognitionWrite) -> NativeResult<()> {
        self.job(move |ing| Box::pin(async move { ing.store_recognition(write).await }))
            .await
    }

    /// Await the queue drained to the current point (every message enqueued
    /// before this call has been processed). The boot-readiness / test barrier.
    pub async fn flush(&self) {
        let Some(tx) = self.sender() else {
            return;
        };
        let (done_tx, done_rx) = oneshot::channel();
        if tx.send(IngestMsg::Flush(done_tx)).is_err() {
            return;
        }
        let _ = done_rx.await;
    }

    /// Drain the queue, flush the index savers durably, and join the drain
    /// task. Idempotent. Called by `Kernel::close` before the collection closes.
    ///
    /// # Panics
    ///
    /// Panics if the sender or task mutex is poisoned (a prior holder panicked).
    pub async fn shutdown(&self) {
        drop(self.tx.lock().expect("ingest sender poisoned").take());
        let handle = self.task.lock().expect("ingest task poisoned").take();
        if let Some(handle) = handle {
            let _ = handle.await;
        }
    }

    /// How many drained items/jobs the drain loop caught panicking. Zero in
    /// normal operation; non-zero signals a degraded sole writer (the `/status`
    /// observability hook for [`drain_loop`]'s panic boundary).
    pub fn drain_panics(&self) -> u64 {
        self.drain_panics
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// The single writer: owns the watermark floors and every index/derived
/// mutation path. All shared sub-stores are `Arc`s the kernel also holds for its
/// read paths (search/status) — writes run only here (on the actor task), reads
/// run concurrently against the same engine. Every method runs on the actor
/// task, so the floors need no cross-thread guard beyond the (uncontended)
/// `Mutex` that keeps the borrow off the `await` points.
pub struct Ingestor {
    collection: Arc<SerializedCollection>,
    index_set: Arc<IndexSet>,
    derived: Arc<dyn DerivedStore>,
    embed: Arc<RwLock<EmbedSpaces>>,
    recognition_gate: RecognitionGate,
    tag_keys: Arc<TagKeyMap>,
    tag_config: TagCentroidConfig,
    tag_refresh: Arc<TagRefresher>,
    floors: Mutex<WatermarkFloors>,
    /// Count of drained items/jobs whose processing PANICKED and was contained
    /// by the drain loop ([`drain_loop`]). Non-zero means the sole writer hit an
    /// unexpected fault (most likely a poisoned lock) and skipped that work —
    /// the affected notes are un-indexed until a reconcile/rebuild heals them.
    /// `Arc`-shared with the [`IngestHandle`] so `/status` can read it (the
    /// drain task owns the `Ingestor`, the kernel holds only the handle).
    drain_panics: Arc<std::sync::atomic::AtomicU64>,
}

impl Ingestor {
    /// Construct the writer unit over the kernel's shared sub-stores.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        collection: Arc<SerializedCollection>,
        index_set: Arc<IndexSet>,
        derived: Arc<dyn DerivedStore>,
        embed: Arc<RwLock<EmbedSpaces>>,
        recognition_gate: RecognitionGate,
        tag_keys: Arc<TagKeyMap>,
        tag_config: TagCentroidConfig,
        tag_refresh: Arc<TagRefresher>,
    ) -> Arc<Self> {
        Arc::new(Self {
            collection,
            index_set,
            derived,
            embed,
            recognition_gate,
            tag_keys,
            tag_config,
            tag_refresh,
            floors: Mutex::new(WatermarkFloors::default()),
            drain_panics: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    /// The primary embedding service, or `None` when no embedder is attached.
    fn embed_service(&self) -> Option<Arc<EmbedService>> {
        self.embed.read().expect("embed slot poisoned").primary()
    }

    /// The SEPARATE image-primary write route (a dedicated CLIP secondary), or
    /// `None` when the text-primary already writes images (omni / N=1).
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
        let image_key = image_key?;
        if Some(&image_key) == text_key.as_ref() {
            return None;
        }
        let orch = self.index_set.orchestrator_for(&image_key)?;
        Some((orch, image_svc))
    }

    /// Compose orchestrator inputs from collection rows + the derived store's
    /// recognized texts across every vector-minting source. Delegates to the
    /// shared [`compose_embed_inputs`] (the calibration read path uses it too).
    fn compose_embed_inputs(
        &self,
        raw: Vec<(i64, String, Vec<String>)>,
        only_notes: Option<&[i64]>,
    ) -> Vec<EmbedInput> {
        compose_embed_inputs(&*self.derived, &self.recognition_gate, raw, only_notes)
    }

    /// The index-add half of the maintained tail: the text-primary write plus
    /// the separate image-primary write when one exists. Atomic per note (Theme
    /// H): all of a note's vectors land in one orchestrator `add`.
    async fn write_index(
        &self,
        raw_inputs: &[(i64, String, Vec<String>)],
        written: &[i64],
        svc: &EmbedService,
    ) -> NativeResult<()> {
        let inputs = self.compose_embed_inputs(raw_inputs.to_vec(), Some(written));
        self.index_set
            .primary()
            .add(&inputs, &*svc.embedder, svc.images_pair())
            .await?;
        if let Some((iorch, isvc)) = self.image_only_route() {
            if let Some((ie, r)) = isvc.images_pair() {
                iorch
                    .add_with_mode(
                        &inputs,
                        &*isvc.embedder,
                        Some((ie, r)),
                        index_orchestrator::WriteMode::ImageOnly,
                    )
                    .await?;
            }
        }
        Ok(())
    }

    /// The derived-ingest half: group rows per note, ingest in ONE transaction.
    fn ingest_derived(
        &self,
        rows: &[(i64, String, String, String)],
        written: &[i64],
    ) -> NativeResult<()> {
        let mut refs: BTreeMap<i64, Vec<(String, String)>> = BTreeMap::new();
        for (nid, _source, name, value) in rows {
            refs.entry(*nid)
                .or_default()
                .push((name.clone(), value.clone()));
        }
        let batch: Vec<(i64, Vec<(String, String)>)> = written
            .iter()
            .map(|note_id| (*note_id, refs.remove(note_id).unwrap_or_default()))
            .collect();
        self.derived.ingest_many(&batch, FIELD_SOURCE)
    }

    /// Recompute every tag centroid from the engine's current text vectors —
    /// the SYNCHRONOUS refresh boot/rebuild paths use (a no-op without an
    /// embedder). The per-op tail uses the coalesced [`TagRefresher`] instead.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection read fails or the engine rejects the
    /// `tag.text` rebuild.
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
        let tag_engine = self.index_set.tag_engine();
        let built =
            tag_centroids::recompute(&*tag_engine, &rows, total, &self.tag_config, &self.tag_keys)?;
        self.index_set.primary_saver().request_save();
        Ok(built)
    }

    /// Best-effort synchronous tag refresh: never fails the op it rides on.
    async fn refresh_tags_best_effort(&self) {
        if let Err(e) = self.refresh_tag_centroids().await {
            tracing::warn!(error = ?e, "tag centroid refresh failed");
        }
    }

    /// Advance one space's watermark through its floor (a discrete lock, never
    /// held across an `await`), returning the value to stamp (or `None`).
    fn resolve_index(&self, col_mod: i64, ok: bool) -> Option<i64> {
        self.floors
            .lock()
            .expect("floors poisoned")
            .index
            .resolve(col_mod, ok)
    }

    fn resolve_derived(&self, col_mod: i64, ok: bool) -> Option<i64> {
        self.floors
            .lock()
            .expect("floors poisoned")
            .derived
            .resolve(col_mod, ok)
    }

    /// Clear the index poison floor — a whole-collection reconcile/rebuild healed
    /// every prior index-tail failure.
    fn clear_index_poison(&self) {
        self.floors.lock().expect("floors poisoned").index.clear();
    }

    /// Clear the derived poison floor.
    fn clear_derived_poison(&self) {
        self.floors.lock().expect("floors poisoned").derived.clear();
    }

    /// Index a set of present notes + remove an absent/deleted set, then advance
    /// each space's watermark to `advance_to` (with floor/failure handling). The
    /// shared core of [`Self::process_batch`] and [`Self::store_recognition`].
    /// Returns `(index_ok, derived_ok)` for the caller's tag-refresh decision.
    async fn apply_maintenance(
        &self,
        raw_inputs: &[(i64, String, Vec<String>)],
        rows: &[(i64, String, String, String)],
        to_remove: &[i64],
        advance_to: i64,
        floor_on_fail: i64,
    ) -> (bool, bool) {
        let written: Vec<i64> = raw_inputs.iter().map(|(id, _, _)| *id).collect();
        // The index watermark advances ONLY when an embedder is attached (else
        // it must stay so the next attach reconciles). The derived watermark is
        // lexical and always maintained.
        let mut index_ok = self.embed_service().is_some();
        let mut derived_ok = true;

        if !written.is_empty() {
            if let Some(svc) = self.embed_service() {
                if let Err(e) = self.write_index(raw_inputs, &written, &svc).await {
                    tracing::warn!(error = %e, notes = written.len(), "index add failed after commit; leaving the index watermark behind");
                    index_ok = false;
                }
            }
            if let Err(e) = self.ingest_derived(rows, &written) {
                tracing::warn!(error = %e, notes = written.len(), "derived ingest failed after commit; leaving the derived watermark behind");
                derived_ok = false;
            }
        }
        if !to_remove.is_empty() {
            if let Err(e) = self.index_set.remove_all(to_remove) {
                tracing::warn!(error = %e, notes = to_remove.len(), "index removal failed; leaving the index watermark behind");
                index_ok = false;
            }
            if let Err(e) = self.derived.remove(to_remove, None) {
                tracing::warn!(error = %e, notes = to_remove.len(), "derived removal failed; leaving the derived watermark behind");
                derived_ok = false;
            }
        }

        if let Some(v) = self.resolve_derived(
            if derived_ok {
                advance_to
            } else {
                floor_on_fail
            },
            derived_ok,
        ) {
            if let Err(e) = self.derived.set_col_mod(v) {
                tracing::warn!(error = %e, "advancing the derived watermark failed");
            }
        }
        if let Some(v) =
            self.resolve_index(if index_ok { advance_to } else { floor_on_fail }, index_ok)
        {
            self.index_set.set_col_mod_all(v);
            self.index_set.request_save_all();
        }
        (index_ok, derived_ok)
    }

    /// Process one drained batch — the merged maintained tail. The batch is a
    /// contiguous `col.mod` prefix (FIFO), so the watermark advances to its max
    /// `col.mod` on success, or leaves the floor at its min on failure.
    pub async fn process_batch(&self, items: Vec<IngestItem>) {
        if items.is_empty() {
            return;
        }

        // FIFO last-touching kind per id (a later delete supersedes an earlier
        // upsert and vice-versa), plus the batch's col.mod span + membership flag.
        let mut last: BTreeMap<i64, MaintKind> = BTreeMap::new();
        let mut membership_may_have_changed = false;
        let mut min_col_mod = i64::MAX;
        let mut max_col_mod = i64::MIN;
        for item in &items {
            membership_may_have_changed |= item.membership_may_have_changed;
            min_col_mod = min_col_mod.min(item.col_mod);
            max_col_mod = max_col_mod.max(item.col_mod);
            match item.kind {
                MaintKind::AdvanceOnly => {}
                kind => {
                    for &id in &item.ids {
                        last.insert(id, kind);
                    }
                }
            }
        }
        let maintain_ids: Vec<i64> = last
            .iter()
            .filter(|(_, k)| **k == MaintKind::Maintain)
            .map(|(id, _)| *id)
            .collect();
        let mut to_remove: BTreeSet<i64> = last
            .iter()
            .filter(|(_, k)| **k == MaintKind::Remove)
            .map(|(id, _)| *id)
            .collect();

        // Re-read the maintain ids' content at drain (the drain-merge truth):
        // present ⇒ embed + derived ingest, absent ⇒ remove its vectors/rows.
        let (raw_inputs, rows, tagged) = if maintain_ids.is_empty() {
            (Vec::new(), Vec::new(), false)
        } else {
            let ids = maintain_ids.clone();
            let read = self
                .collection
                .run(move |core| -> NativeResult<_> {
                    let inputs = core.note_embed_inputs(&ids)?;
                    let rows = core.derived_field_rows(&ids)?;
                    let tagged = core.any_tagged(&ids)?;
                    Ok((inputs, rows, tagged))
                })
                .await;
            match read.and_then(|r| r) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        notes = maintain_ids.len(),
                        "ingest re-read failed after the collection write committed; \
                         leaving the watermarks behind for next-boot drift to heal"
                    );
                    self.resolve_index(min_col_mod, false);
                    self.resolve_derived(min_col_mod, false);
                    return;
                }
            }
        };
        let present: BTreeSet<i64> = raw_inputs.iter().map(|(id, _, _)| *id).collect();
        for id in &maintain_ids {
            if !present.contains(id) {
                to_remove.insert(*id);
            }
        }
        let to_remove: Vec<i64> = to_remove.into_iter().collect();

        let (index_ok, _derived_ok) = self
            .apply_maintenance(&raw_inputs, &rows, &to_remove, max_col_mod, min_col_mod)
            .await;

        // Tag centroids derive from the text vectors just written: refresh off
        // the tail (coalesced) when the index landed and membership is relevant,
        // when a removed note was a member, or on a metadata membership change.
        let written: Vec<i64> = raw_inputs.iter().map(|(id, _, _)| *id).collect();
        let touched_member = (index_ok && (tagged || self.tag_keys.any_member_of(&written)))
            || self.tag_keys.any_member_of(&to_remove)
            || membership_may_have_changed;
        if touched_member {
            self.tag_refresh.request();
        }
    }

    /// Persist a recognition sweep's text + segments + below-gate markers and
    /// re-embed the affected notes — all on the actor, so the write serializes
    /// with rebuild (#828 closed). Recognition is orthogonal to the `col.mod`
    /// watermark (it stores media-derived text, which never bumps `col.mod`), so
    /// the re-embed re-certifies the CURRENT `col.mod` for the affected notes.
    ///
    /// # Errors
    ///
    /// Returns an error if a derived write fails. The re-embed tail is
    /// best-effort (a failed embed leaves the index watermark behind to heal).
    pub async fn store_recognition(&self, write: RecognitionWrite) -> NativeResult<()> {
        for (note_id, refs_text) in &write.touched {
            self.derived.ingest(*note_id, &write.source, refs_text)?;
        }
        for (note_id, name, json) in &write.segments {
            self.derived
                .put_segments(*note_id, &write.source, name, json)?;
        }
        self.derived.mark_gated(&write.source, &write.gated)?;

        if write.affected.is_empty() {
            return Ok(());
        }
        // Re-read current col.mod + the affected notes' content in one job, then
        // index + advance (recognition mints vectors at the current col.mod).
        let ids = write.affected.clone();
        let read = self
            .collection
            .run(move |core| -> NativeResult<_> {
                let col_mod = core.col_mod()?;
                let inputs = core.note_embed_inputs(&ids)?;
                let rows = core.derived_field_rows(&ids)?;
                Ok((col_mod, inputs, rows))
            })
            .await;
        let (col_mod, raw_inputs, rows) = match read.and_then(|r| r) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "recognition re-embed read failed; leaving the watermark behind");
                return Ok(());
            }
        };
        self.apply_maintenance(&raw_inputs, &rows, &[], col_mod, col_mod)
            .await;
        Ok(())
    }

    /// Whole-collection drift reconcile of the index (the boot/reload/import
    /// path). Re-embeds only changed/new notes, drops deleted, stamps its own
    /// snapshot `col.mod`, and clears the index poison floor.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection read, embedding, or a reconcile fails.
    #[must_use = "whether reindexing ran is the caller's signal to refresh status"]
    pub async fn reindex_if_needed(&self) -> NativeResult<bool> {
        let Some(svc) = self.embed_service() else {
            return Ok(false);
        };
        let orch = self.index_set.primary();
        let model_id = svc.embedder.fingerprint();
        // Read col.mod AND the live id list in ONE collection job, so the
        // stamped watermark matches the set this drift pass reconciles against.
        // The note CONTENT is read in chunks by the streaming producer (O(chunk)
        // memory), pipelined against the embed.
        let (col_mod, ids) = self
            .collection
            .run(|core| -> NativeResult<_> {
                let col_mod = core.col_mod()?;
                let ids = core.find_notes("")?;
                Ok((col_mod, ids))
            })
            .await??;
        if !orch.check_drift(col_mod, model_id.as_deref(), svc.images.is_some()) {
            return Ok(false);
        }
        if ids.is_empty() {
            if let Some(dim) = svc.embedder.dim() {
                orch.materialize_empty(dim, col_mod, model_id.as_deref());
            }
            self.clear_index_poison();
            self.refresh_tags_best_effort().await;
            return Ok(true);
        }
        // The PRIMARY text space: streamed reconcile (or rebuild on a model swap
        // / first build), with read‖embed pipelined.
        self.stream_drift(
            &orch,
            index_orchestrator::WriteMode::TextAndImage,
            &*svc.embedder,
            svc.images_pair(),
            &ids,
            col_mod,
            model_id,
        )
        .await?;
        self.clear_index_poison();
        // The SEPARATE image-primary space, if any: its own drift, its own
        // streamed pass over the same ids in ImageOnly mode.
        if let Some((iorch, isvc)) = self.image_only_route() {
            if let Some((ie, r)) = isvc.images_pair() {
                let image_model = isvc.embedder.fingerprint();
                if iorch.check_drift(col_mod, image_model.as_deref(), true) {
                    self.stream_drift(
                        &iorch,
                        index_orchestrator::WriteMode::ImageOnly,
                        &*isvc.embedder,
                        Some((ie, r)),
                        &ids,
                        col_mod,
                        image_model,
                    )
                    .await?;
                }
            }
        }
        self.refresh_tags_best_effort().await;
        Ok(true)
    }

    /// Drive a streamed drift reconcile (or a full rebuild on a model swap /
    /// first build) of one index space over `ids`, the collection read
    /// pipelined against the embed via [`Self::stream_inputs`].
    #[allow(clippy::too_many_arguments)]
    async fn stream_drift(
        &self,
        orch: &index_orchestrator::IndexOrchestrator,
        mode: index_orchestrator::WriteMode,
        embedder: &dyn Embedder,
        images: Option<(&dyn ImageEmbedder, &dyn ImageResolver)>,
        ids: &[i64],
        col_mod: i64,
        model_id: Option<String>,
    ) -> NativeResult<()> {
        let total = ids.len() as u64;
        let do_rebuild = {
            let stored = orch.model_id();
            (stored.is_some() && stored != model_id) || !orch.has_hashes()
        };
        let rx = self.stream_inputs(ids.to_vec());
        if do_rebuild {
            orch.rebuild_streamed(rx, total, col_mod, model_id, embedder, images, mode)
                .await
        } else {
            orch.reconcile_streamed(rx, total, col_mod, model_id, embedder, images, mode)
                .await
        }
    }

    /// Spawn a producer that reads `note_embed_inputs` for each `STREAM_CHUNK`
    /// of `ids` on the collection actor (drive_sync) + composes recognized text,
    /// sending composed chunks on a bounded channel (depth 2 = one chunk
    /// prefetched while the consumer embeds on drive_compute). Only O(chunk)
    /// inputs live at once, and the next read overlaps this chunk's embed.
    fn stream_inputs(
        &self,
        ids: Vec<i64>,
    ) -> tokio::sync::mpsc::Receiver<NativeResult<Vec<EmbedInput>>> {
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        let collection = Arc::clone(&self.collection);
        let derived = Arc::clone(&self.derived);
        let gate = self.recognition_gate.clone();
        runtime::handle().spawn(async move {
            for chunk_ids in ids.chunks(index_orchestrator::STREAM_CHUNK) {
                let read_ids = chunk_ids.to_vec();
                let read = collection
                    .run(move |core| core.note_embed_inputs(&read_ids))
                    .await;
                let composed: NativeResult<Vec<EmbedInput>> = match read.and_then(|r| r) {
                    Ok(raw) => Ok(compose_embed_inputs(&*derived, &gate, raw, Some(chunk_ids))),
                    Err(e) => Err(e),
                };
                let is_err = composed.is_err();
                // A send error (consumer dropped the rx) or a read error ends
                // the producer; the read error rides the channel to the consumer.
                if tx.send(composed).await.is_err() || is_err {
                    break;
                }
            }
        });
        rx
    }

    /// Explicit full index rebuild (`/index/rebuild`): drop and re-embed
    /// everything, streamed (read‖embed pipelined, O(chunk) memory).
    ///
    /// # Errors
    ///
    /// Returns `Unavailable` when no embedder is attached, or an error if the
    /// collection read, embedding, or the engine rebuild fails.
    pub async fn rebuild_index(&self) -> NativeResult<usize> {
        let Some(svc) = self.embed_service() else {
            return Err(NativeError::unavailable(
                "no embedding service attached — start embedding first",
            ));
        };
        let model_id = svc.embedder.fingerprint();
        let (col_mod, ids) = self
            .collection
            .run(|core| -> NativeResult<_> {
                let col_mod = core.col_mod()?;
                let ids = core.find_notes("")?;
                Ok((col_mod, ids))
            })
            .await??;
        let total = ids.len();
        let rx = self.stream_inputs(ids.clone());
        self.index_set
            .primary()
            .rebuild_streamed(
                rx,
                total as u64,
                col_mod,
                model_id,
                &*svc.embedder,
                svc.images_pair(),
                index_orchestrator::WriteMode::TextAndImage,
            )
            .await?;
        if let Some((iorch, isvc)) = self.image_only_route() {
            if let Some((ie, r)) = isvc.images_pair() {
                let rx2 = self.stream_inputs(ids.clone());
                iorch
                    .rebuild_streamed(
                        rx2,
                        total as u64,
                        col_mod,
                        isvc.embedder.fingerprint(),
                        &*isvc.embedder,
                        Some((ie, r)),
                        index_orchestrator::WriteMode::ImageOnly,
                    )
                    .await?;
            }
        }
        self.clear_index_poison();
        self.refresh_tags_best_effort().await;
        Ok(total)
    }

    /// Full derived-text (FTS5) rebuild, kernel-side. Serialized with the hot
    /// path on the actor — so a concurrent per-op derived ingest can no longer
    /// land inside the snapshot→build→prune window (#828 closed structurally).
    /// Returns `(row_count, col_mod)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection read or the derived-store build fails.
    pub async fn rebuild_derived(&self) -> NativeResult<(usize, i64)> {
        // Read the live id list + col.mod (cheap); the field-row CONTENT streams
        // in chunks (O(chunk) memory), pipelined against the FTS5 inserts.
        let (ids, dmod) = self
            .collection
            .run(|core| -> NativeResult<_> {
                let ids = core.find_notes("")?;
                let dmod = core.col_mod()?;
                Ok((ids, dmod))
            })
            .await??;
        // Producer: read derived_field_rows per chunk on the collection actor
        // (drive_sync), sending on a bounded channel (depth 2 = one chunk
        // prefetched while the build inserts the prior).
        let (tx, mut rx) =
            tokio::sync::mpsc::channel::<NativeResult<Vec<(i64, String, String, String)>>>(2);
        let collection = Arc::clone(&self.collection);
        let producer_ids = ids.clone();
        runtime::handle().spawn(async move {
            for chunk_ids in producer_ids.chunks(index_orchestrator::STREAM_CHUNK) {
                let read_ids = chunk_ids.to_vec();
                let rows = collection
                    .run(move |core| core.derived_field_rows(&read_ids))
                    .await
                    .and_then(|r| r);
                let is_err = rows.is_err();
                if tx.send(rows).await.is_err() || is_err {
                    break;
                }
            }
        });
        // Consumer: the FTS5 rebuild on the compute pool, pulling chunks via
        // `blocking_recv` (a plain compute thread, not a runtime context). The
        // read overlaps the insert; the whole rebuild is ONE transaction.
        let derived = Arc::clone(&self.derived);
        let live = ids;
        let n = runtime::dispatch_compute(move || {
            let mut next = || rx.blocking_recv();
            derived.build_streamed(&mut next, &live, dmod)
        })
        .await?;
        // A clean whole-collection rebuild re-ingests every note and stamps its
        // own snapshot col_mod → prior per-op derived failures are healed.
        self.clear_derived_poison();
        Ok((n, dmod))
    }

    /// Flush the index savers durably (shutdown's "watermark durable" step). The
    /// derived watermark is already durable (`set_col_mod` is a synchronous
    /// SQLite write).
    fn flush_durable(&self) {
        for saver in self.index_set.all_savers() {
            saver.flush();
        }
    }
}

/// The persistent FIFO consumer. Drains items in batches, runs jobs/flushes in
/// strict order, and exits when the channel closes (every sender dropped), then
/// flushes the index savers durably.
///
/// This task is the kernel's SOLE index/derived writer, so an uncaught panic in
/// processing one item would kill it for the kernel's life — hot-path enqueues
/// then silently no-op (writes commit to anki but are never indexed), with no
/// recovery. A poisoned lock (e.g. an embed-slot writer panicking under the
/// `RwLock`) is the realistic trigger: every later `.read().expect()` would
/// panic. So each item/job is processed under a panic boundary
/// ([`process_caught`]): a caught panic loses only that item — logged + counted
/// for `/status` — and the loop survives. Next-boot drift then re-indexes the
/// skipped notes.
async fn drain_loop(ingestor: Arc<Ingestor>, mut rx: mpsc::UnboundedReceiver<IngestMsg>) {
    let mut pending: Option<IngestMsg> = None;
    loop {
        let msg = match pending.take() {
            Some(m) => m,
            None => match rx.recv().await {
                Some(m) => m,
                None => break,
            },
        };
        match msg {
            IngestMsg::Item(item) => {
                let mut batch = vec![item];
                // Coalesce consecutive items into one batch, stashing the first
                // non-item so it is handled (in order) on the next iteration.
                loop {
                    match rx.try_recv() {
                        Ok(IngestMsg::Item(it)) => batch.push(it),
                        Ok(other) => {
                            pending = Some(other);
                            break;
                        }
                        Err(_) => break,
                    }
                }
                let ing = Arc::clone(&ingestor);
                let work = async move { ing.process_batch(batch).await }.boxed();
                process_caught(&ingestor.drain_panics, "batch", work).await;
            }
            IngestMsg::Job(run) => {
                let ing = Arc::clone(&ingestor);
                process_caught(&ingestor.drain_panics, "job", run(ing)).await;
            }
            IngestMsg::Flush(done) => {
                let _ = done.send(());
            }
        }
    }
    ingestor.flush_durable();
}

/// Drive one drain-loop unit of work under a panic boundary, so a fault in
/// processing one item (a poisoned lock, an unexpected bug) loses only that item
/// instead of killing the sole writer. A caught panic is logged once at WARNING
/// and counted on `panics` (the `/status` degraded signal).
async fn process_caught(
    panics: &std::sync::atomic::AtomicU64,
    kind: &str,
    work: BoxFuture<'static, ()>,
) {
    if std::panic::AssertUnwindSafe(work)
        .catch_unwind()
        .await
        .is_err()
    {
        panics.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        tracing::warn!(
            %kind,
            "ingest drain caught a panic; that unit's notes are un-indexed until the next reconcile heals them"
        );
    }
}

/// Compose orchestrator inputs from collection rows + the derived store's
/// recognized texts across every vector-minting source: the index derives from
/// collection text + OCR + ASR + VLM-describe, so reconcile == rebuild keeps
/// holding after recognition, and a note's recognized text mints vectors on any
/// (re-)embed path. Vector-worthiness re-judges from the stored text (confidence
/// already gated at ingest). Shared by the ingest actor's write paths and the
/// kernel's secondary-floor calibration read path.
pub(crate) fn compose_embed_inputs(
    derived: &dyn DerivedStore,
    gate: &RecognitionGate,
    raw: Vec<(i64, String, Vec<String>)>,
    only_notes: Option<&[i64]>,
) -> Vec<EmbedInput> {
    let mut recognized_map: std::collections::HashMap<i64, Vec<String>> =
        std::collections::HashMap::new();
    // One query per source (a small fixed set — ocr/vlm/asr), each bounded by
    // rows that EXIST for that source (most notes have none), so this is
    // proportional to recognized-content volume, not 3× the collection.
    for source in crate::Kernel::vector_minting_sources() {
        let texts = match only_notes {
            Some(ids) => derived.texts_for_source_for_notes(source, ids),
            None => derived.texts_for_source(source),
        };
        match texts {
            Ok(rows) => {
                for (nid, _r, text) in rows {
                    if gate.vector_worthy(&text) {
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
        .map(|(note_id, text, image_names)| EmbedInput {
            note_id,
            text,
            image_names,
            ocr_texts: recognized_map.remove(&note_id).unwrap_or_default(),
        })
        .collect()
}

/// Spawn the drain task and return the kernel-held handle.
pub fn spawn(ingestor: Arc<Ingestor>) -> IngestHandle {
    let (tx, rx) = mpsc::unbounded_channel::<IngestMsg>();
    let drain_panics = Arc::clone(&ingestor.drain_panics);
    let task = runtime::handle().spawn(drain_loop(ingestor, rx));
    IngestHandle {
        tx: Mutex::new(Some(tx)),
        task: Mutex::new(Some(task)),
        drain_panics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A panicking unit of drain work is CONTAINED: the counter increments and
    /// `process_caught` returns normally (the loop survives), instead of the
    /// panic unwinding out and killing the sole writer. A clean unit leaves the
    /// counter untouched. This is the M3 panic-boundary guarantee in isolation.
    #[test]
    fn process_caught_contains_a_panicking_unit() {
        let panics = Arc::new(AtomicU64::new(0));
        let p = Arc::clone(&panics);
        // Quiet the default panic hook for the duration: catch_unwind still
        // catches, but we don't want the backtrace noise in test output.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        runtime::testing::run_with_sync(async move {
            process_caught(&p, "job", async { panic!("boom") }.boxed()).await;
            assert_eq!(p.load(Ordering::Relaxed), 1, "the panic was counted");
            process_caught(&p, "job", async {}.boxed()).await;
            assert_eq!(
                p.load(Ordering::Relaxed),
                1,
                "a clean unit leaves the counter untouched"
            );
        });
        std::panic::set_hook(prev);
    }
}
