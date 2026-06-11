//! The index orchestrator (#332, S3c-2): the `VectorIndex` *policy* re-homed.
//!
//! The engine (`shrike-index`) has owned storage since #273; this module owns
//! what the Python orchestrator owned — the meta/hashes sidecars, the drift
//! rules, the per-note change fingerprints, the state machine, and the
//! debounced saver (over the injected [`TimerHost`]). File formats are
//! byte-compatible with the Python orchestrator's (`index.meta.json`,
//! `index.hashes.json`), and the note hash is bit-identical to Python's
//! `hashlib.blake2b(digest_size=8)` — so an index built before the swap loads
//! without a rebuild (the "text-only never rebuilds on upgrade" invariant,
//! once more).
//!
//! This first slice is the embed-free core: load/materialize/drift/hashes/
//! save + the saver. The embed-coupled ops (add/reconcile/rebuild,
//! calibration) follow with the `PyEmbedder` inversion (S3c-3).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use serde::{Deserialize, Serialize};

use shrike_ffi::{NativeError, NativeResult};
use shrike_index::MultiModalIndex;

use crate::{TimerCancel, TimerHost};

/// The on-disk schema version of a freshly built index (mirrors
/// `index.INDEX_SCHEMA_VERSION`: v2 = per-modality sub-indexes, #201a).
pub const INDEX_SCHEMA_VERSION: i64 = 2;

/// Saver defaults (mirrors `index.DEFAULT_SAVE_DELAY` / `_THRESHOLD`).
pub const DEFAULT_SAVE_DELAY: f64 = 60.0;
pub const DEFAULT_SAVE_THRESHOLD: u64 = 100;

/// Stable fingerprint of a note's embedding text — bit-identical to Python's
/// `hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()`.
pub fn hash_text(text: &str) -> String {
    let mut hasher = Blake2bVar::new(8).expect("8-byte blake2b");
    hasher.update(text.as_bytes());
    let mut out = [0u8; 8];
    hasher.finalize_variable(&mut out).expect("8-byte output");
    out.iter().map(|b| format!("{b:02x}")).collect()
}

/// Per-note change fingerprint (mirrors `VectorIndex._note_hash`): folds image
/// filenames in only when the backend embeds images *and the image resolves*,
/// so a text-only backend's hash is byte-identical to the text-only scheme.
pub fn note_hash(
    text: &str,
    image_names: &[String],
    embeds_images: bool,
    image_exists: &dyn Fn(&str) -> bool,
    ocr_texts: &[String],
) -> String {
    // Recognized texts fold in behind a distinct separator (#228), so an
    // OCR change re-embeds exactly that note; with none, every branch below
    // is byte-identical to the pre-OCR scheme (no upgrade rebuild).
    let text_part: std::borrow::Cow<'_, str> = if ocr_texts.is_empty() {
        std::borrow::Cow::Borrowed(text)
    } else {
        let mut sorted: Vec<&str> = ocr_texts.iter().map(String::as_str).collect();
        sorted.sort_unstable();
        std::borrow::Cow::Owned(format!("{text}\u{1e}{}", sorted.join("\u{1e}")))
    };
    if embeds_images {
        let mut present: Vec<&str> = image_names
            .iter()
            .map(String::as_str)
            .filter(|n| image_exists(n))
            .collect();
        if !present.is_empty() {
            present.sort_unstable();
            let joined = format!("{text_part}\u{1f}{}", present.join("\u{1f}"));
            return hash_text(&joined);
        }
    }
    hash_text(&text_part)
}

/// `index.meta.json` — the exact shape the Python orchestrator wrote.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMeta {
    pub ndim: i64,
    #[serde(default)]
    pub col_mod: Option<i64>,
    #[serde(default)]
    pub model_id: Option<String>,
    /// Marker absent → pre-#201a (v1) single-index layout.
    #[serde(default = "default_schema_v1")]
    pub schema: i64,
    /// Presence (even empty) records that calibration ran (one-shot).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation: Option<BTreeMap<String, BTreeMap<String, f64>>>,
}

fn default_schema_v1() -> i64 {
    1
}

/// The index build/availability state (mirrors `IndexState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrchestratorState {
    Ready,
    Building,
    Unavailable,
    Error,
}

struct Shared {
    state: OrchestratorState,
    col_mod: Option<i64>,
    model_id: Option<String>,
    schema: i64,
    activation: Option<BTreeMap<String, BTreeMap<String, f64>>>,
    /// note_id → embedding-text fingerprint; `None` = no per-note state (an
    /// old or never-built index) → the next reconcile full-rebuilds.
    note_hashes: Option<BTreeMap<i64, String>>,
    /// Populated by the embed-coupled ops; read by status().
    build_progress: (u64, u64),
    error: Option<String>,
}

/// The embed-free orchestration core over one engine + its sidecars. The
/// engine is `Arc`-shared: the harness's search surface holds the same engine
/// the orchestrator maintains (one set of vectors, two roles).
pub struct IndexOrchestrator {
    pub(crate) dir: PathBuf,
    engine: Arc<MultiModalIndex>,
    shared: Mutex<Shared>,
}

impl IndexOrchestrator {
    /// Create over a directory, loading any existing on-disk index + sidecars
    /// (the Python `_load` semantics: corrupt meta → unloaded; corrupt/missing
    /// hashes → `None` (rebuild-on-reconcile); engine restore failure clears
    /// both so drift forces a full rebuild).
    pub fn open(dir: impl Into<PathBuf>, engine: Arc<MultiModalIndex>) -> Self {
        let dir = dir.into();
        let mut shared = Shared {
            state: OrchestratorState::Unavailable,
            col_mod: None,
            model_id: None,
            schema: INDEX_SCHEMA_VERSION,
            activation: None,
            note_hashes: None,
            build_progress: (0, 0),
            error: None,
        };

        let meta_path = dir.join("index.meta.json");
        let index_path = dir.join("index.usearch");
        if index_path.exists() && meta_path.exists() {
            match std::fs::read_to_string(&meta_path)
                .map_err(|e| e.to_string())
                .and_then(|s| serde_json::from_str::<IndexMeta>(&s).map_err(|e| e.to_string()))
            {
                Ok(meta) => {
                    shared.col_mod = meta.col_mod;
                    shared.model_id = meta.model_id;
                    shared.schema = meta.schema;
                    shared.activation = meta.activation;

                    let hashes_path = dir.join("index.hashes.json");
                    if hashes_path.exists() {
                        shared.note_hashes = std::fs::read_to_string(&hashes_path)
                            .ok()
                            .and_then(|s| serde_json::from_str::<BTreeMap<String, String>>(&s).ok())
                            .and_then(|m| {
                                m.into_iter()
                                    .map(|(k, v)| k.parse::<i64>().map(|k| (k, v)).ok())
                                    .collect::<Option<BTreeMap<i64, String>>>()
                            });
                        if shared.note_hashes.is_none() {
                            tracing::warn!(path = %hashes_path.display(), "corrupt index hashes");
                        }
                    }

                    let candidates: Option<Vec<i64>> = shared
                        .note_hashes
                        .as_ref()
                        .map(|m| m.keys().copied().collect());
                    let restored =
                        engine.restore(dir.to_str().unwrap_or_default(), candidates.as_deref());
                    if !restored {
                        shared.note_hashes = None; // rebuild repopulates both together
                    }
                }
                Err(e) => {
                    tracing::warn!(path = %meta_path.display(), error = %e, "corrupt index metadata");
                }
            }
        }

        Self {
            dir,
            engine,
            shared: Mutex::new(shared),
        }
    }

    pub fn engine(&self) -> &MultiModalIndex {
        &self.engine
    }

    /// The shared engine, for a harness search handle over the same vectors.
    pub fn engine_arc(&self) -> Arc<MultiModalIndex> {
        Arc::clone(&self.engine)
    }

    /// Embedder detached: searchable vectors remain on disk/memory but the
    /// semantic surface is down (mirrors the Python `set_backend(None)`).
    pub fn mark_unavailable(&self) {
        self.shared.lock().expect("orchestrator poisoned").state = OrchestratorState::Unavailable;
    }

    /// Embedder (re)attached: flip back to ready ONLY from unavailable — a
    /// building or errored state is the (re)build path's to resolve.
    pub fn mark_ready_if_loaded(&self) {
        let mut shared = self.shared.lock().expect("orchestrator poisoned");
        if shared.state == OrchestratorState::Unavailable {
            shared.state = OrchestratorState::Ready;
        }
    }

    /// The drift policy (mirrors `VectorIndex.check_drift`): nothing loaded /
    /// no stamp / model change / a v1 layout under an image-capable backend →
    /// rebuild; a bare `col_mod` move → reconcile; otherwise current.
    pub fn check_drift(
        &self,
        current_col_mod: i64,
        current_model_id: Option<&str>,
        embeds_images: bool,
    ) -> bool {
        let shared = self.shared.lock().expect("orchestrator poisoned");
        if self.engine.size() == 0 && self.engine.ndim().is_none() {
            return true; // no index loaded
        }
        let Some(col_mod) = shared.col_mod else {
            return true; // no stamp in the metadata
        };
        if let Some(current) = current_model_id {
            if shared.model_id.as_deref() != Some(current) {
                return true; // different model → different vector space
            }
        }
        if shared.schema < INDEX_SCHEMA_VERSION && embeds_images {
            return true; // pre-#201a layout can't split per modality
        }
        col_mod != current_col_mod
    }

    /// Which notes changed/added/vanished vs the per-note fingerprints — the
    /// reconcile diff. `None` = no prior state, do a full rebuild instead.
    #[allow(clippy::type_complexity)]
    pub fn reconcile_diff(
        &self,
        new_hashes: &BTreeMap<i64, String>,
    ) -> Option<(Vec<i64>, Vec<i64>)> {
        let shared = self.shared.lock().expect("orchestrator poisoned");
        let old = shared.note_hashes.as_ref()?;
        let to_embed: Vec<i64> = new_hashes
            .iter()
            .filter(|(nid, h)| old.get(nid) != Some(h))
            .map(|(nid, _)| *nid)
            .collect();
        let to_remove: Vec<i64> = old
            .keys()
            .filter(|nid| !new_hashes.contains_key(nid))
            .copied()
            .collect();
        Some((to_embed, to_remove))
    }

    pub fn state(&self) -> OrchestratorState {
        self.shared.lock().expect("orchestrator poisoned").state
    }

    pub fn col_mod(&self) -> Option<i64> {
        self.shared.lock().expect("orchestrator poisoned").col_mod
    }

    pub fn set_col_mod(&self, value: i64) {
        self.shared.lock().expect("orchestrator poisoned").col_mod = Some(value);
    }

    pub fn model_id(&self) -> Option<String> {
        self.shared
            .lock()
            .expect("orchestrator poisoned")
            .model_id
            .clone()
    }

    pub fn build_progress(&self) -> (u64, u64) {
        self.shared
            .lock()
            .expect("orchestrator poisoned")
            .build_progress
    }

    /// True when per-note fingerprints exist (an incremental reconcile is
    /// possible; absent → the next drift handles via full rebuild).
    pub fn has_note_hashes(&self) -> bool {
        self.shared
            .lock()
            .expect("orchestrator poisoned")
            .note_hashes
            .is_some()
    }

    /// Persist the engine files + both sidecars (the Python `save` semantics:
    /// meta carries ndim/col_mod/model_id/schema and the activation key only
    /// once calibration ran; hashes ride alongside when present).
    pub fn save(&self) -> NativeResult<()> {
        let shared = self.shared.lock().expect("orchestrator poisoned");
        let Some(ndim) = self.engine.ndim() else {
            return Ok(()); // nothing built — nothing to persist
        };
        std::fs::create_dir_all(&self.dir)
            .map_err(|e| NativeError::internal(format!("index dir: {e}")))?;
        self.engine
            .save(self.dir.to_str().unwrap_or_default())
            .map_err(|e| NativeError::internal(format!("engine save: {e}")))?;
        let meta = IndexMeta {
            ndim: ndim as i64,
            col_mod: shared.col_mod,
            model_id: shared.model_id.clone(),
            schema: shared.schema,
            activation: shared.activation.clone(),
        };
        let meta_json = serde_json::to_string(&meta)
            .map_err(|e| NativeError::internal(format!("meta encode: {e}")))?;
        std::fs::write(self.dir.join("index.meta.json"), meta_json)
            .map_err(|e| NativeError::internal(format!("meta write: {e}")))?;
        if let Some(hashes) = &shared.note_hashes {
            let as_strings: BTreeMap<String, &String> =
                hashes.iter().map(|(k, v)| (k.to_string(), v)).collect();
            let hashes_json = serde_json::to_string(&as_strings)
                .map_err(|e| NativeError::internal(format!("hashes encode: {e}")))?;
            std::fs::write(self.dir.join("index.hashes.json"), hashes_json)
                .map_err(|e| NativeError::internal(format!("hashes write: {e}")))?;
        }
        Ok(())
    }
}

/// The debounced saver over the injected [`TimerHost`] (mirrors `IndexSaver`):
/// a flush `delay` seconds after the last change, or immediately at
/// `threshold` unsaved changes — whichever first. No threads, no loop
/// assumption: the timer host is whatever the harness runs on.
pub struct DebouncedSaver {
    orchestrator: Arc<IndexOrchestrator>,
    timers: Arc<dyn TimerHost>,
    delay: f64,
    threshold: u64,
    pending: Mutex<PendingState>,
}

#[derive(Default)]
struct PendingState {
    changes: u64,
    armed: Option<Box<dyn TimerCancel>>,
}

impl DebouncedSaver {
    pub fn new(
        orchestrator: Arc<IndexOrchestrator>,
        timers: Arc<dyn TimerHost>,
        delay: f64,
        threshold: u64,
    ) -> Arc<Self> {
        Arc::new(Self {
            orchestrator,
            timers,
            delay,
            threshold,
            pending: Mutex::new(PendingState::default()),
        })
    }

    /// Record a change: re-arm the idle timer, or flush now at the burst cap.
    pub fn request_save(self: &Arc<Self>) {
        let flush_now = {
            let mut pending = self.pending.lock().expect("saver poisoned");
            pending.changes += 1;
            if let Some(armed) = pending.armed.take() {
                armed.cancel();
            }
            pending.changes >= self.threshold
        };
        if flush_now {
            self.flush();
            return;
        }
        let this = Arc::clone(self);
        let cancel = self
            .timers
            .schedule(self.delay, Box::new(move || this.flush()));
        self.pending.lock().expect("saver poisoned").armed = Some(cancel);
    }

    /// Unsaved changes since the last flush.
    pub fn pending_changes(&self) -> u64 {
        self.pending.lock().expect("saver poisoned").changes
    }

    pub fn flush(&self) {
        {
            let mut pending = self.pending.lock().expect("saver poisoned");
            if let Some(armed) = pending.armed.take() {
                armed.cancel();
            }
            pending.changes = 0;
        }
        if let Err(e) = self.orchestrator.save() {
            tracing::warn!(error = ?e, "debounced index save failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_text_matches_python_blake2b() {
        // Pinned against hashlib.blake2b(b"...", digest_size=8).hexdigest().
        assert_eq!(hash_text(""), "e4a6a0577479b2b4");
        assert_eq!(hash_text("the cell"), hash_text("the cell"));
        assert_ne!(hash_text("a"), hash_text("b"));
        assert_eq!(hash_text("").len(), 16);
    }

    #[test]
    fn note_hash_folds_only_resolvable_images() {
        let always = |_: &str| true;
        let never = |_: &str| false;
        let names = vec!["b.png".to_owned(), "a.png".to_owned()];
        // Text-only backend → byte-identical to the bare text hash.
        assert_eq!(note_hash("t", &names, false, &always, &[]), hash_text("t"));
        // Image backend + nothing resolves → still the bare text hash.
        assert_eq!(note_hash("t", &names, true, &never, &[]), hash_text("t"));
        // Resolvable images fold in, sorted (matches the Python scheme).
        let folded = note_hash("t", &names, true, &always, &[]);
        assert_eq!(folded, hash_text("t\u{1f}a.png\u{1f}b.png"));
        assert_ne!(folded, hash_text("t"));
    }

    #[test]
    fn note_hash_folds_recognized_texts() {
        // No OCR → byte-identical to the pre-#228 scheme (no upgrade rebuild).
        let never = |_: &str| false;
        assert_eq!(note_hash("t", &[], false, &never, &[]), hash_text("t"));
        // OCR texts fold in sorted behind the record separator, so an OCR
        // change re-embeds exactly this note...
        let ocr = vec!["zeta".to_owned(), "alpha".to_owned()];
        let folded = note_hash("t", &[], false, &never, &ocr);
        assert_eq!(folded, hash_text("t\u{1e}alpha\u{1e}zeta"));
        // ...and composes with the image fold.
        let names = vec!["a.png".to_owned()];
        let always = |_: &str| true;
        let both = note_hash("t", &names, true, &always, &ocr);
        assert_eq!(both, hash_text("t\u{1e}alpha\u{1e}zeta\u{1f}a.png"));
    }

    #[test]
    fn meta_round_trips_the_python_shape() {
        let python_meta = r#"{"ndim": 384, "col_mod": 7, "model_id": "onnx-rs:m", "schema": 2,
                              "activation": {"image": {"n": 40.0, "mean": 0.2, "std": 0.05}}}"#;
        let meta: IndexMeta = serde_json::from_str(python_meta).unwrap();
        assert_eq!(meta.ndim, 384);
        assert_eq!(meta.schema, 2);
        assert!(meta.activation.is_some());
        // A v1 meta (no schema marker) defaults to 1, like the Python load.
        let v1: IndexMeta = serde_json::from_str(r#"{"ndim": 8}"#).unwrap();
        assert_eq!(v1.schema, 1);
        assert!(v1.col_mod.is_none());
    }

    struct ManualTimers {
        jobs: Mutex<Vec<Box<dyn FnOnce() + Send>>>,
    }

    struct NoCancel;
    impl TimerCancel for NoCancel {
        fn cancel(&self) {}
    }

    impl TimerHost for ManualTimers {
        fn schedule(
            &self,
            _delay: f64,
            job: Box<dyn FnOnce() + Send + 'static>,
        ) -> Box<dyn TimerCancel> {
            self.jobs.lock().unwrap().push(job);
            Box::new(NoCancel)
        }
    }

    fn temp_orchestrator() -> Arc<IndexOrchestrator> {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "shrike-orch-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let engine = Arc::new(MultiModalIndex::new(vec!["text".to_owned()]).unwrap());
        Arc::new(IndexOrchestrator::open(dir, engine))
    }

    #[test]
    fn drift_policy_matches_the_python_rules() {
        let orch = temp_orchestrator();
        // Nothing loaded → drift.
        assert!(orch.check_drift(1, Some("m"), false));
        // Materialize state by hand (the embed-coupled ops land next slice).
        orch.engine().ensure("text", 4).unwrap();
        orch.set_col_mod(5);
        {
            let mut shared = orch.shared.lock().unwrap();
            shared.model_id = Some("m".to_owned());
            shared.schema = INDEX_SCHEMA_VERSION;
        }
        assert!(!orch.check_drift(5, Some("m"), false)); // current
        assert!(orch.check_drift(6, Some("m"), false)); // col_mod moved
        assert!(orch.check_drift(5, Some("other"), false)); // model changed
        {
            orch.shared.lock().unwrap().schema = 1;
        }
        assert!(!orch.check_drift(5, Some("m"), false)); // v1 + text-only: fine
        assert!(orch.check_drift(5, Some("m"), true)); // v1 + images: rebuild
    }

    #[test]
    fn reconcile_diff_finds_changes_and_removals() {
        let orch = temp_orchestrator();
        assert!(orch.reconcile_diff(&BTreeMap::new()).is_none()); // no prior state
        {
            let mut shared = orch.shared.lock().unwrap();
            shared.note_hashes = Some(BTreeMap::from([
                (1, hash_text("one")),
                (2, hash_text("two")),
                (3, hash_text("three")),
            ]));
        }
        let new = BTreeMap::from([
            (1, hash_text("one")),      // unchanged
            (2, hash_text("two-edit")), // changed
            (4, hash_text("four")),     // new
        ]);
        let (to_embed, to_remove) = orch.reconcile_diff(&new).unwrap();
        assert_eq!(to_embed, vec![2, 4]);
        assert_eq!(to_remove, vec![3]);
    }

    #[test]
    fn saver_debounces_and_bursts() {
        let orch = temp_orchestrator();
        let timers = Arc::new(ManualTimers {
            jobs: Mutex::new(Vec::new()),
        });
        let saver = DebouncedSaver::new(orch, Arc::clone(&timers) as Arc<dyn TimerHost>, 60.0, 3);
        saver.request_save();
        saver.request_save();
        assert_eq!(saver.pending_changes(), 2);
        assert_eq!(timers.jobs.lock().unwrap().len(), 2); // re-armed per change
        saver.request_save(); // hits the burst cap → immediate flush
        assert_eq!(saver.pending_changes(), 0);
        // Firing a stale timer after the flush is harmless (a fresh save).
        let job = timers.jobs.lock().unwrap().pop().unwrap();
        job();
        assert_eq!(saver.pending_changes(), 0);
    }
}

// ── embed-coupled ops (#332, S3c-3) ──────────────────────────────────────────
// The orchestrator's write side, as kernel async fns over the Embedder seam
// (the harness's PyEmbedder, or any native impl). Background builds are plain
// futures the harness drives (an asyncio task over the bridge) — no
// spawn_compute primitive, per the runtime model.

use crate::{Embedder, ImageEmbedder, ImageResolver, MediaItem};

/// Calibration parameters (mirror `index.CALIB_SAMPLE` / `CALIB_MIN`).
pub const CALIB_SAMPLE: usize = 256;
pub const CALIB_MIN: usize = 30;
/// Embed/add chunk size (mirrors `index.BATCH_SIZE`).
pub const BATCH_SIZE: usize = 64;
const TEXT: &str = "text";

/// One note's embedding inputs (mirrors the Python `NoteEmbedInput`).
#[derive(Debug, Clone)]
pub struct EmbedInput {
    pub note_id: i64,
    pub text: String,
    pub image_names: Vec<String>,
    /// Recognized, vector-worthy texts for this note (#199/#228): each mints
    /// its own TEXT-space vector under the note key (no modality gap — the
    /// text encoder reads them like any text), so a note ranks by
    /// max-over-items. Empty for notes without recognition — the hash stays
    /// byte-identical to the pre-OCR scheme, so upgrades never rebuild.
    pub ocr_texts: Vec<String>,
}

// The media-resolver and image-embedder seams live in shrike-engine-api
// (#342); the orchestrator consumes them through the crate-root re-exports.

impl IndexOrchestrator {
    fn hash_for(
        &self,
        input: &EmbedInput,
        images: Option<(&dyn ImageEmbedder, &dyn ImageResolver)>,
    ) -> String {
        let embeds_images = images.is_some();
        let exists = |name: &str| images.map(|(_, r)| r.exists(name)).unwrap_or(false);
        note_hash(
            &input.text,
            &input.image_names,
            embeds_images,
            &exists,
            &input.ocr_texts,
        )
    }

    /// Embed and (replace-)add notes — text vectors always; one vector per
    /// *resolvable* image when an image embedder + resolver are supplied.
    /// Maintains the per-note hashes (so the next reconcile sees these notes
    /// as current). Mirrors `VectorIndex.add`.
    pub async fn add(
        &self,
        inputs: &[EmbedInput],
        embedder: &dyn Embedder,
        images: Option<(&dyn ImageEmbedder, &dyn ImageResolver)>,
    ) -> NativeResult<usize> {
        if inputs.is_empty() {
            return Ok(0);
        }
        let mut added = 0usize;
        for batch in inputs.chunks(BATCH_SIZE) {
            let texts: Vec<String> = batch.iter().map(|i| i.text.clone()).collect();
            let keys: Vec<i64> = batch.iter().map(|i| i.note_id).collect();

            // Image bytes, read lazily (only for notes being added) — keys
            // built in lockstep with the items so they align with vectors.
            let mut image_keys: Vec<i64> = Vec::new();
            let mut items: Vec<MediaItem> = Vec::new();
            if let Some((_, resolver)) = images {
                for input in batch {
                    for name in &input.image_names {
                        if let Some(data) = resolver.read(name) {
                            image_keys.push(input.note_id);
                            items.push(MediaItem::from_named(name, data));
                        }
                    }
                }
            }

            // Recognized-text vectors (#199/#228): each vector-worthy OCR/ASR
            // text embeds via the SAME text encoder, landing in the text
            // region with no modality gap, keyed by its note — so the
            // per-modality ranking is max-over-items for free.
            let mut ocr_keys: Vec<i64> = Vec::new();
            let mut ocr_texts: Vec<String> = Vec::new();
            for input in batch {
                for t in &input.ocr_texts {
                    ocr_keys.push(input.note_id);
                    ocr_texts.push(t.clone());
                }
            }

            // The batch's engine work is independent — text, recognized-text,
            // and image vectors share no ordering — so all futures are built
            // BEFORE any await and joined. The kernel only states the
            // independence; whether they truly overlap is the host's lane
            // assignment (two engines on one GPU share a narrow lane, a
            // remote engine gets a wide one).
            let text_fut = embedder.embed(texts);
            let ocr_fut = async {
                if ocr_texts.is_empty() {
                    Ok(Vec::new())
                } else {
                    embedder.embed(ocr_texts).await
                }
            };
            let image_fut = async {
                match images {
                    Some((image_embedder, _)) if !items.is_empty() => {
                        image_embedder.embed_images(items).await
                    }
                    _ => Ok(Vec::new()),
                }
            };
            let (vectors, ocr_vectors, image_vectors) =
                futures::try_join!(text_fut, ocr_fut, image_fut)?;
            let ndim = vectors.first().map(Vec::len).unwrap_or(0);
            if ndim == 0 {
                return Err(NativeError::internal("embedder returned empty vectors"));
            }

            self.engine.ensure(TEXT, ndim)?;
            // Replace semantics: drop a re-added note's existing vectors first.
            let _ = self.engine.remove(&keys)?;
            self.engine.add(TEXT, &keys, &vectors)?;
            if !ocr_keys.is_empty() {
                self.engine.add(TEXT, &ocr_keys, &ocr_vectors)?;
            }
            if !image_keys.is_empty() {
                let image_ndim = image_vectors.first().map(Vec::len).unwrap_or(0);
                self.engine.ensure("image", image_ndim)?;
                self.engine.add("image", &image_keys, &image_vectors)?;
            }

            // Hash OUTSIDE the lock: the resolver's `exists` may be a harness
            // call (a stat, or Python via the injected resolver) — never run
            // foreign code under the orchestrator mutex.
            let new_hashes: Vec<(i64, String)> = batch
                .iter()
                .map(|input| (input.note_id, self.hash_for(input, images)))
                .collect();
            let mut shared = self.shared.lock().expect("orchestrator poisoned");
            if let Some(hashes) = shared.note_hashes.as_mut() {
                hashes.extend(new_hashes);
            }
            drop(shared);
            added += batch.len();
        }
        Ok(added)
    }

    /// Remove every vector for the given notes (all modalities) and their
    /// hashes. Returns the count of notes actually present (text removals).
    pub fn remove(&self, note_ids: &[i64]) -> NativeResult<usize> {
        if note_ids.is_empty() {
            return Ok(0);
        }
        let removed = self.engine.remove(note_ids)?;
        let mut shared = self.shared.lock().expect("orchestrator poisoned");
        if let Some(hashes) = shared.note_hashes.as_mut() {
            for nid in note_ids {
                hashes.remove(nid);
            }
        }
        Ok(removed)
    }

    /// Create an empty, ready index for an empty collection (#148 semantics).
    pub fn materialize_empty(&self, ndim: usize, col_mod: i64, model_id: Option<&str>) {
        if self.engine.ndim().is_some() {
            return; // an index already exists — drift handling is reconcile's job
        }
        if self.engine.ensure(TEXT, ndim).is_err() {
            return;
        }
        let mut shared = self.shared.lock().expect("orchestrator poisoned");
        if shared.note_hashes.is_none() {
            shared.note_hashes = Some(BTreeMap::new());
        }
        shared.col_mod = Some(col_mod);
        shared.model_id = model_id.map(str::to_owned);
        shared.state = OrchestratorState::Ready;
        drop(shared);
        let _ = self.save();
    }

    /// Full rebuild: clear, re-embed everything, recalibrate, persist.
    /// Mirrors `VectorIndex.rebuild` (progress lands in `build_progress`).
    pub async fn rebuild(
        &self,
        inputs: Vec<EmbedInput>,
        col_mod: i64,
        model_id: Option<String>,
        embedder: &dyn Embedder,
        images: Option<(&dyn ImageEmbedder, &dyn ImageResolver)>,
    ) -> NativeResult<()> {
        let total = inputs.len() as u64;
        {
            let mut shared = self.shared.lock().expect("orchestrator poisoned");
            shared.state = OrchestratorState::Building;
            shared.build_progress = (0, total);
            shared.error = None;
            shared.schema = INDEX_SCHEMA_VERSION; // rebuilds land on the current layout
            shared.note_hashes = Some(BTreeMap::new());
        }
        self.engine.clear();

        let result: NativeResult<()> = async {
            let mut indexed = 0u64;
            for batch in inputs.chunks(BATCH_SIZE) {
                self.add(batch, embedder, images).await?;
                indexed += batch.len() as u64;
                self.shared
                    .lock()
                    .expect("orchestrator poisoned")
                    .build_progress = (indexed, total);
            }
            {
                let mut shared = self.shared.lock().expect("orchestrator poisoned");
                shared.col_mod = Some(col_mod);
                shared.model_id = model_id;
            }
            self.calibrate_activation()?;
            self.save()?;
            Ok(())
        }
        .await;

        let mut shared = self.shared.lock().expect("orchestrator poisoned");
        match &result {
            Ok(()) => shared.state = OrchestratorState::Ready,
            Err(e) => {
                shared.state = OrchestratorState::Error;
                shared.error = Some(format!("{e:?}"));
            }
        }
        result
    }

    /// Incremental reconcile: re-embed only changed/new notes, drop vanished
    /// ones; end state identical to a full rebuild over the same inputs.
    /// Falls back to `rebuild` when there is no prior per-note state.
    /// Mirrors `VectorIndex.reconcile`.
    pub async fn reconcile(
        &self,
        inputs: Vec<EmbedInput>,
        col_mod: i64,
        model_id: Option<String>,
        embedder: &dyn Embedder,
        images: Option<(&dyn ImageEmbedder, &dyn ImageResolver)>,
    ) -> NativeResult<()> {
        let new_hashes: BTreeMap<i64, String> = inputs
            .iter()
            .map(|i| (i.note_id, self.hash_for(i, images)))
            .collect();
        let Some((to_embed, to_remove)) = self.reconcile_diff(&new_hashes) else {
            return self
                .rebuild(inputs, col_mod, model_id, embedder, images)
                .await;
        };
        if to_embed.is_empty() && to_remove.is_empty() {
            // A non-embedding edit bumped col.mod: advance the watermark only.
            let mut shared = self.shared.lock().expect("orchestrator poisoned");
            shared.col_mod = Some(col_mod);
            drop(shared);
            self.save()?;
            return Ok(());
        }
        {
            let mut shared = self.shared.lock().expect("orchestrator poisoned");
            shared.state = OrchestratorState::Building;
            shared.build_progress = (0, to_embed.len() as u64);
        }
        let result: NativeResult<()> = async {
            if !to_remove.is_empty() {
                self.remove(&to_remove)?;
            }
            let embed_set: std::collections::HashSet<i64> = to_embed.iter().copied().collect();
            let changed: Vec<EmbedInput> = inputs
                .into_iter()
                .filter(|i| embed_set.contains(&i.note_id))
                .collect();
            self.add(&changed, embedder, images).await?;
            {
                let mut shared = self.shared.lock().expect("orchestrator poisoned");
                shared.note_hashes = Some(new_hashes);
                shared.col_mod = Some(col_mod);
                shared.model_id = model_id;
            }
            self.calibrate_activation()?;
            self.save()?;
            Ok(())
        }
        .await;
        let mut shared = self.shared.lock().expect("orchestrator poisoned");
        match &result {
            Ok(()) => shared.state = OrchestratorState::Ready,
            Err(e) => {
                shared.state = OrchestratorState::Error;
                shared.error = Some(format!("{e:?}"));
            }
        }
        result
    }

    /// Recompute the #201b activation stats from the engine's own sampling
    /// (text-only collections get none — the gate stays off).
    pub fn calibrate_activation(&self) -> NativeResult<()> {
        let stats = self
            .engine
            .calibrate_activation(CALIB_SAMPLE, 1, CALIB_MIN)?;
        let mut shared = self.shared.lock().expect("orchestrator poisoned");
        shared.activation = Some(
            stats
                .into_iter()
                .map(|(modality, n, mean, std)| {
                    (
                        modality,
                        BTreeMap::from([
                            ("n".to_owned(), n),
                            ("mean".to_owned(), mean),
                            ("std".to_owned(), std),
                        ]),
                    )
                })
                .collect(),
        );
        Ok(())
    }

    /// The status block (state, size, progress, stamps) for the harness.
    pub fn status(&self) -> serde_json::Value {
        let shared = self.shared.lock().expect("orchestrator poisoned");
        let state = match shared.state {
            OrchestratorState::Ready => "ready",
            OrchestratorState::Building => "building",
            OrchestratorState::Unavailable => "unavailable",
            OrchestratorState::Error => "error",
        };
        serde_json::json!({
            "state": state,
            "size": self.engine.size(),
            "ndim": self.engine.ndim(),
            "col_mod": shared.col_mod,
            "model_id": shared.model_id,
            "progress": {"indexed": shared.build_progress.0, "total": shared.build_progress.1},
            "error": shared.error,
            "activation": shared.activation,
        })
    }
}

#[cfg(test)]
mod op_tests {
    use super::*;
    use futures::executor::block_on;
    use futures::future::BoxFuture;

    /// Deterministic 4-dim unit vectors from a text hash (no model needed).
    struct StubEmbedder;
    impl Embedder for StubEmbedder {
        fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            Box::pin(async move {
                Ok(texts
                    .iter()
                    .map(|t| {
                        let h = hash_text(t);
                        let b = u8::from_str_radix(&h[..2], 16).unwrap() as f32 / 255.0;
                        let n = (b * b + 1.0).sqrt();
                        vec![b / n, 1.0 / n, 0.0, 0.0]
                    })
                    .collect())
            })
        }
    }

    fn temp_orch() -> Arc<IndexOrchestrator> {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "shrike-orch-ops-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let engine =
            Arc::new(MultiModalIndex::new(vec![TEXT.to_owned(), "image".to_owned()]).unwrap());
        Arc::new(IndexOrchestrator::open(dir, engine))
    }

    fn input(nid: i64, text: &str) -> EmbedInput {
        EmbedInput {
            note_id: nid,
            text: text.to_owned(),
            image_names: vec![],
            ocr_texts: vec![],
        }
    }

    /// Event log for the overlap pin: (engine, edge) pairs in arrival order.
    type Events = Arc<std::sync::Mutex<Vec<(&'static str, &'static str)>>>;

    /// A thread-backed slow engine: compute starts when the future is BUILT
    /// (the eager-adapter shape — `OnExecutor` submits at call time) and
    /// completes through a oneshot ~50ms later, recording start/end edges.
    struct SlowEngine {
        name: &'static str,
        events: Events,
    }

    impl SlowEngine {
        fn run(&self, n: usize) -> BoxFuture<'static, NativeResult<Vec<Vec<f32>>>> {
            let (tx, rx) = futures::channel::oneshot::channel();
            let (name, events) = (self.name, Arc::clone(&self.events));
            events.lock().unwrap().push((name, "start"));
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                events.lock().unwrap().push((name, "end"));
                let _ = tx.send(Ok(vec![vec![1.0, 0.0, 0.0, 0.0]; n]));
            });
            Box::pin(async move {
                rx.await
                    .map_err(|_| NativeError::internal("slow engine dropped"))?
            })
        }
    }

    impl Embedder for SlowEngine {
        fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            self.run(texts.len())
        }
    }

    impl ImageEmbedder for SlowEngine {
        fn embed_images(
            &self,
            images: Vec<MediaItem>,
        ) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            self.run(images.len())
        }
    }

    struct AlwaysResolver;
    impl ImageResolver for AlwaysResolver {
        fn read(&self, _name: &str) -> Option<Vec<u8>> {
            Some(vec![1, 2, 3])
        }
        fn exists(&self, _name: &str) -> bool {
            true
        }
    }

    /// The pipeline-parallelism pin (#342 P2): `add` builds the batch's text
    /// and image engine futures BEFORE awaiting either, so two engines whose
    /// compute runs off-thread overlap. The old text-then-image sequential
    /// awaits would start the image engine only after the text engine
    /// finished — the interleaving below would be impossible.
    #[test]
    fn batch_engine_futures_overlap() {
        let events: Events = Arc::default();
        let text = SlowEngine {
            name: "text",
            events: Arc::clone(&events),
        };
        let image = SlowEngine {
            name: "image",
            events: Arc::clone(&events),
        };
        let orch = temp_orch();
        let mut one = input(1, "alpha");
        one.image_names = vec!["a.png".to_owned()];
        block_on(orch.add(&[one], &text, Some((&image, &AlwaysResolver)))).unwrap();

        let log = events.lock().unwrap().clone();
        let pos = |engine: &str, edge: &str| {
            log.iter()
                .position(|(n, e)| *n == engine && *e == edge)
                .unwrap_or_else(|| panic!("missing {engine}:{edge} in {log:?}"))
        };
        assert!(
            pos("image", "start") < pos("text", "end"),
            "image embed must start while the text embed is in flight: {log:?}"
        );
        assert!(
            pos("text", "start") < pos("image", "end"),
            "text embed must start while the image embed is in flight: {log:?}"
        );
        // And both actually landed: a text vector + an image vector.
        assert_eq!(orch.engine().size(), 2);
    }

    #[test]
    fn rebuild_then_incremental_add_and_remove() {
        let orch = temp_orch();
        block_on(orch.rebuild(
            vec![input(1, "alpha"), input(2, "beta")],
            10,
            Some("m".into()),
            &StubEmbedder,
            None,
        ))
        .unwrap();
        assert_eq!(orch.state(), OrchestratorState::Ready);
        assert_eq!(orch.engine().size(), 2);
        assert_eq!(orch.col_mod(), Some(10));
        assert!(!orch.check_drift(10, Some("m"), false));

        block_on(orch.add(&[input(3, "gamma")], &StubEmbedder, None)).unwrap();
        assert_eq!(orch.engine().size(), 3);
        assert_eq!(orch.remove(&[1, 99]).unwrap(), 1); // 99 not present
        assert_eq!(orch.engine().size(), 2);
    }

    #[test]
    fn reconcile_matches_rebuild_end_state() {
        // The invariant the per-note hashes exist for: an incremental
        // reconcile lands on the identical state a full rebuild would.
        let inputs_v1 = vec![input(1, "one"), input(2, "two"), input(3, "three")];
        let inputs_v2 = vec![input(1, "one"), input(2, "two EDITED"), input(4, "four")];

        let reconciled = temp_orch();
        block_on(reconciled.rebuild(inputs_v1, 1, Some("m".into()), &StubEmbedder, None)).unwrap();
        block_on(reconciled.reconcile(inputs_v2.clone(), 2, Some("m".into()), &StubEmbedder, None))
            .unwrap();

        let rebuilt = temp_orch();
        block_on(rebuilt.rebuild(inputs_v2, 2, Some("m".into()), &StubEmbedder, None)).unwrap();

        assert_eq!(reconciled.engine().size(), rebuilt.engine().size());
        let mut a = reconciled.engine().keys();
        let mut b = rebuilt.engine().keys();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        for key in a {
            assert_eq!(reconciled.engine().get(key), rebuilt.engine().get(key));
        }
    }

    #[test]
    fn watermark_only_drift_advances_without_embedding() {
        let orch = temp_orch();
        let inputs = vec![input(1, "alpha")];
        block_on(orch.rebuild(inputs.clone(), 1, Some("m".into()), &StubEmbedder, None)).unwrap();
        // Same content, new col_mod (a tags/deck edit): no re-embed, stamp moves.
        block_on(orch.reconcile(inputs, 2, Some("m".into()), &StubEmbedder, None)).unwrap();
        assert_eq!(orch.col_mod(), Some(2));
        assert_eq!(orch.engine().size(), 1);
    }

    #[test]
    fn persistence_round_trips_without_drift() {
        let orch = temp_orch();
        block_on(orch.rebuild(
            vec![input(1, "alpha"), input(2, "beta")],
            7,
            Some("m".into()),
            &StubEmbedder,
            None,
        ))
        .unwrap();
        let dir = orch.dir.clone();
        drop(orch);

        let engine =
            Arc::new(MultiModalIndex::new(vec![TEXT.to_owned(), "image".to_owned()]).unwrap());
        let reopened = IndexOrchestrator::open(dir, engine);
        assert_eq!(reopened.engine().size(), 2);
        assert_eq!(reopened.col_mod(), Some(7));
        assert!(!reopened.check_drift(7, Some("m"), false));
        // The hashes sidecar survived too: a same-content reconcile is a no-op.
        let (to_embed, to_remove) = reopened
            .reconcile_diff(&BTreeMap::from([
                (1, hash_text("alpha")),
                (2, hash_text("beta")),
            ]))
            .unwrap();
        assert!(to_embed.is_empty() && to_remove.is_empty());
    }

    #[test]
    fn materialize_empty_is_ready_and_stamped() {
        let orch = temp_orch();
        orch.materialize_empty(4, 3, Some("m"));
        assert_eq!(orch.state(), OrchestratorState::Ready);
        assert!(!orch.check_drift(3, Some("m"), false));
        // Later notes index incrementally (the #148 behaviour).
        block_on(orch.add(&[input(1, "late")], &StubEmbedder, None)).unwrap();
        assert_eq!(orch.engine().size(), 1);
    }
}
