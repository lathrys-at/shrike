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
) -> String {
    if embeds_images {
        let mut present: Vec<&str> = image_names
            .iter()
            .map(String::as_str)
            .filter(|n| image_exists(n))
            .collect();
        if !present.is_empty() {
            present.sort_unstable();
            let joined = format!("{text}\u{1f}{}", present.join("\u{1f}"));
            return hash_text(&joined);
        }
    }
    hash_text(text)
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
    /// Populated by the embed-coupled ops (the S3c-3 slice); read by status().
    #[allow(dead_code)]
    build_progress: (u64, u64),
    #[allow(dead_code)]
    error: Option<String>,
}

/// The embed-free orchestration core over one engine + its sidecars.
pub struct IndexOrchestrator {
    dir: PathBuf,
    engine: MultiModalIndex,
    shared: Mutex<Shared>,
}

impl IndexOrchestrator {
    /// Create over a directory, loading any existing on-disk index + sidecars
    /// (the Python `_load` semantics: corrupt meta → unloaded; corrupt/missing
    /// hashes → `None` (rebuild-on-reconcile); engine restore failure clears
    /// both so drift forces a full rebuild).
    pub fn open(dir: impl Into<PathBuf>, engine: MultiModalIndex) -> Self {
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
        assert_eq!(note_hash("t", &names, false, &always), hash_text("t"));
        // Image backend + nothing resolves → still the bare text hash.
        assert_eq!(note_hash("t", &names, true, &never), hash_text("t"));
        // Resolvable images fold in, sorted (matches the Python scheme).
        let folded = note_hash("t", &names, true, &always);
        assert_eq!(folded, hash_text("t\u{1f}a.png\u{1f}b.png"));
        assert_ne!(folded, hash_text("t"));
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
        let engine = MultiModalIndex::new(vec!["text".to_owned()]).unwrap();
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
