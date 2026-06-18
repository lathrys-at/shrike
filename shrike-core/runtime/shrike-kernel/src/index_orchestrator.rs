//! The index orchestrator (#332, S3c-2): the `VectorIndex` *policy* re-homed.
//!
//! The engine (`shrike-index`) has owned storage since #273; this module owns
//! what the Python orchestrator owned — the meta/hashes sidecars, the drift
//! rules, the per-note change fingerprints, the state machine, and the
//! debounced saver (tokio::time, #374). File formats are
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
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use blake2::digest::consts::U8;
use blake2::{Blake2b, Digest};
use serde::{Deserialize, Serialize};

use shrike_error::{ErrorKind, NativeError, NativeResult, ResultExt};
use shrike_store::VectorIndex;

/// The on-disk schema version of a freshly built index (mirrors
/// `index.INDEX_SCHEMA_VERSION`: v2 = per-modality sub-indexes, #201a).
pub const INDEX_SCHEMA_VERSION: i64 = 2;

/// Saver defaults (mirrors `index.DEFAULT_SAVE_DELAY` / `_THRESHOLD`).
///
/// Idle-debounce window: flush this many seconds after the last change.
pub const DEFAULT_SAVE_DELAY: f64 = 60.0;
/// Burst cap: flush immediately once this many unsaved changes accumulate.
pub const DEFAULT_SAVE_THRESHOLD: u64 = 100;

/// Stable fingerprint of a note's embedding text — bit-identical to Python's
/// `hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()`. The
/// fixed-size `Blake2b<U8>` folds the same digest length into the parameter
/// block as `Blake2bVar::new(8)` did (same bytes, no per-call fallible
/// construction on the hot hash path — #382; the parity tests pin it).
pub fn hash_text(text: &str) -> String {
    let out = Blake2b::<U8>::digest(text.as_bytes());
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

/// What a space embeds + hashes on the write path (#232's per-modality-primary
/// routing). The existing `add`/`reconcile`/`rebuild` default to
/// [`WriteMode::TextAndImage`] (byte-identical to pre-#232: text always, image
/// when the space carries an image pair); a SECONDARY image-primary space uses
/// [`WriteMode::ImageOnly`] so it stores only image vectors and hashes only its
/// image refs — a pure-text edit then never re-embeds it, and the dedicated
/// text space's stored vectors are the only text vectors (no waste, no
/// double-index).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteMode {
    /// Embed text (+ image when an image pair is supplied) and fold both into
    /// the per-note hash. The N=1 / omni-primary / dedicated-text-primary path
    /// — byte-identical to the single-orchestrator era.
    TextAndImage,
    /// Embed ONLY images, store ONLY image vectors, and hash ONLY the present
    /// image refs (#232). The image-primary SECONDARY space's path: a pure-text
    /// edit leaves its hash unchanged (no spurious re-embed); an image
    /// add/remove/swap re-embeds it.
    ImageOnly,
}

/// The per-note change fingerprint for an [`WriteMode::ImageOnly`] space (#232):
/// folds ONLY the present-and-resolvable image refs — no text, no OCR — so the
/// hash moves iff the note's images do. An image-primary space with no
/// resolvable images for a note hashes the empty-image sentinel (a stable
/// constant), so a note that loses its last image still re-embeds (its vector
/// must leave the image index) and a never-imaged note has a stable hash.
pub fn note_hash_images_only(
    image_names: &[String],
    image_exists: &dyn Fn(&str) -> bool,
) -> String {
    let mut present: Vec<&str> = image_names
        .iter()
        .map(String::as_str)
        .filter(|n| image_exists(n))
        .collect();
    present.sort_unstable();
    // A distinct namespace prefix so an image-only hash never collides with a
    // text hash of the same bytes (defensive; the two live in separate spaces'
    // sidecars anyway).
    let joined = format!("\u{1f}img\u{1f}{}", present.join("\u{1f}"));
    hash_text(&joined)
}

/// `index.meta.json` — the exact shape the Python orchestrator wrote, plus the
/// owning-collection identity (#67).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMeta {
    /// The text-modality vector width (the index's primary dimension).
    pub ndim: i64,
    /// The `col.mod` at last build (absent before the first build).
    #[serde(default)]
    pub col_mod: Option<i64>,
    /// The embedding model fingerprint at last build (absent if unset).
    #[serde(default)]
    pub model_id: Option<String>,
    /// Marker absent → pre-#201a (v1) single-index layout.
    #[serde(default = "default_schema_v1")]
    pub schema: i64,
    /// The owning collection's path-derived identity (#67): the canonicalized
    /// collection path. Recorded so a mismatched/moved cache (the index dir
    /// belongs to a different collection) is detected and rebuilt rather than
    /// silently reused. Absent → a pre-#67 index (loaded as-is; the next save
    /// stamps the owner). Serialized last and skipped when absent so a text-
    /// only index built before #67 round-trips byte-identically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
    /// Presence (even empty) records that calibration ran (one-shot).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation: Option<ActivationStats>,
}

/// The #201b calibration stats, keyed by modality (`image`). Replaces the old
/// `BTreeMap<String, f64>` value with magic `"n"/"mean"/"std"` keys — a typo'd
/// key no longer compiles, and the per-modality stat is a named struct rather
/// than three string lookups.
///
/// On-disk/wire format is UNCHANGED: the outer key→value map stays a
/// `BTreeMap` (sorted-key JSON object), and [`ActivationStat`] serializes its
/// three fields in `mean, n, std` order — byte-for-byte the sorted order the
/// old `BTreeMap<String, f64>` emitted. `n` stays `f64` so it round-trips the
/// Python shape (`{"n": 40.0, ...}`, not `40`).
pub type ActivationStats = BTreeMap<String, ActivationStat>;

/// One modality's calibrated best-match distribution (#201b): the sample count
/// and the `(mean, std)` the host-side activation floor reads as
/// `mean + margin·std`. All three are `f64` — `n` included — to round-trip the
/// Python on-disk shape byte-for-byte (it wrote `n` as a float).
///
/// Field order is deliberate (`mean, n, std`): serde serializes struct fields
/// in declaration order, and these match the alphabetical key order the prior
/// `BTreeMap<String, f64>` produced, so the serialized bytes are identical.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ActivationStat {
    /// Mean of the modality's typical best-match cosine.
    pub mean: f64,
    /// The sample count (`f64` to round-trip the Python on-disk shape).
    pub n: f64,
    /// Standard deviation of the best-match cosine.
    pub std: f64,
}

fn default_schema_v1() -> i64 {
    1
}

/// Whether a recorded owner conflicts with the expected one (#67). A conflict
/// is only when BOTH are known and they differ — an absent expected owner (the
/// injection seam) never conflicts, and an absent recorded owner (a pre-#67
/// index) is adopted, never rejected.
fn owner_mismatch(expected: Option<&str>, recorded: Option<&str>) -> bool {
    matches!((expected, recorded), (Some(e), Some(r)) if e != r)
}

/// Write via a same-directory `.tmp` + rename — atomic on one filesystem, so
/// a crash mid-write leaves the old file complete, never a torn one (#381).
fn write_atomic(path: &Path, contents: &str) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, contents)?;
    std::fs::rename(&tmp, path)
}

/// The index build/availability state (mirrors `IndexState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OrchestratorState {
    /// The index is built and serving search.
    Ready,
    /// A rebuild/reconcile is in progress (see [`BuildProgress`]).
    Building,
    /// No embedder attached — the index loads on-disk vectors but cannot grow.
    Unavailable,
    /// A build failed (the error is carried in [`OrchestratorStatus`]).
    Error,
}

/// The status block (state, size, progress, stamps) the harness serves —
/// typed since #391; the binding serializes once at the edge.
#[derive(Debug, Clone, Serialize)]
pub struct OrchestratorStatus {
    /// The current build/availability state.
    pub state: OrchestratorState,
    /// Total vectors across all sub-indexes (text-modality note count).
    pub size: usize,
    /// The text-modality vector width (`None` before any width is set).
    pub ndim: Option<usize>,
    /// The `col.mod` at last build.
    pub col_mod: Option<i64>,
    /// The embedding model fingerprint at last build.
    pub model_id: Option<String>,
    /// Build progress (indexed so far / total planned).
    pub progress: BuildProgress,
    /// The build error message when `state` is [`OrchestratorState::Error`].
    pub error: Option<String>,
    /// Per-modality calibration stats, once calibration has run.
    pub activation: Option<ActivationStats>,
    /// Per-modality sub-index breakdown (#684): the text and image sub-indexes
    /// report their OWN size/ndim, which the aggregate `size`/`ndim` above
    /// (sum / text-modality width) can't express. Ordered text-first.
    pub modalities: Vec<ModalityStat>,
}

/// One sub-index's `(name, size, ndim)` for the status breakdown (#684).
/// `ndim` is `None` for an empty sub-index (width not yet set).
#[derive(Debug, Clone, Serialize)]
pub struct ModalityStat {
    /// The sub-index name (e.g. `text`, `image`).
    pub modality: String,
    /// Vectors in this sub-index.
    pub size: usize,
    /// This sub-index's vector width (`None` until its width is set).
    pub ndim: Option<usize>,
}

/// `status()`'s build progress pair (indexed so far / total planned).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BuildProgress {
    /// Notes embedded so far in the current build.
    pub indexed: u64,
    /// Total notes planned for the current build.
    pub total: u64,
}

struct Shared {
    state: OrchestratorState,
    col_mod: Option<i64>,
    model_id: Option<String>,
    schema: i64,
    /// The owning collection's identity (#67), written into the meta so a
    /// moved/mismatched cache is detected. `None` only on the injection seam
    /// (a `compose`d kernel has no collection path) — then nothing is stamped
    /// and ownership is unenforced, exactly as before #67.
    owner: Option<String>,
    activation: Option<ActivationStats>,
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
    engine: Arc<dyn VectorIndex>,
    shared: Mutex<Shared>,
    /// Serializes SAVERS only (#445): `save` snapshots `shared` briefly and
    /// writes files OUTSIDE that lock, so two concurrent saves would race the
    /// deterministic tmp paths without this. Ops/status never take it.
    save_guard: Mutex<()>,
}

impl IndexOrchestrator {
    /// Create over a directory, loading any existing on-disk index + sidecars
    /// (the Python `_load` semantics: corrupt meta → unloaded; corrupt/missing
    /// hashes → `None` (rebuild-on-reconcile); engine restore failure clears
    /// both so drift forces a full rebuild).
    pub fn open(dir: impl Into<PathBuf>, engine: Arc<dyn VectorIndex>) -> Self {
        Self::open_owned(dir, engine, None)
    }

    /// [`Self::open`] recording the owning collection's identity (#67). On load
    /// a meta whose recorded owner differs from `owner` is a foreign/moved
    /// cache: the index is left unloaded (stamps cleared) so the next drift
    /// check rebuilds it for this collection, never silently serving another
    /// collection's vectors. A meta with no owner (a pre-#67 index) loads as-is
    /// and the next save stamps `owner` in. `owner = None` (the injection seam)
    /// disables the check entirely.
    pub fn open_owned(
        dir: impl Into<PathBuf>,
        engine: Arc<dyn VectorIndex>,
        owner: Option<String>,
    ) -> Self {
        let dir = dir.into();
        let mut shared = Shared {
            state: OrchestratorState::Unavailable,
            col_mod: None,
            model_id: None,
            schema: INDEX_SCHEMA_VERSION,
            owner: owner.clone(),
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
                Ok(meta) if owner_mismatch(owner.as_deref(), meta.collection.as_deref()) => {
                    // A different collection owns this index dir — refuse it.
                    // Leaving `shared` at its unloaded defaults means the next
                    // drift check sees "no index" and rebuilds for us.
                    tracing::warn!(
                        path = %meta_path.display(),
                        recorded = ?meta.collection,
                        expected = ?owner,
                        "index belongs to a different collection; ignoring (will rebuild)"
                    );
                }
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
                    // A non-UTF-8 cache dir can't be handed to the engine's
                    // path-taking API — treat it as a failed restore rather
                    // than silently retargeting to the cwd (#382).
                    let restored = match dir.to_str() {
                        Some(d) => engine.restore(d, candidates.as_deref()),
                        None => {
                            tracing::warn!(path = %dir.display(), "non-UTF-8 index dir; skipping restore");
                            false
                        }
                    };
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
            save_guard: Mutex::new(()),
        }
    }

    /// The backing vector engine (borrowed).
    pub fn engine(&self) -> &dyn VectorIndex {
        &*self.engine
    }

    /// The shared engine, for a harness search handle over the same vectors.
    pub fn engine_arc(&self) -> Arc<dyn VectorIndex> {
        Arc::clone(&self.engine)
    }

    /// Embedder detached: searchable vectors remain on disk/memory but the
    /// semantic surface is down (mirrors the Python `set_backend(None)`).
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
    pub fn mark_unavailable(&self) {
        self.shared.lock().expect("orchestrator poisoned").state = OrchestratorState::Unavailable;
    }

    /// Embedder (re)attached: flip back to ready ONLY from unavailable — a
    /// building or errored state is the (re)build path's to resolve.
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
    pub fn mark_ready_if_loaded(&self) {
        let mut shared = self.shared.lock().expect("orchestrator poisoned");
        if shared.state == OrchestratorState::Unavailable {
            shared.state = OrchestratorState::Ready;
        }
    }

    /// The drift policy (mirrors `VectorIndex.check_drift`): nothing loaded /
    /// no stamp / model change / a v1 layout under an image-capable backend →
    /// rebuild; a bare `col_mod` move → reconcile; otherwise current.
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
    #[must_use]
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
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
    #[allow(clippy::type_complexity)]
    #[must_use]
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

    /// The current build/availability state.
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
    pub fn state(&self) -> OrchestratorState {
        self.shared.lock().expect("orchestrator poisoned").state
    }

    /// The `col.mod` at last build (`None` before the first build).
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
    pub fn col_mod(&self) -> Option<i64> {
        self.shared.lock().expect("orchestrator poisoned").col_mod
    }

    /// Advance the stored `col.mod` watermark (a maintained write's tail).
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
    pub fn set_col_mod(&self, value: i64) {
        self.shared.lock().expect("orchestrator poisoned").col_mod = Some(value);
    }

    /// The embedding model fingerprint at last build (`None` if unset).
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
    pub fn model_id(&self) -> Option<String> {
        self.shared
            .lock()
            .expect("orchestrator poisoned")
            .model_id
            .clone()
    }

    /// Build progress as `(indexed, total)`.
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
    pub fn build_progress(&self) -> (u64, u64) {
        self.shared
            .lock()
            .expect("orchestrator poisoned")
            .build_progress
    }

    /// True when per-note fingerprints exist (an incremental reconcile is
    /// possible; absent → the next drift handles via full rebuild).
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
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
    ///
    /// Every artifact lands via tmp + rename, ordered engine → hashes → meta:
    /// meta is the commit point, so it must never describe vectors or hashes
    /// that aren't already durably on disk (#381).
    ///
    /// # Errors
    ///
    /// Returns an error if the cache dir is non-UTF-8 or cannot be created, the
    /// engine save fails, a sidecar cannot be encoded, or an artifact write
    /// fails.
    ///
    /// # Panics
    ///
    /// Panics if the save guard or the shared-state mutex is poisoned (a prior
    /// holder panicked).
    pub fn save(&self) -> NativeResult<()> {
        // Savers serialize against each other; everything else stays free.
        let _saving = self.save_guard.lock().expect("save guard poisoned");
        let Some(ndim) = self.engine.ndim() else {
            return Ok(()); // nothing built — nothing to persist
        };
        // SNAPSHOT under the shared lock, WRITE outside it (#445): the old
        // shape held `shared` across the multi-hundred-MB engine file write,
        // so every op tail's watermark update and every /status blocked for
        // the whole flush. The snapshot is cheap (clones of small fields +
        // the hash map stringify); the writes below run lock-free. Ordering
        // invariant (#381) holds: meta is the commit point and describes the
        // snapshot, which is ≤ the engine state written after it — a
        // too-old meta only costs an extra reconcile on load, never lies.
        let (hashes_json, meta) = {
            let shared = self.shared.lock().expect("orchestrator poisoned");
            let hashes_json = match &shared.note_hashes {
                Some(hashes) => {
                    let as_strings: BTreeMap<String, &String> =
                        hashes.iter().map(|(k, v)| (k.to_string(), v)).collect();
                    Some(
                        serde_json::to_string(&as_strings)
                            .context(ErrorKind::Internal, "hashes encode")?,
                    )
                }
                None => None,
            };
            let meta = IndexMeta {
                ndim: ndim as i64,
                col_mod: shared.col_mod,
                model_id: shared.model_id.clone(),
                schema: shared.schema,
                collection: shared.owner.clone(),
                activation: shared.activation.clone(),
            };
            (hashes_json, meta)
        };
        std::fs::create_dir_all(&self.dir).context(ErrorKind::Internal, "index dir")?;
        // A non-UTF-8 cache dir must error, not silently save to the cwd (#382).
        let dir_str = self.dir.to_str().ok_or_else(|| {
            NativeError::internal(format!("non-UTF-8 index dir: {}", self.dir.display()))
        })?;
        self.engine
            .save(dir_str)
            .context(ErrorKind::Internal, "engine save")?;
        if let Some(hashes_json) = hashes_json {
            write_atomic(&self.dir.join("index.hashes.json"), &hashes_json)
                .context(ErrorKind::Internal, "hashes write")?;
        }
        let meta_json = serde_json::to_string(&meta).context(ErrorKind::Internal, "meta encode")?;
        write_atomic(&self.dir.join("index.meta.json"), &meta_json)
            .context(ErrorKind::Internal, "meta write")?;
        Ok(())
    }

    /// Persist ONLY the meta sidecar (#445): the watermark-only reconcile
    /// path (a metadata-level col.mod bump — tags/decks/templates) changes no
    /// vector and no hash, but previously rewrote the full engine files to
    /// persist one i64. The vectors/hashes on disk are unchanged by
    /// definition on this path, so meta stays truthful.
    ///
    /// Caller contract: only sound when the on-disk engine/hashes already
    /// match memory — today's sole caller (reconcile's empty-diff branch) is
    /// reachable only after `open` loaded persisted hashes, which exist only
    /// if a prior full `save()` landed. A new caller on a path where `add`
    /// populated hashes without a persist would write a lying meta.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache dir cannot be created or the meta cannot be
    /// encoded/written.
    ///
    /// # Panics
    ///
    /// Panics if the save guard or the shared-state mutex is poisoned (a prior
    /// holder panicked).
    pub fn save_meta_only(&self) -> NativeResult<()> {
        let _saving = self.save_guard.lock().expect("save guard poisoned");
        let Some(ndim) = self.engine.ndim() else {
            return Ok(());
        };
        let meta = {
            let shared = self.shared.lock().expect("orchestrator poisoned");
            IndexMeta {
                ndim: ndim as i64,
                col_mod: shared.col_mod,
                model_id: shared.model_id.clone(),
                schema: shared.schema,
                collection: shared.owner.clone(),
                activation: shared.activation.clone(),
            }
        };
        std::fs::create_dir_all(&self.dir).context(ErrorKind::Internal, "index dir")?;
        let meta_json = serde_json::to_string(&meta).context(ErrorKind::Internal, "meta encode")?;
        write_atomic(&self.dir.join("index.meta.json"), &meta_json)
            .context(ErrorKind::Internal, "meta write")?;
        Ok(())
    }
}

/// The debounced saver on the kernel's own clock (#374 B2 — tokio::time;
/// mirrors `IndexSaver`): a flush `delay` seconds after the last change, or
/// immediately at `threshold` unsaved changes — whichever first. The armed
/// timer is one spawned task sleeping then flushing; re-arm aborts it (an
/// abort lands only at the sleep — once past it the flush completes, which
/// is fine: flush is idempotent).
pub struct DebouncedSaver {
    orchestrator: Arc<IndexOrchestrator>,
    delay: std::time::Duration,
    threshold: u64,
    pending: Mutex<PendingState>,
}

#[derive(Default)]
struct PendingState {
    changes: u64,
    armed: Option<tokio::task::AbortHandle>,
}

impl DebouncedSaver {
    /// Build a saver over `orchestrator` with the idle-debounce `delay_secs`
    /// (clamped non-negative) and the burst-cap `threshold`.
    pub fn new(orchestrator: Arc<IndexOrchestrator>, delay_secs: f64, threshold: u64) -> Arc<Self> {
        Arc::new(Self {
            orchestrator,
            delay: std::time::Duration::from_secs_f64(delay_secs.max(0.0)),
            threshold,
            pending: Mutex::new(PendingState::default()),
        })
    }

    /// Record a change: re-arm the idle timer, or flush now at the burst cap.
    ///
    /// # Panics
    ///
    /// Panics if the pending-state mutex is poisoned (a prior holder panicked).
    pub fn request_save(self: &Arc<Self>) {
        // abort-old + spawn + store happen under ONE lock scope: two
        // concurrent op tails re-arming must never each take a None and
        // leave an orphaned (un-abortable) timer behind. The spawn is
        // non-blocking, so holding the std mutex across it is fine.
        let mut pending = self.pending.lock().expect("saver poisoned");
        pending.changes += 1;
        if let Some(armed) = pending.armed.take() {
            armed.abort();
        }
        if pending.changes >= self.threshold {
            drop(pending);
            // Off the op tail (#445): the 100th upsert's caller previously
            // ate the entire multi-second file write inline.
            self.flush_background();
            return;
        }
        let this = Arc::clone(self);
        let delay = self.delay;
        let task = crate::runtime::handle().spawn(async move {
            tokio::time::sleep(delay).await;
            // Blocking fs work belongs on the blocking pool, not a runtime
            // worker (#445).
            this.flush_background();
        });
        pending.armed = Some(task.abort_handle());
    }

    /// Unsaved changes since the last flush.
    ///
    /// # Panics
    ///
    /// Panics if the pending-state mutex is poisoned (a prior holder panicked).
    pub fn pending_changes(&self) -> u64 {
        self.pending.lock().expect("saver poisoned").changes
    }

    /// Disarm the timer and zero the counter — synchronously, so
    /// `pending_changes` means "changes not yet handed to a flush" regardless
    /// of where the file write runs.
    fn reset_pending(&self) {
        let mut pending = self.pending.lock().expect("saver poisoned");
        if let Some(armed) = pending.armed.take() {
            // Possibly our own handle (the timer task flushing) — an
            // abort after the sleep is a no-op.
            armed.abort();
        }
        pending.changes = 0;
    }

    /// Flush synchronously: disarm the timer, zero the counter, and persist
    /// (a save failure is logged, not propagated — used at shutdown so close()
    /// returns only after the write lands).
    ///
    /// # Panics
    ///
    /// Panics if the pending-state mutex is poisoned (a prior holder panicked).
    pub fn flush(&self) {
        self.reset_pending();
        if let Err(e) = self.orchestrator.save() {
            tracing::warn!(error = ?e, "debounced index save failed");
        }
    }

    /// `flush`, but the file write rides the blocking pool — the timer and
    /// burst-threshold paths use this so neither a tokio worker nor an op
    /// tail carries the multi-second write (#445). The counter resets
    /// synchronously (same contract as `flush`); shutdown keeps the
    /// synchronous `flush` so close() returns only after the write lands.
    fn flush_background(self: &Arc<Self>) {
        self.reset_pending();
        let this = Arc::clone(self);
        crate::runtime::handle().spawn_blocking(move || {
            if let Err(e) = this.orchestrator.save() {
                tracing::warn!(error = ?e, "debounced index save failed");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shrike_index::MultiModalIndex;

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
        let stat = meta.activation.as_ref().unwrap().get("image").unwrap();
        assert_eq!(
            *stat,
            ActivationStat {
                mean: 0.2,
                n: 40.0,
                std: 0.05
            }
        );
        // The typed stat re-serializes byte-for-byte as the prior
        // BTreeMap<String, f64> did: keys in sorted order (mean, n, std) and
        // `n` as a float, so the on-disk activation block is unchanged.
        assert_eq!(
            serde_json::to_string(meta.activation.as_ref().unwrap()).unwrap(),
            r#"{"image":{"mean":0.2,"n":40.0,"std":0.05}}"#,
        );
        // A v1 meta (no schema marker) defaults to 1, like the Python load.
        let v1: IndexMeta = serde_json::from_str(r#"{"ndim": 8}"#).unwrap();
        assert_eq!(v1.schema, 1);
        assert!(v1.col_mod.is_none());
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
        // The threshold (burst) path is synchronous and deterministic; the
        // re-arm path is observable through pending_changes staying put while
        // the (long) idle timer never fires.
        let orch = temp_orchestrator();
        let saver = DebouncedSaver::new(orch, 60.0, 3);
        saver.request_save();
        saver.request_save();
        assert_eq!(saver.pending_changes(), 2); // armed, not flushed
        saver.request_save(); // hits the burst cap → immediate flush
        assert_eq!(saver.pending_changes(), 0);
        // Explicit flush after a fresh change cancels the armed timer.
        saver.request_save();
        assert_eq!(saver.pending_changes(), 1);
        saver.flush();
        assert_eq!(saver.pending_changes(), 0);
    }

    #[test]
    fn saver_idle_timer_actually_fires() {
        // The real-clock half (bounded, generous): a short delay flushes
        // without reaching the threshold.
        let orch = temp_orchestrator();
        let saver = DebouncedSaver::new(orch, 0.05, 100);
        saver.request_save();
        assert_eq!(saver.pending_changes(), 1);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while saver.pending_changes() != 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(saver.pending_changes(), 0, "the idle debounce flushed");
    }
}

// ── embed-coupled ops (#332, S3c-3) ──────────────────────────────────────────
// The orchestrator's write side, as kernel async fns over the Embedder seam
// (the harness's PyEmbedder, or any native impl). Background builds are plain
// futures the harness drives (an asyncio task over the bridge) — no
// spawn_compute primitive, per the runtime model.

use crate::{Embedder, ImageEmbedder, ImageResolver, MediaItem};

/// Calibration parameters (mirror `index.CALIB_SAMPLE` / `CALIB_MIN`).
///
/// How many stored text vectors to sample as pseudo-queries when calibrating.
pub const CALIB_SAMPLE: usize = 256;
/// Minimum usable samples below which calibration is skipped (the gate stays
/// disabled rather than calibrate off too few points).
pub const CALIB_MIN: usize = 30;
/// Per-sample search depth: must be ≥ 2 per the engine's contract — a
/// pseudo-query whose own image is the nearest hit needs a non-self hit to
/// record. At 1, self-hit-heavy collections (most notes carrying both text
/// and an image) silently shrink the sample below `CALIB_MIN` and the
/// activation gate disables exactly where it's needed (#446).
pub const CALIB_K: usize = 2;
/// Embed/add chunk size (mirrors `index.BATCH_SIZE`).
pub const BATCH_SIZE: usize = 64;
const TEXT: &str = "text";

/// One note's embedding inputs (mirrors the Python `NoteEmbedInput`).
#[derive(Debug, Clone)]
pub struct EmbedInput {
    /// The note this input embeds for (the index key).
    pub note_id: i64,
    /// The note's normalized embedding text.
    pub text: String,
    /// The note's referenced image filenames (resolved lazily at embed time).
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
        mode: WriteMode,
    ) -> String {
        let exists = |name: &str| images.map(|(_, r)| r.exists(name)).unwrap_or(false);
        match mode {
            // Image-only (#232): fold ONLY the present image refs — text edits
            // don't move it. (An ImageOnly space always carries an image pair.)
            WriteMode::ImageOnly => note_hash_images_only(&input.image_names, &exists),
            // Text (+image when paired): the existing media-aware hash — at N=1
            // this is byte-identical to pre-#232.
            WriteMode::TextAndImage => note_hash(
                &input.text,
                &input.image_names,
                images.is_some(),
                &exists,
                &input.ocr_texts,
            ),
        }
    }

    /// Embed and (replace-)add notes — text vectors always; one vector per
    /// *resolvable* image when an image embedder + resolver are supplied.
    /// Maintains the per-note hashes (so the next reconcile sees these notes
    /// as current). Mirrors `VectorIndex.add`.
    ///
    /// Visibility (#395): replace is remove-then-add across separate engine
    /// holds, NOT atomic — a concurrent search can transiently miss a note
    /// mid-replace (or see its text vector before its image one). That is
    /// inside the index's contract: a lag-tolerant derived cache whose
    /// results may be stale, never wrong-note.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding fails or the engine rejects the add.
    pub async fn add(
        &self,
        inputs: &[EmbedInput],
        embedder: &dyn Embedder,
        images: Option<(&dyn ImageEmbedder, &dyn ImageResolver)>,
    ) -> NativeResult<usize> {
        self.add_with_mode(inputs, embedder, images, WriteMode::TextAndImage)
            .await
    }

    /// [`Self::add`] with an explicit [`WriteMode`] (#232): the image-primary
    /// secondary space uses [`WriteMode::ImageOnly`] to store only image vectors
    /// + hash only image refs. `TextAndImage` is the byte-identical default.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding fails or the engine rejects the add.
    pub async fn add_with_mode(
        &self,
        inputs: &[EmbedInput],
        embedder: &dyn Embedder,
        images: Option<(&dyn ImageEmbedder, &dyn ImageResolver)>,
        mode: WriteMode,
    ) -> NativeResult<usize> {
        self.apply(inputs, embedder, images, true, mode).await
    }

    /// The shared embed→engine pipeline behind [`Self::add`] (replace
    /// semantics) and [`Self::rebuild`] (`replace = false`: the engine was
    /// just cleared, so per-batch removes against an empty index are skipped).
    async fn apply(
        &self,
        inputs: &[EmbedInput],
        embedder: &dyn Embedder,
        images: Option<(&dyn ImageEmbedder, &dyn ImageResolver)>,
        replace: bool,
        mode: WriteMode,
    ) -> NativeResult<usize> {
        if inputs.is_empty() {
            return Ok(0);
        }
        let mut added = 0usize;
        for batch in inputs.chunks(BATCH_SIZE) {
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

            let image_fut = async {
                match images {
                    Some((image_embedder, _)) if !items.is_empty() => {
                        image_embedder.embed_images(items).await
                    }
                    _ => Ok(Vec::new()),
                }
            };

            match mode {
                // ── Image-only (#232): embed + store ONLY image vectors. The
                // text/OCR halves are skipped entirely — a secondary CLIP space
                // never holds text vectors (PR-C reads only its image ranking).
                WriteMode::ImageOnly => {
                    let image_vectors = image_fut.await?;
                    // Replace: drop the note's prior vectors (all modalities)
                    // before re-adding (skipped during rebuild — just cleared).
                    if replace {
                        let _ = self.engine.remove(&keys)?;
                    }
                    if !image_keys.is_empty() {
                        let image_ndim = image_vectors.first().map(Vec::len).unwrap_or(0);
                        if image_ndim == 0 {
                            return Err(NativeError::internal(
                                "image embedder returned empty vectors",
                            ));
                        }
                        self.engine.ensure("image", image_ndim)?;
                        self.engine.add("image", &image_keys, &image_vectors)?;
                    }
                    // A note with no resolvable image adds nothing — but its
                    // hash still records (it's pending=current for reconcile).
                }
                // ── Text (+image when paired): the existing pipeline, verbatim
                // for N=1 / the text-or-omni primary.
                WriteMode::TextAndImage => {
                    let texts: Vec<String> = batch.iter().map(|i| i.text.clone()).collect();
                    // Recognized-text vectors (#199/#228): each vector-worthy
                    // OCR/ASR text embeds via the SAME text encoder, landing in
                    // the text region with no modality gap, keyed by its note.
                    let mut ocr_keys: Vec<i64> = Vec::new();
                    let mut ocr_texts: Vec<String> = Vec::new();
                    for input in batch {
                        for t in &input.ocr_texts {
                            ocr_keys.push(input.note_id);
                            ocr_texts.push(t.clone());
                        }
                    }
                    // The batch's engine work is independent — text,
                    // recognized-text, and image vectors share no ordering — so
                    // all futures are built BEFORE any await and joined.
                    let text_fut = embedder.embed(texts);
                    let ocr_fut = async {
                        if ocr_texts.is_empty() {
                            Ok(Vec::new())
                        } else {
                            embedder.embed(ocr_texts).await
                        }
                    };
                    let (vectors, ocr_vectors, image_vectors) =
                        futures::try_join!(text_fut, ocr_fut, image_fut)?;
                    let ndim = vectors.first().map(Vec::len).unwrap_or(0);
                    if ndim == 0 {
                        return Err(NativeError::internal("embedder returned empty vectors"));
                    }

                    self.engine.ensure(TEXT, ndim)?;
                    // Replace semantics: drop a re-added note's existing vectors
                    // first (skipped during rebuild — the index was just cleared).
                    if replace {
                        let _ = self.engine.remove(&keys)?;
                    }
                    self.engine.add(TEXT, &keys, &vectors)?;
                    if !ocr_keys.is_empty() {
                        self.engine.add(TEXT, &ocr_keys, &ocr_vectors)?;
                    }
                    if !image_keys.is_empty() {
                        let image_ndim = image_vectors.first().map(Vec::len).unwrap_or(0);
                        self.engine.ensure("image", image_ndim)?;
                        self.engine.add("image", &image_keys, &image_vectors)?;
                    }
                }
            }

            // Hash OUTSIDE the lock: the resolver's `exists` may be a harness
            // call (a stat, or Python via the injected resolver) — never run
            // foreign code under the orchestrator mutex.
            let new_hashes: Vec<(i64, String)> = batch
                .iter()
                .map(|input| (input.note_id, self.hash_for(input, images, mode)))
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
    ///
    /// # Errors
    ///
    /// Returns an error if the engine rejects the removal.
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
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
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
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
    ///
    /// # Errors
    ///
    /// Returns an error if embedding or an engine write fails mid-rebuild (the
    /// state flips to [`OrchestratorState::Error`]).
    pub async fn rebuild(
        &self,
        inputs: Vec<EmbedInput>,
        col_mod: i64,
        model_id: Option<String>,
        embedder: &dyn Embedder,
        images: Option<(&dyn ImageEmbedder, &dyn ImageResolver)>,
    ) -> NativeResult<()> {
        self.rebuild_with_mode(
            inputs,
            col_mod,
            model_id,
            embedder,
            images,
            WriteMode::TextAndImage,
        )
        .await
    }

    /// [`Self::rebuild`] with an explicit [`WriteMode`] (#232): an image-primary
    /// secondary space rebuilds in `ImageOnly` mode (only image vectors land).
    /// `TextAndImage` is the byte-identical default.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding or an engine write fails mid-rebuild (the
    /// state flips to [`OrchestratorState::Error`]).
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
    pub async fn rebuild_with_mode(
        &self,
        inputs: Vec<EmbedInput>,
        col_mod: i64,
        model_id: Option<String>,
        embedder: &dyn Embedder,
        images: Option<(&dyn ImageEmbedder, &dyn ImageResolver)>,
        mode: WriteMode,
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
            // The outer chunking exists to land build_progress between
            // batches (each ≤ BATCH_SIZE slice is a single pass inside
            // `apply`); `replace = false` skips the per-batch removes that
            // would otherwise run against the just-cleared index (#395).
            for batch in inputs.chunks(BATCH_SIZE) {
                self.apply(batch, embedder, images, false, mode).await?;
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
    ///
    /// # Errors
    ///
    /// Returns an error if embedding or an engine write fails (the state flips
    /// to [`OrchestratorState::Error`]).
    pub async fn reconcile(
        &self,
        inputs: Vec<EmbedInput>,
        col_mod: i64,
        model_id: Option<String>,
        embedder: &dyn Embedder,
        images: Option<(&dyn ImageEmbedder, &dyn ImageResolver)>,
    ) -> NativeResult<()> {
        self.reconcile_with_mode(
            inputs,
            col_mod,
            model_id,
            embedder,
            images,
            WriteMode::TextAndImage,
        )
        .await
    }

    /// [`Self::reconcile`] with an explicit [`WriteMode`] (#232): an
    /// image-primary secondary space reconciles in `ImageOnly` mode (its hash
    /// folds image refs only, so a text edit is a no-op for it). `TextAndImage`
    /// is the byte-identical default.
    ///
    /// # Errors
    ///
    /// Returns an error if a fallback rebuild, embedding, or an engine write
    /// fails (the state flips to [`OrchestratorState::Error`]).
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
    pub async fn reconcile_with_mode(
        &self,
        inputs: Vec<EmbedInput>,
        col_mod: i64,
        model_id: Option<String>,
        embedder: &dyn Embedder,
        images: Option<(&dyn ImageEmbedder, &dyn ImageResolver)>,
        mode: WriteMode,
    ) -> NativeResult<()> {
        // A MODEL SWAP must full-rebuild, never reconcile (#586). The per-note
        // hash folds text/images/OCR but NEVER the model_id, so a swap that
        // changes no note content yields an empty (or partial) diff — leaving
        // unchanged notes' vectors in the OLD model's space. The diff cannot see
        // it; only the stamped model_id can. This mirrors `check_drift`'s own
        // "different model → rebuild" rule and the documented invariant
        // ("model_id differs → full rebuild"). `rebuild_with_mode` re-embeds
        // EVERY note into the new space and stamps the new model_id, so the
        // end-state equals a full rebuild and drift goes quiet correctly.
        // (A first build — stored model_id None — falls through to the
        // empty-prior-state rebuild below; this gate only fires on a real swap.)
        let stored_model_id = self.model_id();
        if stored_model_id.is_some() && stored_model_id != model_id {
            return self
                .rebuild_with_mode(inputs, col_mod, model_id, embedder, images, mode)
                .await;
        }
        let new_hashes: BTreeMap<i64, String> = inputs
            .iter()
            .map(|i| (i.note_id, self.hash_for(i, images, mode)))
            .collect();
        let Some((to_embed, to_remove)) = self.reconcile_diff(&new_hashes) else {
            return self
                .rebuild_with_mode(inputs, col_mod, model_id, embedder, images, mode)
                .await;
        };
        if to_embed.is_empty() && to_remove.is_empty() {
            // A non-embedding edit bumped col.mod: advance the watermark only.
            let mut shared = self.shared.lock().expect("orchestrator poisoned");
            shared.col_mod = Some(col_mod);
            drop(shared);
            self.save_meta_only()?;
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
            self.add_with_mode(&changed, embedder, images, mode).await?;
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
    ///
    /// # Errors
    ///
    /// Returns an error if the engine's sampling search fails.
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
    pub fn calibrate_activation(&self) -> NativeResult<()> {
        let stats = self
            .engine
            .calibrate_activation(CALIB_SAMPLE, CALIB_K, CALIB_MIN)?;
        let mut shared = self.shared.lock().expect("orchestrator poisoned");
        shared.activation = Some(
            stats
                .into_iter()
                .map(|(modality, n, mean, std)| (modality, ActivationStat { mean, n, std }))
                .collect(),
        );
        Ok(())
    }

    /// The status block (state, size, progress, stamps) for the harness.
    ///
    /// # Panics
    ///
    /// Panics if the shared-state mutex is poisoned (a prior holder panicked).
    pub fn status(&self) -> OrchestratorStatus {
        // Per-modality breakdown (#684), text-first so it renders as the lead
        // sub-index (the engine maps modalities in a BTreeMap → name order).
        let mut modalities: Vec<ModalityStat> = self
            .engine
            .modality_stats()
            .into_iter()
            .map(|(modality, size, ndim)| ModalityStat {
                modality,
                size,
                ndim,
            })
            .collect();
        modalities.sort_by_key(|m| (m.modality != TEXT, m.modality.clone()));
        let shared = self.shared.lock().expect("orchestrator poisoned");
        OrchestratorStatus {
            state: shared.state,
            size: self.engine.size(),
            ndim: self.engine.ndim(),
            col_mod: shared.col_mod,
            model_id: shared.model_id.clone(),
            progress: BuildProgress {
                indexed: shared.build_progress.0,
                total: shared.build_progress.1,
            },
            error: shared.error.clone(),
            activation: shared.activation.clone(),
            modalities,
        }
    }
}

#[cfg(test)]
mod op_tests {
    use super::*;
    use futures::executor::block_on;
    use futures::future::BoxFuture;
    use shrike_index::MultiModalIndex;

    #[test]
    fn status_wire_shape() {
        // The harness parses this wire — pin it at the type's home so a
        // serde attribute change can't silently reshape it (keys present
        // even when null; lowercase state labels).
        let status = OrchestratorStatus {
            state: OrchestratorState::Building,
            size: 4,
            ndim: None,
            col_mod: Some(7),
            model_id: None,
            progress: BuildProgress {
                indexed: 1,
                total: 9,
            },
            error: None,
            activation: None,
            modalities: vec![
                ModalityStat {
                    modality: "text".to_owned(),
                    size: 4,
                    ndim: Some(768),
                },
                ModalityStat {
                    modality: "image".to_owned(),
                    size: 0,
                    ndim: None,
                },
            ],
        };
        assert_eq!(
            serde_json::to_string(&status).unwrap(),
            r#"{"state":"building","size":4,"ndim":null,"col_mod":7,"model_id":null,"progress":{"indexed":1,"total":9},"error":null,"activation":null,"modalities":[{"modality":"text","size":4,"ndim":768},{"modality":"image","size":0,"ndim":null}]}"#
        );
    }

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
    /// (the eager-adapter shape — `Blocking` schedules at call time) and
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

    // ── Model swap → full rebuild (#586 = audit S7-1) ───────────────────────
    //
    // The companion to `reconcile_matches_rebuild_end_state`: that test only
    // covers a SAME-model col_mod bump (the fast path). These cover the gap a
    // model swap opens — the per-note hash never folds model_id, so the diff
    // can't see a swap and unchanged notes keep OLD-model vectors unless the
    // reconcile gates to a full rebuild on a model_id mismatch.

    /// An embedder whose vectors carry a per-model `tag` component, so the same
    /// text under two "models" produces DIFFERENT vectors (a real space change).
    struct TaggedEmbedder {
        tag: f32,
    }
    impl Embedder for TaggedEmbedder {
        fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            let tag = self.tag;
            Box::pin(async move {
                Ok(texts
                    .iter()
                    .map(|t| {
                        let b = (t.len() as f32 % 7.0) / 7.0;
                        vec![b, 1.0 - b, tag, 0.0]
                    })
                    .collect())
            })
        }
    }

    /// Pure swap (col_mod UNCHANGED): a new model arrives via /embedding/start
    /// with no note edit. Pre-#586 the empty diff left model_id + every vector
    /// stale forever (drift re-fires as a no-op). Fixed: reconcile gates to a
    /// full rebuild — every vector is the new model's and model_id is stamped.
    #[test]
    fn reconcile_on_pure_model_swap_rebuilds_into_the_new_space() {
        let model_a = TaggedEmbedder { tag: 0.10 };
        let model_b = TaggedEmbedder { tag: 0.90 };
        let inputs = vec![input(1, "one"), input(2, "two"), input(3, "three")];
        let col_mod = 7; // UNCHANGED across the model swap

        let reconciled = temp_orch();
        block_on(reconciled.rebuild(
            inputs.clone(),
            col_mod,
            Some("model-A".into()),
            &model_a,
            None,
        ))
        .unwrap();
        // A model swap must register as drift.
        assert!(
            reconciled.check_drift(col_mod, Some("model-B"), false),
            "a model swap must register as drift"
        );
        block_on(reconciled.reconcile(
            inputs.clone(),
            col_mod,
            Some("model-B".into()),
            &model_b,
            None,
        ))
        .unwrap();

        let rebuilt = temp_orch();
        block_on(rebuilt.rebuild(inputs, col_mod, Some("model-B".into()), &model_b, None)).unwrap();

        // model_id stamped to the NEW model (drift now goes quiet correctly).
        assert_eq!(
            reconciled.model_id(),
            rebuilt.model_id(),
            "reconcile must stamp the new model_id after a swap (else it drifts forever)"
        );
        assert!(
            !reconciled.check_drift(col_mod, Some("model-B"), false),
            "after the swap-rebuild, the new model no longer drifts"
        );
        // Every vector equals a full rebuild's (no OLD-model vectors left).
        let mut a = reconciled.engine().keys();
        let mut b = rebuilt.engine().keys();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        for key in a {
            assert_eq!(
                reconciled.engine().get(key),
                rebuilt.engine().get(key),
                "note {key} holds an OLD-model vector after the swap"
            );
        }
    }

    /// Swap + 1 edit (col_mod BUMPS): the insidious variant. Pre-#586 the
    /// non-empty diff stamped the new model_id while the 2/3 UNCHANGED notes
    /// kept OLD-model vectors → a silent MIXED-MODEL index that reports clean.
    /// Fixed: the model_id gate forces a full rebuild regardless of the diff.
    #[test]
    fn reconcile_on_model_swap_plus_edit_is_not_a_mixed_model_index() {
        let model_a = TaggedEmbedder { tag: 0.10 };
        let model_b = TaggedEmbedder { tag: 0.90 };
        let v1 = vec![input(1, "one"), input(2, "two"), input(3, "three")];
        // Note 2 edited (its hash moves); notes 1 and 3 unchanged.
        let v2 = vec![
            input(1, "one"),
            input(2, "two EDITED LONGER"),
            input(3, "three"),
        ];

        let reconciled = temp_orch();
        block_on(reconciled.rebuild(v1, 1, Some("model-A".into()), &model_a, None)).unwrap();
        block_on(reconciled.reconcile(v2.clone(), 2, Some("model-B".into()), &model_b, None))
            .unwrap();

        let rebuilt = temp_orch();
        block_on(rebuilt.rebuild(v2, 2, Some("model-B".into()), &model_b, None)).unwrap();

        // Drift reports clean (model_id stamped) — and that's only SOUND because
        // every note, not just the edited one, was re-embedded into model-B.
        assert_eq!(reconciled.model_id(), Some("model-B".to_owned()));
        let mut a = reconciled.engine().keys();
        let mut b = rebuilt.engine().keys();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        for key in a {
            assert_eq!(
                reconciled.engine().get(key),
                rebuilt.engine().get(key),
                "unchanged note {key} kept an OLD-model vector → mixed-model index"
            );
        }
    }

    /// Boundary guard: a SAME-model col_mod bump must STILL take the incremental
    /// reconcile fast path (#585/#38), not get swept into a full rebuild by the
    /// model gate. Re-embeds only the changed note; unchanged notes' vectors are
    /// untouched and identical to the rebuild end-state.
    #[test]
    fn same_model_col_mod_bump_still_reconciles_incrementally() {
        let model = TaggedEmbedder { tag: 0.42 };
        let v1 = vec![input(1, "one"), input(2, "two"), input(3, "three")];
        let v2 = vec![input(1, "one"), input(2, "two EDITED"), input(3, "three")];

        let orch = temp_orch();
        block_on(orch.rebuild(v1, 1, Some("m".into()), &model, None)).unwrap();
        // Capture an unchanged note's vector before the reconcile.
        let before = orch.engine().get(1);
        block_on(orch.reconcile(v2.clone(), 2, Some("m".into()), &model, None)).unwrap();
        // Unchanged note 1 was NOT re-embedded (same vector object/content) — the
        // fast path held (a full rebuild would also reproduce it, but the point
        // is the gate didn't fire on a same-model bump).
        assert_eq!(orch.engine().get(1), before, "unchanged note re-embedded");

        let rebuilt = temp_orch();
        block_on(rebuilt.rebuild(v2, 2, Some("m".into()), &model, None)).unwrap();
        let mut a = orch.engine().keys();
        let mut b = rebuilt.engine().keys();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        for key in a {
            assert_eq!(orch.engine().get(key), rebuilt.engine().get(key));
        }
    }

    // ── The gate CLEARS after a swap-rebuild — incremental reconcile RESUMES
    // (the "doesn't stick / rebuild-forever" guards, from engine's #586
    // cross-review). After a swap re-stamps model_id, a subsequent SAME-model
    // edit must take the incremental fast path again, not full-rebuild forever.

    /// (1) After an A→B swap-rebuild, a same-model(B) col_mod bump editing only
    /// note 2 reconciles INCREMENTALLY: note 1 keeps its exact model-B vector
    /// (not re-embedded), the end-state equals a full rebuild, and drift is
    /// quiet. Proves the model gate fired ONCE on the swap and then released.
    #[test]
    fn incremental_reconcile_resumes_after_a_swap_rebuild() {
        let model_a = TaggedEmbedder { tag: 0.10 };
        let model_b = TaggedEmbedder { tag: 0.90 };
        let v1 = vec![input(1, "one"), input(2, "two"), input(3, "three")];

        let orch = temp_orch();
        block_on(orch.rebuild(v1.clone(), 1, Some("model-A".into()), &model_a, None)).unwrap();
        // Swap A→B (col_mod unchanged) → full rebuild into model-B's space.
        block_on(orch.reconcile(v1, 1, Some("model-B".into()), &model_b, None)).unwrap();
        // Capture note 1's model-B vector; it must survive the next reconcile.
        let note1_b = orch.engine().get(1);

        // A SAME-model(B) edit of only note 2 (col_mod bumps): incremental.
        let v2 = vec![input(1, "one"), input(2, "two EDITED"), input(3, "three")];
        block_on(orch.reconcile(v2.clone(), 2, Some("model-B".into()), &model_b, None)).unwrap();
        assert_eq!(
            orch.engine().get(1),
            note1_b,
            "note 1 was re-embedded — the gate stuck on 'rebuild forever' instead of resuming incremental"
        );
        // Drift is quiet (same model, watermark advanced) and the end-state
        // equals a full rebuild in model-B.
        assert!(
            !orch.check_drift(2, Some("model-B"), false),
            "after the same-model edit, model-B no longer drifts"
        );
        let rebuilt = temp_orch();
        block_on(rebuilt.rebuild(v2, 2, Some("model-B".into()), &model_b, None)).unwrap();
        let mut a = orch.engine().keys();
        let mut b = rebuilt.engine().keys();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        for key in a {
            assert_eq!(orch.engine().get(key), rebuilt.engine().get(key));
        }
    }

    /// (2) Image-route analogue of (1): after a CLIP A→B swap-rebuild via
    /// `reconcile_with_mode(ImageOnly)`, a same-CLIP-model image edit reconciles
    /// incrementally and matches a full ImageOnly rebuild; image drift is quiet.
    #[test]
    fn image_only_incremental_reconcile_resumes_after_a_swap_rebuild() {
        let clip_a = TaggedImageEmbedder { tag: 0.10 };
        let clip_b = TaggedImageEmbedder { tag: 0.90 };
        let v1 = vec![
            img_input(1, "t1", &["a.png"]),
            img_input(2, "t2", &["b.png"]),
            img_input(3, "t3", &["c.png"]),
        ];

        let orch = temp_orch();
        block_on(orch.rebuild_with_mode(
            v1.clone(),
            1,
            Some("clip-A".into()),
            &StubEmbedder,
            Some((&clip_a, &AlwaysResolver)),
            WriteMode::ImageOnly,
        ))
        .unwrap();
        // CLIP swap A→B (col_mod unchanged) → full ImageOnly rebuild into B.
        block_on(orch.reconcile_with_mode(
            v1,
            1,
            Some("clip-B".into()),
            &StubEmbedder,
            Some((&clip_b, &AlwaysResolver)),
            WriteMode::ImageOnly,
        ))
        .unwrap();
        let note1_b = orch.engine().modality_get("image", 1);
        // INTERMEDIATE: the swap leg must have rebuilt note 1 into clip-B (catches
        // a gate that never fires — note 1's image didn't change, so only the
        // model_id gate re-embeds it).
        let fresh_b_swap = temp_orch();
        block_on(fresh_b_swap.rebuild_with_mode(
            vec![
                img_input(1, "t1", &["a.png"]),
                img_input(2, "t2", &["b.png"]),
                img_input(3, "t3", &["c.png"]),
            ],
            1,
            Some("clip-B".into()),
            &StubEmbedder,
            Some((&clip_b, &AlwaysResolver)),
            WriteMode::ImageOnly,
        ))
        .unwrap();
        assert_eq!(
            note1_b,
            fresh_b_swap.engine().modality_get("image", 1),
            "the CLIP swap left note 1's image vector in the OLD clip-A space"
        );

        // Same-CLIP-model(B) image edit: note 2's image swapped (col_mod bumps).
        let v2 = vec![
            img_input(1, "t1", &["a.png"]),
            img_input(2, "t2", &["b2.png"]),
            img_input(3, "t3", &["c.png"]),
        ];
        block_on(orch.reconcile_with_mode(
            v2.clone(),
            2,
            Some("clip-B".into()),
            &StubEmbedder,
            Some((&clip_b, &AlwaysResolver)),
            WriteMode::ImageOnly,
        ))
        .unwrap();
        assert_eq!(
            orch.engine().modality_get("image", 1),
            note1_b,
            "image note 1 re-embedded — the image-route gate stuck instead of resuming incremental"
        );
        assert!(
            !orch.check_drift(2, Some("clip-B"), true),
            "after the same-CLIP edit, the image space no longer drifts"
        );
        let rebuilt = temp_orch();
        block_on(rebuilt.rebuild_with_mode(
            v2,
            2,
            Some("clip-B".into()),
            &StubEmbedder,
            Some((&clip_b, &AlwaysResolver)),
            WriteMode::ImageOnly,
        ))
        .unwrap();
        let mut a = orch.engine().keys();
        let mut b = rebuilt.engine().keys();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        for key in a {
            assert_eq!(
                orch.engine().modality_get("image", key),
                rebuilt.engine().modality_get("image", key),
            );
        }
    }

    /// (3) Round-trip A→B→A: swapping back to the ORIGINAL model rebuilds into
    /// A's space with NO stale B vectors — the end-state equals a fresh rebuild
    /// in A. The gate must fire on each direction of the swap, not just the
    /// first; a per-note hash (which never folds model_id) is identical across
    /// A→B→A, so only the model_id gate catches the return trip.
    #[test]
    fn round_trip_model_swap_lands_back_in_the_original_space() {
        let model_a = TaggedEmbedder { tag: 0.10 };
        let model_b = TaggedEmbedder { tag: 0.90 };
        let inputs = vec![input(1, "one"), input(2, "two"), input(3, "three")];
        let col_mod = 7; // unchanged across both swaps

        let orch = temp_orch();
        block_on(orch.rebuild(
            inputs.clone(),
            col_mod,
            Some("model-A".into()),
            &model_a,
            None,
        ))
        .unwrap();
        block_on(orch.reconcile(
            inputs.clone(),
            col_mod,
            Some("model-B".into()),
            &model_b,
            None,
        ))
        .unwrap();
        // INTERMEDIATE: the A→B leg must already have rebuilt into B (catches a
        // gate that never fires — every vector now carries model-B's tag 0.90).
        assert_eq!(orch.model_id(), Some("model-B".to_owned()));
        let fresh_b = temp_orch();
        block_on(fresh_b.rebuild(
            inputs.clone(),
            col_mod,
            Some("model-B".into()),
            &model_b,
            None,
        ))
        .unwrap();
        for key in fresh_b.engine().keys() {
            assert_eq!(
                orch.engine().get(key),
                fresh_b.engine().get(key),
                "A→B leg left note {key} in the OLD model-A space"
            );
        }
        // …and back to A (the return trip — only the model_id gate catches it,
        // since the per-note hash is identical across A→B→A).
        block_on(orch.reconcile(
            inputs.clone(),
            col_mod,
            Some("model-A".into()),
            &model_a,
            None,
        ))
        .unwrap();

        assert_eq!(orch.model_id(), Some("model-A".to_owned()));
        let fresh_a = temp_orch();
        block_on(fresh_a.rebuild(inputs, col_mod, Some("model-A".into()), &model_a, None)).unwrap();
        let mut a = orch.engine().keys();
        let mut b = fresh_a.engine().keys();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        for key in a {
            assert_eq!(
                orch.engine().get(key),
                fresh_a.engine().get(key),
                "note {key} holds a stale model-B vector after the A→B→A round-trip"
            );
        }
    }

    // ── Image-only write mode (#232) ────────────────────────────────────────

    /// An EmbedInput with image refs (the image-only path's input shape).
    fn img_input(nid: i64, text: &str, images: &[&str]) -> EmbedInput {
        EmbedInput {
            note_id: nid,
            text: text.to_owned(),
            image_names: images.iter().map(|s| s.to_string()).collect(),
            ocr_texts: vec![],
        }
    }

    #[test]
    fn image_only_hash_folds_images_not_text() {
        // The image-only hash moves iff the note's PRESENT images move — a text
        // change is invisible to it, an image add/remove/swap changes it.
        let exists_all = |_: &str| true;
        let exists_none = |_: &str| false;

        let a = note_hash_images_only(&["x.png".into()], &exists_all);
        // Same images, DIFFERENT call — stable (it never reads text).
        let a2 = note_hash_images_only(&["x.png".into()], &exists_all);
        assert_eq!(a, a2, "same images → same hash");
        // DIFFERENT image → different hash.
        let b = note_hash_images_only(&["y.png".into()], &exists_all);
        assert_ne!(a, b, "swapped image → hash moves");
        // An UNRESOLVABLE image folds like no image (presence-aware).
        let none = note_hash_images_only(&["x.png".into()], &exists_none);
        let empty = note_hash_images_only(&[], &exists_all);
        assert_eq!(none, empty, "an unresolvable image == no image");
        // It is distinct from a text-only note_hash of the same text (namespaced).
        let text_hash = note_hash("x.png", &[], false, &exists_none, &[]);
        assert_ne!(a, text_hash);
    }

    #[test]
    fn image_only_reconcile_matches_rebuild() {
        // reconcile==rebuild on the IMAGE-ONLY path (the secondary space): an
        // incremental image reconcile lands on the identical end state a full
        // image-only rebuild would. Only the `image` modality is populated (no
        // text vectors — ImageOnly skips the text embed).
        let v1 = vec![
            img_input(1, "t1", &["a.png"]),
            img_input(2, "t2", &["b.png"]),
            img_input(3, "t3", &["c.png"]),
        ];
        // v2: note 2's image swapped, note 3 gone, note 4 added. (Text changes on
        // note 1 are IGNORED by the image-only hash → not re-embedded.)
        let v2 = vec![
            img_input(1, "t1-EDITED", &["a.png"]),
            img_input(2, "t2", &["b2.png"]),
            img_input(4, "t4", &["d.png"]),
        ];

        let reconciled = temp_orch();
        block_on(reconciled.rebuild_with_mode(
            v1,
            1,
            Some("clip".into()),
            &StubEmbedder,
            Some((
                &SlowEngine {
                    name: "i",
                    events: Arc::default(),
                },
                &AlwaysResolver,
            )),
            WriteMode::ImageOnly,
        ))
        .unwrap();
        block_on(reconciled.reconcile_with_mode(
            v2.clone(),
            2,
            Some("clip".into()),
            &StubEmbedder,
            Some((
                &SlowEngine {
                    name: "i",
                    events: Arc::default(),
                },
                &AlwaysResolver,
            )),
            WriteMode::ImageOnly,
        ))
        .unwrap();

        let rebuilt = temp_orch();
        block_on(rebuilt.rebuild_with_mode(
            v2,
            2,
            Some("clip".into()),
            &StubEmbedder,
            Some((
                &SlowEngine {
                    name: "i",
                    events: Arc::default(),
                },
                &AlwaysResolver,
            )),
            WriteMode::ImageOnly,
        ))
        .unwrap();

        // Same image-modality key set + vectors; NO text-modality vectors.
        let mut a = reconciled.engine().keys();
        let mut b = rebuilt.engine().keys();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "image-only reconcile lands on the rebuild key set");
        for key in &a {
            assert!(reconciled.engine().modality_contains("image", *key));
            assert!(
                !reconciled.engine().modality_contains(TEXT, *key),
                "no text vectors"
            );
        }
    }

    /// An image embedder whose vectors carry a per-model `tag`, so a CLIP model
    /// swap produces DIFFERENT image vectors (the image-route analogue of
    /// `TaggedEmbedder`).
    struct TaggedImageEmbedder {
        tag: f32,
    }
    impl ImageEmbedder for TaggedImageEmbedder {
        fn embed_images(
            &self,
            images: Vec<MediaItem>,
        ) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            let tag = self.tag;
            Box::pin(async move { Ok(vec![vec![tag, 1.0 - tag, 0.0, 0.0]; images.len()]) })
        }
    }

    /// #586 cross-surface: the IMAGE-PRIMARY route (the separate ImageOnly
    /// secondary, on a CLIP model swap) flows through the same
    /// `reconcile_with_mode`, so the model_id gate must rebuild it too — else the
    /// secondary's image vectors stay in the OLD CLIP space. Pure swap (col_mod
    /// unchanged); pre-#586 the empty image-diff left every image vector stale.
    #[test]
    fn image_only_reconcile_on_model_swap_rebuilds_into_the_new_space() {
        let clip_a = TaggedImageEmbedder { tag: 0.10 };
        let clip_b = TaggedImageEmbedder { tag: 0.90 };
        let inputs = vec![
            img_input(1, "t1", &["a.png"]),
            img_input(2, "t2", &["b.png"]),
            img_input(3, "t3", &["c.png"]),
        ];
        let col_mod = 5; // UNCHANGED across the CLIP swap

        let reconciled = temp_orch();
        block_on(reconciled.rebuild_with_mode(
            inputs.clone(),
            col_mod,
            Some("clip-A".into()),
            &StubEmbedder,
            Some((&clip_a, &AlwaysResolver)),
            WriteMode::ImageOnly,
        ))
        .unwrap();
        block_on(reconciled.reconcile_with_mode(
            inputs.clone(),
            col_mod,
            Some("clip-B".into()),
            &StubEmbedder,
            Some((&clip_b, &AlwaysResolver)),
            WriteMode::ImageOnly,
        ))
        .unwrap();

        let rebuilt = temp_orch();
        block_on(rebuilt.rebuild_with_mode(
            inputs,
            col_mod,
            Some("clip-B".into()),
            &StubEmbedder,
            Some((&clip_b, &AlwaysResolver)),
            WriteMode::ImageOnly,
        ))
        .unwrap();

        assert_eq!(
            reconciled.model_id(),
            Some("clip-B".to_owned()),
            "image-route reconcile must stamp the new CLIP model_id"
        );
        let mut a = reconciled.engine().keys();
        let mut b = rebuilt.engine().keys();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        for key in a {
            assert_eq!(
                reconciled.engine().modality_get("image", key),
                rebuilt.engine().modality_get("image", key),
                "image key {key} kept an OLD-CLIP vector after the swap"
            );
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

        // A completed save leaves no staging files; stranded ones from a
        // crashed atomic save are ignored on load (exact-name reads only).
        assert!(!dir.join("index.meta.json.tmp").exists());
        assert!(!dir.join("index.hashes.json.tmp").exists());
        std::fs::write(dir.join("index.meta.json.tmp"), b"torn").unwrap();
        std::fs::write(dir.join("index.hashes.json.tmp"), b"torn").unwrap();
        std::fs::write(dir.join("index.usearch.tmp"), b"torn").unwrap();

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

    fn temp_dir_unique() -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "shrike-orch-owner-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }

    fn engine() -> Arc<MultiModalIndex> {
        Arc::new(MultiModalIndex::new(vec![TEXT.to_owned(), "image".to_owned()]).unwrap())
    }

    #[test]
    fn owner_is_recorded_in_meta_and_reloads_under_the_same_owner() {
        // #67: a built index stamps its owner; a reopen under the SAME owner
        // loads it (no drift), so the path-keyed namespacing costs the rightful
        // collection nothing.
        let dir = temp_dir_unique();
        let owner = Some("/coll/a.anki2".to_owned());
        let orch = IndexOrchestrator::open_owned(&dir, engine(), owner.clone());
        block_on(orch.rebuild(
            vec![input(1, "alpha")],
            7,
            Some("m".into()),
            &StubEmbedder,
            None,
        ))
        .unwrap();
        drop(orch);

        // The owner landed in the meta on disk.
        let meta: IndexMeta =
            serde_json::from_str(&std::fs::read_to_string(dir.join("index.meta.json")).unwrap())
                .unwrap();
        assert_eq!(meta.collection.as_deref(), Some("/coll/a.anki2"));

        let reopened = IndexOrchestrator::open_owned(&dir, engine(), owner);
        assert_eq!(reopened.engine().size(), 1);
        assert!(!reopened.check_drift(7, Some("m"), false)); // current — no rebuild
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mismatched_owner_is_ignored_and_forces_a_rebuild() {
        // #67: a cache dir written by a DIFFERENT collection (moved/wrong cache)
        // is not silently served — the orchestrator loads nothing, so the next
        // drift check reports "no index" and rebuilds for the current owner.
        let dir = temp_dir_unique();
        let orch =
            IndexOrchestrator::open_owned(&dir, engine(), Some("/coll/original.anki2".to_owned()));
        block_on(orch.rebuild(
            vec![input(1, "alpha")],
            7,
            Some("m".into()),
            &StubEmbedder,
            None,
        ))
        .unwrap();
        drop(orch);

        // Reopen claiming a different collection owns this dir.
        let foreign =
            IndexOrchestrator::open_owned(&dir, engine(), Some("/coll/other.anki2".to_owned()));
        assert_eq!(foreign.engine().size(), 0, "a foreign index must not load");
        assert_eq!(
            foreign.col_mod(),
            None,
            "no stamp adopted from a foreign meta"
        );
        assert!(foreign.check_drift(7, Some("m"), false), "drift → rebuild");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pre_67_index_without_owner_loads_then_gets_stamped() {
        // An index built before #67 records no owner; it must load as-is (the
        // single-collection user's index survives the upgrade) and the next
        // save stamps the owner in.
        let dir = temp_dir_unique();
        // Build WITHOUT an owner (the pre-#67 shape: meta has no `collection`).
        let pre = IndexOrchestrator::open_owned(&dir, engine(), None);
        block_on(pre.rebuild(
            vec![input(1, "alpha")],
            7,
            Some("m".into()),
            &StubEmbedder,
            None,
        ))
        .unwrap();
        drop(pre);
        let meta: IndexMeta =
            serde_json::from_str(&std::fs::read_to_string(dir.join("index.meta.json")).unwrap())
                .unwrap();
        assert_eq!(meta.collection, None, "pre-#67 meta carries no owner");

        // Reopen as a real (#67) collection: the ownerless index is adopted...
        let adopted =
            IndexOrchestrator::open_owned(&dir, engine(), Some("/coll/a.anki2".to_owned()));
        assert_eq!(adopted.engine().size(), 1);
        assert!(!adopted.check_drift(7, Some("m"), false));
        // ...and the owner is persisted on the next save.
        adopted.save().unwrap();
        let meta2: IndexMeta =
            serde_json::from_str(&std::fs::read_to_string(dir.join("index.meta.json")).unwrap())
                .unwrap();
        assert_eq!(meta2.collection.as_deref(), Some("/coll/a.anki2"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
