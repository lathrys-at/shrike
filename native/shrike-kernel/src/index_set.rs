//! The N-space index coordinator (#232, the multi-space substrate's data
//! layer — #229). One [`IndexOrchestrator`] + [`DebouncedSaver`] per embedding
//! space, keyed by the space's CONTENT fingerprint (the same key
//! [`EmbedSpaces`](crate::EmbedSpaces) uses), so the index set stays in lockstep
//! with the embed set.
//!
//! ## The N=1 migration rule (load-bearing, the resolved #229 decision)
//!
//! The PRIMARY text space keeps using `cache_dir`/the per-collection index dir
//! **DIRECTLY** (no subdir) — so an existing single-space user's on-disk index
//! (`index.usearch` / `index.meta.json` / `index.hashes.json`) loads UNCHANGED,
//! with **zero rebuild** ("text-only never rebuilds on upgrade"). Only the 2nd+
//! spaces get a subdir (`<index_dir>/<space-key-hash>/`); a secondary space is
//! always new on this release, so it materializes fresh — no migration. The
//! asymmetry (the primary is special-cased to the base dir) is the price of
//! zero-migration, and it is exactly what keeps N=1 byte-identical.
//!
//! ## Lockstep with the embed set
//!
//! The kernel opens with NO embedder, so the primary orchestrator is created
//! eagerly at [`IndexSet::open`] over the base dir (byte-identical to the
//! pre-#232 single orchestrator) but **un-keyed**. The FIRST embedder attach
//! ([`IndexSet::bind_space`]) claims the primary slot for that embedder's key;
//! a SECOND distinct key creates a secondary orchestrator in a subdir. This
//! keeps the embed-space → index-space mapping exact without the index path
//! ever consuming more than the primary this PR (the fan-out is PR-C).
//!
//! ## What is fanned out this PR
//!
//! - **Removal by note id** fans out to EVERY space (a deleted note leaves all
//!   indexes).
//! - **The watermark advance + save request** fan out to every space (each
//!   space's `col_mod`/saver tracks its own drift).
//! - **Reindex/rebuild** loop the spaces, each against its own embedder; a
//!   model swap on one space drifts only that space (per-space `model_id`).
//!
//! The index/search path still reads [`IndexSet::primary`] — the one engine
//! the orchestrator/search consume until PR-C wires cross-space fusion.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use blake2::digest::consts::U16;
use blake2::{Blake2b, Digest};
use shrike_ffi::NativeResult;
use shrike_store_api::VectorIndex;

use crate::index_orchestrator::{DebouncedSaver, IndexOrchestrator};

/// A factory that builds a fresh engine over a set of modalities — injected so
/// the coordinator can materialize a secondary space's engine without naming a
/// concrete impl (the kernel passes a `MultiModalIndex`-building closure).
pub type EngineFactory = Arc<dyn Fn(&[String]) -> NativeResult<Arc<dyn VectorIndex>> + Send + Sync>;

/// One attached index space: its content-fingerprint key, the orchestrator over
/// its own dir/drift/hashes/persistence, the debounced saver, and the
/// modalities its engine spans.
pub struct IndexSpace {
    /// The CONTENT fingerprint that keys this space, or `None` while the
    /// primary space is still un-bound (no embedder attached yet).
    pub key: Option<String>,
    pub orchestrator: Arc<IndexOrchestrator>,
    pub saver: Arc<DebouncedSaver>,
    pub modalities: Vec<String>,
    /// Whether this space lives at the base index dir directly (the primary —
    /// the in-place-no-subdir migration rule) vs. a `<base>/<hash>/` subdir.
    pub primary: bool,
}

/// The base layout + saver tuning a secondary space needs to materialize its
/// own orchestrator in a subdir under the base index dir.
struct SetConfig {
    base_dir: PathBuf,
    owner: Option<String>,
    save_delay: f64,
    save_threshold: u64,
    engine_factory: EngineFactory,
}

/// The ordered set of index spaces (#232). The first element is the PRIMARY
/// space (base dir, no subdir); the rest are secondaries (subdirs). Insertion-
/// ordered, keyed by content fingerprint with replace semantics (re-binding the
/// same key is a no-op; a new key appends a secondary).
pub struct IndexSet {
    config: SetConfig,
    spaces: RwLock<Vec<IndexSpace>>,
}

impl IndexSet {
    /// Open the set with its PRIMARY space materialized eagerly at the base dir
    /// (byte-identical to the pre-#232 single orchestrator): the primary
    /// orchestrator loads any existing on-disk index in place, un-keyed until an
    /// embedder attaches. `primary_modalities` is what the primary engine spans
    /// (the kernel passes the note modalities + the `tag.text` space, exactly as
    /// the single-engine build did).
    pub fn open(
        base_dir: PathBuf,
        owner: Option<String>,
        primary_engine: Arc<dyn VectorIndex>,
        primary_modalities: Vec<String>,
        save_delay: f64,
        save_threshold: u64,
        engine_factory: EngineFactory,
    ) -> NativeResult<Arc<Self>> {
        let orchestrator = Arc::new(IndexOrchestrator::open_owned(
            base_dir.clone(),
            primary_engine,
            owner.clone(),
        ));
        let saver = DebouncedSaver::new(Arc::clone(&orchestrator), save_delay, save_threshold);
        let primary = IndexSpace {
            key: None,
            orchestrator,
            saver,
            modalities: primary_modalities,
            primary: true,
        };
        Ok(Arc::new(Self {
            config: SetConfig {
                base_dir,
                owner,
                save_delay,
                save_threshold,
                engine_factory,
            },
            spaces: RwLock::new(vec![primary]),
        }))
    }

    /// The PRIMARY orchestrator — the one engine the index/search paths consume
    /// this PR. With one declared embedder it is the sole space at the base dir,
    /// so every drift/reconcile/save/wire path is byte-identical to pre-#232.
    pub fn primary(&self) -> Arc<IndexOrchestrator> {
        Arc::clone(&self.spaces.read().expect("index set poisoned")[0].orchestrator)
    }

    /// The PRIMARY space's debounced saver — the index-maintenance tail's saver
    /// (the index path is primary-only this PR).
    pub fn primary_saver(&self) -> Arc<DebouncedSaver> {
        Arc::clone(&self.spaces.read().expect("index set poisoned")[0].saver)
    }

    /// The engine the `tag.text` centroids bind to (#178/#232): the PRIMARY /
    /// dedicated text space's engine. Tag centroids are a pure function of THAT
    /// space's text vectors — never fanned out across spaces with different
    /// geometries.
    pub fn tag_engine(&self) -> Arc<dyn VectorIndex> {
        self.spaces.read().expect("index set poisoned")[0]
            .orchestrator
            .engine_arc()
    }

    /// The number of attached index spaces.
    pub fn len(&self) -> usize {
        self.spaces.read().expect("index set poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        // The primary always exists (materialized at open), so the set is never
        // truly empty; this satisfies the clippy len/is_empty pairing.
        self.spaces.read().expect("index set poisoned").is_empty()
    }

    /// Bind an embedding space's key (its CONTENT fingerprint) to an index
    /// space (#232) — the lockstep entry the kernel's `attach_embedder_space`
    /// drives. The FIRST bind claims the un-keyed PRIMARY space (no new dir);
    /// a key matching an already-bound space is a no-op (a model re-attach that
    /// keeps its fingerprint); a NEW key materializes a secondary orchestrator
    /// in a `<base>/<hash>/` subdir over `modalities`. Returns the bound space's
    /// orchestrator.
    pub fn bind_space(
        &self,
        key: &str,
        modalities: &[String],
    ) -> NativeResult<Arc<IndexOrchestrator>> {
        let mut spaces = self.spaces.write().expect("index set poisoned");

        // A space already holds this key → no-op (return its orchestrator).
        if let Some(existing) = spaces.iter().find(|s| s.key.as_deref() == Some(key)) {
            return Ok(Arc::clone(&existing.orchestrator));
        }

        // The primary is still un-bound → claim it for this key in place (the
        // in-place-no-subdir rule). Its engine/modalities were fixed at open;
        // binding only records the key.
        if spaces[0].key.is_none() {
            spaces[0].key = Some(key.to_string());
            return Ok(Arc::clone(&spaces[0].orchestrator));
        }

        // A new secondary space → its own subdir + fresh engine (always new on
        // this release → materializes fresh, no migration).
        let subdir = self.config.base_dir.join(space_subdir(key));
        std::fs::create_dir_all(&subdir)
            .map_err(|e| shrike_ffi::NativeError::internal(format!("index space dir: {e}")))?;
        let engine = (self.config.engine_factory)(modalities)?;
        let orchestrator = Arc::new(IndexOrchestrator::open_owned(
            subdir,
            engine,
            self.config.owner.clone(),
        ));
        let saver = DebouncedSaver::new(
            Arc::clone(&orchestrator),
            self.config.save_delay,
            self.config.save_threshold,
        );
        spaces.push(IndexSpace {
            key: Some(key.to_string()),
            orchestrator: Arc::clone(&orchestrator),
            saver,
            modalities: modalities.to_vec(),
            primary: false,
        });
        Ok(orchestrator)
    }

    /// The orchestrator bound to `key`, if any (#232) — the kernel routes a
    /// space's index op to its own orchestrator through this.
    pub fn orchestrator_for(&self, key: &str) -> Option<Arc<IndexOrchestrator>> {
        self.spaces
            .read()
            .expect("index set poisoned")
            .iter()
            .find(|s| s.key.as_deref() == Some(key))
            .map(|s| Arc::clone(&s.orchestrator))
    }

    /// Every space's orchestrator, in declaration order (primary first) — the
    /// fan-out targets for removal and the watermark advance.
    pub fn all_orchestrators(&self) -> Vec<Arc<IndexOrchestrator>> {
        self.spaces
            .read()
            .expect("index set poisoned")
            .iter()
            .map(|s| Arc::clone(&s.orchestrator))
            .collect()
    }

    /// Every space's saver, in declaration order — the fan-out for save
    /// requests after a maintained write.
    pub fn all_savers(&self) -> Vec<Arc<DebouncedSaver>> {
        self.spaces
            .read()
            .expect("index set poisoned")
            .iter()
            .map(|s| Arc::clone(&s.saver))
            .collect()
    }

    /// Remove a set of notes from EVERY space's index (#232): a deleted note
    /// leaves all indexes. Returns the primary's removal count (the one the
    /// single-space path returned, byte-identical for N=1).
    pub fn remove_all(&self, note_ids: &[i64]) -> NativeResult<usize> {
        let orchestrators = self.all_orchestrators();
        let mut primary_removed = 0usize;
        for (i, orch) in orchestrators.iter().enumerate() {
            let removed = orch.remove(note_ids)?;
            if i == 0 {
                primary_removed = removed;
            }
        }
        Ok(primary_removed)
    }

    /// Advance every space's stored `col_mod` watermark (#232) — a maintained
    /// write touched all of them. With one space this is the single
    /// `set_col_mod`, byte-identical.
    pub fn set_col_mod_all(&self, value: i64) {
        for orch in self.all_orchestrators() {
            orch.set_col_mod(value);
        }
    }

    /// Request a debounced save on every space's saver (#232).
    pub fn request_save_all(&self) {
        for saver in self.all_savers() {
            saver.request_save();
        }
    }
}

/// The subdir name for a SECONDARY space: a short blake2b hash of the space key
/// so a long/odd fingerprint maps to a filesystem-safe directory name (the
/// primary never uses this — it lives at the base dir directly).
fn space_subdir(key: &str) -> String {
    let out = Blake2b::<U16>::digest(key.as_bytes());
    out.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index_orchestrator::EmbedInput;
    use crate::Embedder;
    use futures::future::BoxFuture;
    use shrike_index::MultiModalIndex;

    /// A deterministic stub embedder (text → a 4-d vector keyed on a byte of the
    /// text hash) — enough to drive reconcile/rebuild for the per-space property.
    struct StubEmbedder;
    impl Embedder for StubEmbedder {
        fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            Box::pin(async move {
                Ok(texts
                    .iter()
                    .map(|t| {
                        let h = crate::index_orchestrator::hash_text(t);
                        let b = u8::from_str_radix(&h[..2], 16).unwrap() as f32 / 255.0;
                        let n = (b * b + 1.0).sqrt();
                        vec![b / n, 1.0 / n, 0.0, 0.0]
                    })
                    .collect())
            })
        }
    }

    fn input(nid: i64, text: &str) -> EmbedInput {
        EmbedInput {
            note_id: nid,
            text: text.to_owned(),
            image_names: vec![],
            ocr_texts: vec![],
        }
    }

    fn factory() -> EngineFactory {
        Arc::new(|mods: &[String]| {
            let engine: Arc<dyn VectorIndex> = Arc::new(MultiModalIndex::new(mods.to_vec())?);
            Ok(engine)
        })
    }

    fn temp_dir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "shrike-indexset-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn open_set(dir: &PathBuf) -> Arc<IndexSet> {
        let engine: Arc<dyn VectorIndex> =
            Arc::new(MultiModalIndex::new(vec!["text".to_string(), "image".to_string()]).unwrap());
        IndexSet::open(
            dir.clone(),
            None,
            engine,
            vec!["text".to_string(), "image".to_string()],
            60.0,
            100,
            factory(),
        )
        .unwrap()
    }

    #[test]
    fn primary_lives_at_the_base_dir_and_starts_unbound() {
        let dir = temp_dir();
        let set = open_set(&dir);
        assert_eq!(set.len(), 1, "the primary is materialized at open");
        // The primary orchestrator's dir is the BASE dir (no subdir) — the
        // in-place migration rule.
        assert_eq!(set.primary().dir, dir, "primary uses the base dir directly");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn first_bind_claims_the_primary_in_place_no_subdir() {
        let dir = temp_dir();
        let set = open_set(&dir);
        let primary_dir = set.primary().dir.clone();
        let orch = set.bind_space("space:a", &["text".into()]).unwrap();
        assert_eq!(set.len(), 1, "first bind claims the primary, no new space");
        assert_eq!(orch.dir, primary_dir, "no subdir for the primary");
        // Re-binding the same key is a no-op.
        let again = set.bind_space("space:a", &["text".into()]).unwrap();
        assert_eq!(set.len(), 1);
        assert!(Arc::ptr_eq(&orch, &again));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn second_distinct_key_materializes_a_secondary_in_a_subdir() {
        let dir = temp_dir();
        let set = open_set(&dir);
        set.bind_space("space:a", &["text".into()]).unwrap();
        let secondary = set
            .bind_space("space:b", &["text".into(), "image".into()])
            .unwrap();
        assert_eq!(set.len(), 2);
        // The secondary lives in a SUBDIR under the base dir, distinct from it.
        assert_ne!(secondary.dir, dir);
        assert!(secondary.dir.starts_with(&dir));
        assert!(secondary.dir.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tag_engine_is_the_primary_engine() {
        let dir = temp_dir();
        let set = open_set(&dir);
        set.bind_space("space:a", &["text".into()]).unwrap();
        set.bind_space("space:b", &["text".into()]).unwrap();
        // The tag engine is the PRIMARY space's engine, never a secondary.
        let tag = set.tag_engine();
        let primary = set.primary().engine_arc();
        assert!(Arc::ptr_eq(&tag, &primary));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn orchestrator_for_resolves_each_bound_key() {
        let dir = temp_dir();
        let set = open_set(&dir);
        let a = set.bind_space("space:a", &["text".into()]).unwrap();
        let b = set.bind_space("space:b", &["text".into()]).unwrap();
        assert!(Arc::ptr_eq(&set.orchestrator_for("space:a").unwrap(), &a));
        assert!(Arc::ptr_eq(&set.orchestrator_for("space:b").unwrap(), &b));
        assert!(set.orchestrator_for("space:missing").is_none());
        assert_eq!(set.all_orchestrators().len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reconcile_equals_rebuild_on_a_secondary_space() {
        // The pinned reconcile==rebuild property, run PER SPACE (#232): a
        // SECONDARY space's orchestrator is a fully-independent IndexOrchestrator
        // with its own dir/drift/hashes, so an incremental reconcile on it lands
        // on the identical end state a full rebuild would — exactly the property
        // index_orchestrator.rs pins on the primary type, here proven on a space
        // the coordinator materialized.
        crate::runtime::block_on(async {
            let dir = temp_dir();
            let set = open_set(&dir);
            set.bind_space("space:a", &["text".into()]).unwrap(); // claim primary
            let secondary = set.bind_space("space:b", &["text".into()]).unwrap();

            let v1 = vec![input(1, "one"), input(2, "two"), input(3, "three")];
            let v2 = vec![input(1, "one"), input(2, "two EDITED"), input(4, "four")];

            // Reconcile path on the secondary.
            secondary
                .rebuild(v1, 1, Some("m".into()), &StubEmbedder, None)
                .await
                .unwrap();
            secondary
                .reconcile(v2.clone(), 2, Some("m".into()), &StubEmbedder, None)
                .await
                .unwrap();

            // A fresh full rebuild over v2, in its own dir.
            let fresh = open_set(&dir.join("fresh"));
            let rebuilt = fresh.primary();
            rebuilt
                .rebuild(v2, 2, Some("m".into()), &StubEmbedder, None)
                .await
                .unwrap();

            assert_eq!(secondary.engine().size(), rebuilt.engine().size());
            let mut a = secondary.engine().keys();
            let mut b = rebuilt.engine().keys();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "reconcile lands on the rebuild key set, per space");
            for key in a {
                assert_eq!(secondary.engine().get(key), rebuilt.engine().get(key));
            }
            std::fs::remove_dir_all(&dir).ok();
        });
    }

    #[test]
    fn remove_all_and_set_col_mod_all_fan_out_to_every_space() {
        // The fan-out the kernel's delete + watermark-advance ride (#232): each
        // touches EVERY space. set_col_mod_all advances both watermarks; a
        // remove_all on a note present in both leaves both.
        crate::runtime::block_on(async {
            let dir = temp_dir();
            let set = open_set(&dir);
            let primary = set.bind_space("space:a", &["text".into()]).unwrap();
            let secondary = set.bind_space("space:b", &["text".into()]).unwrap();

            for orch in [&primary, &secondary] {
                orch.rebuild(
                    vec![input(10, "ten"), input(11, "eleven")],
                    5,
                    Some("m".into()),
                    &StubEmbedder,
                    None,
                )
                .await
                .unwrap();
            }
            assert!(primary.engine().contains(10) && secondary.engine().contains(10));

            // Watermark fan-out: both spaces advance.
            set.set_col_mod_all(9);
            assert_eq!(primary.col_mod(), Some(9));
            assert_eq!(secondary.col_mod(), Some(9));

            // Removal fan-out: the note leaves BOTH spaces.
            set.remove_all(&[10]).unwrap();
            assert!(!primary.engine().contains(10));
            assert!(!secondary.engine().contains(10));
            // The untouched note stays in both.
            assert!(primary.engine().contains(11) && secondary.engine().contains(11));
            std::fs::remove_dir_all(&dir).ok();
        });
    }
}
