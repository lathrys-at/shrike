//! Per-collection cache layout: the path-derived identity that namespaces
//! a collection's vector index under the shared cache dir.
//!
//! When one daemon serves several collections, their indexes must not collide.
//! The load-bearing boundary:
//! **index identity keys on a stable function of the collection FILE PATH,
//! never the profile name.** Every collection has a path; not every collection
//! is registered, so the path is the only always-available identity — which
//! lets per-collection layout work independently of the registry (a selector
//! resolves name → path via the registry; the path determines the namespace
//! here).
//!
//! The namespace is a blake2b digest of the *canonicalized* collection path
//! (hex). Canonicalization folds `..`, symlinks, and a relative-vs-absolute
//! spelling of the same file to one identity, so `./c.anki2` and the absolute
//! path land in the same index. A file that doesn't exist yet (a fresh
//! collection) can't be canonicalized — fall back to a lexical absolutize so
//! the key is still stable across runs.

#![deny(missing_docs)]
#![deny(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc
)]

use std::path::{Path, PathBuf};

use blake2::digest::consts::U16;
use blake2::{Blake2b, Digest};

/// The subdirectory under the cache dir holding the per-collection index
/// namespaces (`<cache_dir>/index/<namespace>/`). Distinct from the derived
/// store's `<cache_dir>/derived/<namespace>/` subtree so the two never tangle.
pub const INDEX_SUBDIR: &str = "index";

/// The subdirectory under the cache dir holding the per-collection derived
/// stores (`<cache_dir>/derived/<namespace>/shrike.db`). A parallel subtree to
/// [`INDEX_SUBDIR`] with the same path-derived namespacing, so two collections
/// sharing one daemon's cache dir never share one `shrike.db` (which would
/// cross-contaminate substring/fuzzy/OCR search).
pub const DERIVED_SUBDIR: &str = "derived";

/// The derived store's filename within its per-collection namespace dir.
pub const DERIVED_DB_NAME: &str = "shrike.db";

/// A stable, path-derived identity for a collection's vector index. The same
/// collection file always yields the same key; two different files (even with
/// the same basename — the multi-collection footgun) yield different keys.
///
/// Bit-identical to the Python mirror (`shrike.cache_layout.index_namespace`)
/// so the host resolves the same per-collection path the kernel writes:
/// `blake2b(canonical_path_bytes, digest_size=16).hexdigest()` over the UTF-8
/// bytes of the canonicalized path string.
#[must_use]
pub fn index_namespace(collection_path: &str) -> String {
    let canonical = canonicalize_for_identity(collection_path);
    let out = Blake2b::<U16>::digest(canonical.as_bytes());
    out.iter().map(|b| format!("{b:02x}")).collect()
}

/// The per-collection index directory: `<cache_dir>/index/<namespace>/`.
#[must_use]
pub fn index_dir(cache_dir: &str, collection_path: &str) -> PathBuf {
    Path::new(cache_dir)
        .join(INDEX_SUBDIR)
        .join(index_namespace(collection_path))
}

/// The per-collection derived-store path:
/// `<cache_dir>/derived/<namespace>/shrike.db`. The same path-derived
/// `<namespace>` as [`index_dir`], under a parallel `derived/` subtree — so a
/// daemon serving several collections gives each its own `shrike.db`.
///
/// Bit-identical to the Python mirror (`shrike.cache_layout.derived_db_path`),
/// pinned by the parity test: `registrar`'s host `DerivedTextStore` calls the
/// Python one, the kernel's `DerivedEngine` opens the Rust one, and the two
/// MUST resolve to the same file.
#[must_use]
pub fn derived_db_path(cache_dir: &str, collection_path: &str) -> PathBuf {
    Path::new(cache_dir)
        .join(DERIVED_SUBDIR)
        .join(index_namespace(collection_path))
        .join(DERIVED_DB_NAME)
}

/// The owner identity recorded in `index.meta.json` and checked on load: the
/// canonicalized collection path. A moved/wrong cache (the meta's owner differs
/// from the current collection) is detected and rebuilt, never silently reused.
#[must_use]
pub fn owner_identity(collection_path: &str) -> String {
    canonicalize_for_identity(collection_path)
}

/// Where a collection's vector index lives and who owns it: the directory
/// the `IndexOrchestrator` opens over, plus the owner identity it stamps into
/// (and checks against) `index.meta.json`. Bundled so the kernel's `assemble`
/// takes one layout argument rather than a dir + an owner pair.
#[derive(Debug, Clone)]
pub struct IndexLayout {
    /// The directory holding `index.usearch` / `index.meta.json` / sidecars.
    pub dir: PathBuf,
    /// The owning-collection identity, or `None` to leave ownership unenforced
    /// (the injection seam, which has no collection path).
    pub owner: Option<String>,
}

impl IndexLayout {
    /// The per-collection layout: the namespaced dir under `cache_dir` plus the
    /// collection's owner identity. Also migrates an existing flat
    /// single-collection layout into this namespace, losslessly, so a
    /// single-collection user keeps their built index.
    #[must_use]
    pub fn for_collection(cache_dir: &str, collection_path: &str) -> Self {
        let dir = index_dir(cache_dir, collection_path);
        migrate_flat_layout(cache_dir, &dir, collection_path);
        Self {
            dir,
            owner: Some(owner_identity(collection_path)),
        }
    }

    /// The flat layout (the injection seam): the index sits directly under
    /// `cache_dir` and ownership is unenforced — there is no collection path to
    /// derive a namespace or owner from.
    #[must_use]
    pub fn flat(cache_dir: &str) -> Self {
        Self {
            dir: PathBuf::from(cache_dir),
            owner: None,
        }
    }
}

/// The index sidecar files (engine + meta + hashes). The migration moves these
/// as a set so the namespaced layout is byte-identical to the flat one —
/// preserving the reconcile-vs-rebuild invariant.
const INDEX_FILES: &[&str] = &[
    "index.usearch",
    "index.image.usearch",
    "index.meta.json",
    "index.hashes.json",
];

/// Migrate an existing single-collection FLAT index layout
/// (`<cache_dir>/index.usearch` + `index.meta.json`) into this collection's
/// namespace (`<index_dir>/…`), losslessly, so a single-collection user keeps
/// their built index (no spurious full rebuild).
///
/// Guarded conservatively — the migration runs ONLY when all hold:
/// - the namespaced dir has no index yet (never clobber a namespaced index);
/// - the flat layout has both `index.usearch` and `index.meta.json`;
/// - the flat meta records no owner (the common case) OR an owner matching this
///   collection (so a flat index some other path wrote is never adopted into
///   the wrong namespace).
///
/// Best-effort: any IO error logs a warning and leaves the flat files in place
/// — the kernel then sees no namespaced index and rebuilds, which is correct
/// (the index is a derived cache). The move is per-file rename within one cache
/// dir (same filesystem → atomic); a missing optional file (image sub-index,
/// hashes) is simply skipped.
pub fn migrate_flat_layout(cache_dir: &str, index_dir: &Path, collection_path: &str) {
    let cache = Path::new(cache_dir);
    let flat_meta = cache.join("index.meta.json");
    let flat_engine = cache.join("index.usearch");
    let namespaced_engine = index_dir.join("index.usearch");

    // Never clobber an already-namespaced index, and nothing to migrate
    // without the flat pair.
    if namespaced_engine.exists() || !flat_engine.exists() || !flat_meta.exists() {
        return;
    }

    // Adopt the flat index only when its recorded owner is absent (older) or
    // matches this collection — never steal another collection's flat index.
    if !flat_owner_matches(&flat_meta, collection_path) {
        return;
    }

    if let Err(e) = std::fs::create_dir_all(index_dir) {
        tracing::warn!(path = %index_dir.display(), error = %e, "flat index migration: mkdir failed");
        return;
    }
    for name in INDEX_FILES {
        let from = cache.join(name);
        if !from.exists() {
            continue;
        }
        let to = index_dir.join(name);
        if let Err(e) = std::fs::rename(&from, &to) {
            tracing::warn!(file = name, error = %e, "flat index migration: rename failed");
            // Partial migration is safe: the orchestrator only loads when BOTH
            // engine + meta are present in the namespaced dir; an incomplete
            // set there fails the load gate and rebuilds. Stop so we don't
            // leave a torn mix that looks complete.
            return;
        }
    }
    tracing::info!(path = %index_dir.display(), "migrated flat index layout to per-collection namespace");
}

/// Migrate an existing single-collection FLAT derived store
/// (`<cache_dir>/shrike.db`) into this collection's namespace
/// (`<cache_dir>/derived/<namespace>/shrike.db`), so a single-collection user
/// keeps their built FTS5/OCR derived data without even the (cheap, model-free)
/// rebuild a relocation would otherwise force.
///
/// Guarded conservatively, mirroring [`migrate_flat_layout`]:
/// - the namespaced db must not exist yet (never clobber a namespaced store);
/// - the flat `<cache_dir>/shrike.db` must exist.
///
/// Unlike the index there is **no owner check**: the derived store is a SQLite
/// file recording no owning collection (only a `col_mod` watermark inside). Two
/// facts make adopting it safe anyway. (1) A flat `shrike.db` can only be a
/// single-collection user's — multi-collection mode never writes one, so the
/// open that finds it has exactly one collection and no ambiguity about whose db
/// it is. (2) Even an unlucky adoption self-heals: the derived store rebuilds on
/// a `col_mod` mismatch, so a db whose watermark doesn't match this collection
/// reads as drift and is rebuilt field-source-scoped on first boot — the cache
/// is rebuildable by construction.
///
/// Best-effort: an IO error logs a warning and leaves the flat db in place; the
/// kernel then sees no namespaced db and builds a fresh one (correct — the
/// store is a rebuildable cache). The move is a SQLite-file rename (plus its
/// `-wal`/`-shm` sidecars if present) within one cache dir → same filesystem,
/// atomic per file.
pub fn migrate_flat_derived(cache_dir: &str, collection_path: &str) {
    let cache = Path::new(cache_dir);
    let flat_db = cache.join(DERIVED_DB_NAME);
    let dst_db = derived_db_path(cache_dir, collection_path);

    // Never clobber an already-namespaced store; nothing to migrate without
    // the flat one.
    if dst_db.exists() || !flat_db.exists() {
        return;
    }

    let Some(dst_dir) = dst_db.parent() else {
        return;
    };
    if let Err(e) = std::fs::create_dir_all(dst_dir) {
        tracing::warn!(path = %dst_dir.display(), error = %e, "flat derived migration: mkdir failed");
        return;
    }

    // The main db first — it's the load gate (DerivedEngine::open keys on it).
    // The WAL/SHM sidecars are moved alongside when present; a missing one is
    // harmless (SQLite recreates them on open), so a sidecar rename failure is
    // logged but does not abort — the db itself is what matters.
    if let Err(e) = std::fs::rename(&flat_db, &dst_db) {
        tracing::warn!(error = %e, "flat derived migration: rename failed");
        return;
    }
    for suffix in ["-wal", "-shm"] {
        let from = cache.join(format!("{DERIVED_DB_NAME}{suffix}"));
        if from.exists() {
            let to = dst_dir.join(format!("{DERIVED_DB_NAME}{suffix}"));
            if let Err(e) = std::fs::rename(&from, &to) {
                tracing::warn!(suffix, error = %e, "flat derived migration: sidecar rename failed");
            }
        }
    }
    tracing::info!(path = %dst_db.display(), "migrated flat derived store to per-collection namespace");
}

/// Read the flat meta's `collection` owner field and compare to this
/// collection. Absent owner → adopt; present + matching → adopt; present +
/// different → refuse. A corrupt/unreadable meta refuses (ownership can't be
/// proven, so don't steal it).
fn flat_owner_matches(flat_meta: &Path, collection_path: &str) -> bool {
    let Ok(contents) = std::fs::read_to_string(flat_meta) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) else {
        return false;
    };
    match value.get("collection").and_then(|c| c.as_str()) {
        None => true, // no owner recorded → a single-collection (unowned) index
        Some(owner) => owner == owner_identity(collection_path),
    }
}

/// Resolve the collection path to the stable string the identity hashes.
/// Prefer `canonicalize` (folds symlinks/`..`/relative spellings to one real
/// path), falling back to a lexical absolutize when the file doesn't exist yet
/// (a fresh collection) — `canonicalize` errors there, but the key must still
/// be stable run-to-run. A non-UTF-8 path can't round-trip the hash input
/// deterministically across platforms, so hash its lossy rendering (the same
/// fallback the rest of the kernel uses for such paths).
fn canonicalize_for_identity(collection_path: &str) -> String {
    let path = Path::new(collection_path);
    let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| lexical_absolute(path));
    resolved.to_string_lossy().into_owned()
}

/// Absolutize without touching the filesystem: join a relative path onto the
/// cwd, then collapse `.`/`..` lexically. Used only when `canonicalize` can't
/// (the file is absent). Mirrors Python's `os.path.abspath` (which also doesn't
/// require existence) so the absent-file key matches across the boundary.
fn lexical_absolute(path: &Path) -> PathBuf {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let mut out = PathBuf::new();
    for component in abs.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_is_stable_and_path_distinguishing() {
        // Stable: the same path hashes the same twice.
        let a = index_namespace("/tmp/shrike-ns-a/c.anki2");
        assert_eq!(a, index_namespace("/tmp/shrike-ns-a/c.anki2"));
        // Distinguishing: two different files (even same basename) differ.
        let b = index_namespace("/tmp/shrike-ns-b/c.anki2");
        assert_ne!(a, b);
        // Hex digest of a 16-byte blake2b → 32 chars.
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn relative_and_absolute_spellings_collapse_to_one_identity() {
        // A real file so canonicalize succeeds for both spellings.
        let tmp = std::env::temp_dir().join(format!("shrike-ns-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let file = tmp.join("c.anki2");
        std::fs::write(&file, b"x").unwrap();

        let abs = index_namespace(file.to_str().unwrap());
        // A spelling with a redundant `.` segment resolves to the same file.
        let dotted = tmp.join(".").join("c.anki2");
        assert_eq!(abs, index_namespace(dotted.to_str().unwrap()));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn absent_file_is_lexically_stable() {
        // A path that doesn't exist still hashes stably (fresh-collection case).
        let p = "/tmp/shrike-ns-does-not-exist-67/c.anki2";
        assert!(!Path::new(p).exists());
        assert_eq!(index_namespace(p), index_namespace(p));
        // And `..` collapses lexically: `/a/b/../c.anki2` == `/a/c.anki2`.
        assert_eq!(
            index_namespace("/tmp/shrike-ns-absent/x/../c.anki2"),
            index_namespace("/tmp/shrike-ns-absent/c.anki2"),
        );
    }

    #[test]
    fn index_dir_nests_under_the_index_subdir() {
        let dir = index_dir("/cache", "/tmp/shrike-ns-d/c.anki2");
        let ns = index_namespace("/tmp/shrike-ns-d/c.anki2");
        assert_eq!(dir, Path::new("/cache").join(INDEX_SUBDIR).join(&ns));
    }

    fn temp_cache() -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "shrike-migrate-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Lay down a fake flat index in `cache`. `meta` is the raw
    /// `index.meta.json` contents.
    fn write_flat(cache: &Path, meta: &str) {
        std::fs::write(cache.join("index.usearch"), b"engine-bytes").unwrap();
        std::fs::write(cache.join("index.meta.json"), meta).unwrap();
        std::fs::write(cache.join("index.hashes.json"), b"{}").unwrap();
    }

    #[test]
    fn migrates_a_pre_67_flat_layout_into_the_namespace() {
        let cache = temp_cache();
        let coll = "/coll/migrate.anki2";
        write_flat(
            &cache,
            r#"{"ndim": 4, "col_mod": 7, "model_id": "m", "schema": 2}"#,
        );

        let dst = index_dir(cache.to_str().unwrap(), coll);
        migrate_flat_layout(cache.to_str().unwrap(), &dst, coll);

        // The set moved into the namespace, byte-for-byte...
        assert_eq!(
            std::fs::read(dst.join("index.usearch")).unwrap(),
            b"engine-bytes"
        );
        assert!(dst.join("index.meta.json").exists());
        assert!(dst.join("index.hashes.json").exists());
        // ...and the flat originals are gone (a move, not a copy).
        assert!(!cache.join("index.usearch").exists());
        assert!(!cache.join("index.meta.json").exists());
        std::fs::remove_dir_all(&cache).ok();
    }

    #[test]
    fn migration_is_a_noop_when_a_namespaced_index_already_exists() {
        let cache = temp_cache();
        let coll = "/coll/already.anki2";
        write_flat(&cache, r#"{"ndim": 4}"#);
        let dst = index_dir(cache.to_str().unwrap(), coll);
        std::fs::create_dir_all(&dst).unwrap();
        std::fs::write(dst.join("index.usearch"), b"namespaced-bytes").unwrap();

        migrate_flat_layout(cache.to_str().unwrap(), &dst, coll);

        // The namespaced index is untouched; the flat one is left in place.
        assert_eq!(
            std::fs::read(dst.join("index.usearch")).unwrap(),
            b"namespaced-bytes"
        );
        assert!(cache.join("index.usearch").exists());
        std::fs::remove_dir_all(&cache).ok();
    }

    #[test]
    fn migration_refuses_a_flat_index_owned_by_another_collection() {
        let cache = temp_cache();
        // The flat meta records a DIFFERENT collection's owner.
        let owner = owner_identity("/coll/someone-else.anki2");
        write_flat(
            &cache,
            &format!(r#"{{"ndim": 4, "collection": "{owner}"}}"#),
        );

        let coll = "/coll/me.anki2";
        let dst = index_dir(cache.to_str().unwrap(), coll);
        migrate_flat_layout(cache.to_str().unwrap(), &dst, coll);

        // Not adopted: the flat files stay put, the namespace stays empty.
        assert!(cache.join("index.usearch").exists());
        assert!(!dst.join("index.usearch").exists());
        std::fs::remove_dir_all(&cache).ok();
    }

    #[test]
    fn migration_adopts_a_flat_index_whose_recorded_owner_matches() {
        let cache = temp_cache();
        let coll = "/coll/match.anki2";
        let owner = owner_identity(coll);
        write_flat(
            &cache,
            &format!(r#"{{"ndim": 4, "collection": "{owner}"}}"#),
        );

        let dst = index_dir(cache.to_str().unwrap(), coll);
        migrate_flat_layout(cache.to_str().unwrap(), &dst, coll);

        assert!(dst.join("index.usearch").exists());
        assert!(!cache.join("index.usearch").exists());
        std::fs::remove_dir_all(&cache).ok();
    }

    #[test]
    fn two_collections_namespace_to_distinct_dirs() {
        // The isolation property: two collections sharing one cache dir resolve
        // to different index dirs, so their indexes never collide.
        let cache = "/cache";
        let a = index_dir(cache, "/coll/a.anki2");
        let b = index_dir(cache, "/coll/b.anki2");
        assert_ne!(a, b);
        assert!(a.starts_with(Path::new(cache).join(INDEX_SUBDIR)));
        assert!(b.starts_with(Path::new(cache).join(INDEX_SUBDIR)));
    }

    #[test]
    fn index_layout_for_collection_namespaces_and_owns() {
        let cache = temp_cache();
        let coll = "/coll/layout.anki2";
        let layout = IndexLayout::for_collection(cache.to_str().unwrap(), coll);
        assert_eq!(layout.dir, index_dir(cache.to_str().unwrap(), coll));
        assert_eq!(layout.owner.as_deref(), Some(owner_identity(coll).as_str()));
        std::fs::remove_dir_all(&cache).ok();
    }

    #[test]
    fn index_layout_flat_is_unowned_and_at_the_cache_root() {
        let layout = IndexLayout::flat("/cache");
        assert_eq!(layout.dir, Path::new("/cache"));
        assert_eq!(layout.owner, None);
    }

    #[test]
    fn index_layout_for_collection_migrates_a_flat_index() {
        // The migration is wired through the layout constructor, so opening a
        // collection that has a flat index adopts it into the namespace.
        let cache = temp_cache();
        let coll = "/coll/via-layout.anki2";
        write_flat(&cache, r#"{"ndim": 4}"#);
        let layout = IndexLayout::for_collection(cache.to_str().unwrap(), coll);
        assert!(layout.dir.join("index.usearch").exists());
        assert!(!cache.join("index.usearch").exists());
        std::fs::remove_dir_all(&cache).ok();
    }

    // -- derived store namespacing ------------------------------------------

    #[test]
    fn derived_db_path_nests_under_the_derived_subdir() {
        let coll = "/coll/derived-a.anki2";
        let p = derived_db_path("/cache", coll);
        let ns = index_namespace(coll);
        assert_eq!(
            p,
            Path::new("/cache")
                .join(DERIVED_SUBDIR)
                .join(&ns)
                .join(DERIVED_DB_NAME)
        );
        // It shares the index's namespace but a parallel (distinct) subtree.
        assert_ne!(p.parent().unwrap(), index_dir("/cache", coll));
        assert_eq!(
            p.parent().unwrap().file_name().unwrap().to_str().unwrap(),
            ns
        );
    }

    #[test]
    fn two_collections_derive_to_distinct_db_files() {
        // The isolation property: two collections sharing one cache
        // dir get distinct shrike.db files (no substring/fuzzy/OCR bleed).
        let a = derived_db_path("/cache", "/coll/a.anki2");
        let b = derived_db_path("/cache", "/coll/b.anki2");
        assert_ne!(a, b);
        assert!(a.starts_with(Path::new("/cache").join(DERIVED_SUBDIR)));
        assert!(b.starts_with(Path::new("/cache").join(DERIVED_SUBDIR)));
    }

    /// Lay down a fake flat derived store in `cache`, with optional WAL/SHM
    /// sidecars.
    fn write_flat_derived(cache: &Path, with_sidecars: bool) {
        std::fs::write(cache.join(DERIVED_DB_NAME), b"sqlite-bytes").unwrap();
        if with_sidecars {
            std::fs::write(cache.join("shrike.db-wal"), b"wal").unwrap();
            std::fs::write(cache.join("shrike.db-shm"), b"shm").unwrap();
        }
    }

    #[test]
    fn migrates_a_flat_derived_store_into_the_namespace() {
        let cache = temp_cache();
        let coll = "/coll/derived-migrate.anki2";
        write_flat_derived(&cache, true);

        migrate_flat_derived(cache.to_str().unwrap(), coll);

        let dst = derived_db_path(cache.to_str().unwrap(), coll);
        // The db moved, byte-for-byte, with its sidecars...
        assert_eq!(std::fs::read(&dst).unwrap(), b"sqlite-bytes");
        assert_eq!(
            std::fs::read(dst.parent().unwrap().join("shrike.db-wal")).unwrap(),
            b"wal"
        );
        assert!(dst.parent().unwrap().join("shrike.db-shm").exists());
        // ...and the flat originals are gone (a move, not a copy).
        assert!(!cache.join(DERIVED_DB_NAME).exists());
        assert!(!cache.join("shrike.db-wal").exists());
        std::fs::remove_dir_all(&cache).ok();
    }

    #[test]
    fn derived_migration_is_a_noop_when_a_namespaced_db_exists() {
        let cache = temp_cache();
        let coll = "/coll/derived-already.anki2";
        write_flat_derived(&cache, false);
        let dst = derived_db_path(cache.to_str().unwrap(), coll);
        std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
        std::fs::write(&dst, b"namespaced-bytes").unwrap();

        migrate_flat_derived(cache.to_str().unwrap(), coll);

        // The namespaced db is untouched; the flat one is left in place.
        assert_eq!(std::fs::read(&dst).unwrap(), b"namespaced-bytes");
        assert!(cache.join(DERIVED_DB_NAME).exists());
        std::fs::remove_dir_all(&cache).ok();
    }

    #[test]
    fn derived_migration_is_a_noop_without_a_flat_db() {
        let cache = temp_cache();
        let coll = "/coll/derived-none.anki2";
        // No flat shrike.db → nothing migrated, no namespaced db created.
        migrate_flat_derived(cache.to_str().unwrap(), coll);
        assert!(!derived_db_path(cache.to_str().unwrap(), coll).exists());
        std::fs::remove_dir_all(&cache).ok();
    }

    // ===================================================================
    // Adversarial layer (#743): path derivation is security-relevant. A
    // namespacing collision = two collections sharing one index/derived db
    // (search cross-contamination); a path that escapes the cache root = a
    // hostile collection path writing outside its sandbox. Each test below
    // pins one such invariant.
    // ===================================================================

    /// Inlined SplitMix64 for generative tests (no external dep). Copied from
    /// `shrike-store`'s test Rng.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// Build a pseudo-random *absolute* path string. Absolute so the result is
    /// cwd-independent: the only filesystem-touching step is `canonicalize`,
    /// which fails for these never-created paths, so derivation is a pure
    /// function of the string (lexical absolutize) and the test is hermetic.
    fn random_abs_path(rng: &mut Rng) -> String {
        let alphabet: &[u8] = b"abcdef0123456789-_.XY";
        let depth = (rng.next_u64() % 5) as usize + 1;
        let mut s = String::from("/adv");
        for _ in 0..depth {
            s.push('/');
            let seg_len = (rng.next_u64() % 8) as usize + 1;
            for _ in 0..seg_len {
                let idx = (rng.next_u64() as usize) % alphabet.len();
                s.push(alphabet[idx] as char);
            }
        }
        s.push_str("/c.anki2");
        s
    }

    // -- 2. Determinism / stability pin -------------------------------------

    /// The derivation scheme is PINNED to a known output. A refactor that
    /// changes the hash (algorithm, digest size, hex casing, canonicalization
    /// fallback) silently orphans every existing user's cache — every
    /// collection rebuilds from scratch. This test forces such a change to be
    /// deliberate. The input is an *absolute, non-existent* path so the value
    /// is independent of cwd, platform tmp dir, and filesystem state.
    #[test]
    fn namespace_derivation_is_pinned_to_a_known_value() {
        assert_eq!(
            index_namespace("/srv/anki/profiles/main/collection.anki2"),
            "cb4321a318d7d5c70619c4c53f98db3e",
        );
    }

    /// Stability across many calls (not just two): the same path always maps to
    /// the same namespace within a run. Re-derivation must be a pure function.
    #[test]
    fn namespace_is_idempotent_across_many_calls() {
        let p = "/srv/anki/profiles/main/collection.anki2";
        let first = index_namespace(p);
        for _ in 0..1000 {
            assert_eq!(index_namespace(p), first);
        }
    }

    // -- 1. Namespace injectivity (the cache-sharing hazard) ----------------

    /// Paths that are genuinely distinct files must not collide. These are the
    /// adversarial near-misses: case-only difference, trailing slash, prefix
    /// relationships, deep nesting, separator runs. A collision here means two
    /// collections share an index — the exact bug this crate exists to prevent.
    ///
    /// All inputs are absolute & non-existent, so canonicalize never runs and
    /// the namespace is a pure function of the lexically-absolutized string.
    #[test]
    fn distinct_canonical_paths_do_not_collide() {
        // Each pair (or set) must canonicalize-for-identity to *distinct*
        // strings, hence distinct namespaces. Anything that collapses to the
        // same lexical string (e.g. `/a/./b` == `/a/b`) is deliberately NOT
        // here — that is a correct same-file identification, not a collision.
        let distinct = [
            "/adv/Collection.anki2",      // case A
            "/adv/collection.anki2",      // case B (case-sensitive identity)
            "/adv/sub/collection.anki2",  // prefix-related: deeper
            "/adv/sub",                   // prefix-related: the parent as a path
            "/adv/sub2/collection.anki2", // sibling
            "/adv/a/b/c/d/e/collection.anki2",
            "/adv/a/b/c/d/f/collection.anki2",
            "/adv/x.anki2",
            "/adv/x.anki22", // suffix near-miss
            "/adv/xy.anki2", // join-ambiguity: "x"+"y" vs "xy"
            "/adv/x/y.anki2",
        ];
        let mut seen = std::collections::HashMap::new();
        for p in distinct {
            let ns = index_namespace(p);
            if let Some(prev) = seen.insert(ns.clone(), p) {
                panic!("namespace collision: {prev:?} and {p:?} both hash to {ns}");
            }
        }
    }

    /// Trailing-slash and separator-run spellings: `/a/b` vs `/a/b/` vs
    /// `/a//b`. The lexical absolutize normalizes redundant separators, so
    /// these are the SAME file and SHOULD share a namespace (a slash typo must
    /// not orphan the cache). This pins the intended folding direction.
    #[test]
    fn separator_noise_folds_to_one_identity() {
        let base = index_namespace("/adv/sepnoise/c.anki2");
        // Trailing slash on a file path is unusual but must not fork identity.
        assert_eq!(index_namespace("/adv/sepnoise//c.anki2"), base);
        assert_eq!(index_namespace("/adv/sepnoise/./c.anki2"), base);
        assert_eq!(index_namespace("/adv/sepnoise/x/../c.anki2"), base);
    }

    /// Unicode normalization forms (NFC vs NFD) of the *same* visual filename
    /// are distinct byte sequences. The derivation hashes raw bytes and does
    /// NOT Unicode-normalize, so NFC and NFD spellings produce DIFFERENT
    /// namespaces. This is the conservative/safe direction: never *merge* two
    /// byte-distinct paths (a merge would risk cache-sharing); a refactor that
    /// started normalizing would change every affected user's namespace and
    /// must be deliberate. Pin the non-normalizing behavior.
    #[test]
    fn unicode_nfc_and_nfd_are_distinct_namespaces() {
        // "é": NFC = U+00E9; NFD = "e" + U+0301 (combining acute).
        let nfc = "/adv/caf\u{00e9}/c.anki2";
        let nfd = "/adv/cafe\u{0301}/c.anki2";
        assert_ne!(nfc, nfd, "test inputs must be byte-distinct");
        assert_ne!(
            index_namespace(nfc),
            index_namespace(nfd),
            "byte-distinct paths must not be merged into one cache",
        );
    }

    /// Very long paths (well past common PATH_MAX) must derive without panic
    /// and still produce a fixed-width 32-hex namespace — a hash compresses any
    /// length to 16 bytes, so the directory name never blows up.
    #[test]
    fn very_long_path_derives_to_fixed_width_namespace() {
        let mut p = String::from("/adv");
        for i in 0..5000 {
            p.push_str(&format!("/seg{i}"));
        }
        p.push_str("/c.anki2");
        let ns = index_namespace(&p);
        assert_eq!(ns.len(), 32);
        assert!(ns.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // -- 3. Subdir structure invariants -------------------------------------

    /// The index dir, the derived dir, and the cache root are three distinct
    /// places. If the index and derived subtrees ever resolved equal, the
    /// vector index and the SQLite store would write into one directory and
    /// corrupt each other. Pin: index_dir != derived parent != root, and both
    /// nest strictly under the root.
    #[test]
    fn index_and_derived_dirs_are_distinct_and_under_root() {
        let root = Path::new("/cache");
        let coll = "/adv/structure/c.anki2";
        let idx = index_dir("/cache", coll);
        let der = derived_db_path("/cache", coll);
        let der_dir = der.parent().unwrap();

        // Both strictly under the root, neither equal to it.
        assert!(idx.starts_with(root));
        assert!(der_dir.starts_with(root));
        assert_ne!(idx, root);
        assert_ne!(der_dir, root);
        // Index and derived never collide (separate subtrees).
        assert_ne!(idx, der_dir);
        // The db file itself is under its namespace dir, not the root.
        assert!(der.starts_with(root));
        assert_ne!(der, idx);
    }

    /// Re-deriving every layout path for one collection yields identical paths.
    /// The layout is a deterministic function; a stray per-call randomness or
    /// reliance on mutable global state would break cache reuse across calls.
    #[test]
    fn layout_is_self_consistent_on_redrivation() {
        let coll = "/adv/consistent/c.anki2";
        assert_eq!(index_dir("/cache", coll), index_dir("/cache", coll));
        assert_eq!(
            derived_db_path("/cache", coll),
            derived_db_path("/cache", coll)
        );
        // The shared namespace component is identical between the two subtrees.
        let idx_ns = index_dir("/cache", coll).file_name().unwrap().to_owned();
        let der_ns = derived_db_path("/cache", coll)
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_owned();
        assert_eq!(idx_ns, der_ns);
    }

    // -- 4. Path-escape safety (the traversal hazard) -----------------------

    /// A hostile collection path cannot steer the derived cache paths out of
    /// the cache root. The defense is structural: the namespace is a blake2b
    /// HEX digest — 32 chars from `[0-9a-f]`, containing no `/`, no `.`, no NUL
    /// — so it can only ever be a single leaf directory name under the subdir.
    /// Whatever `..`, absolute prefixes, embedded separators, or NUL bytes the
    /// attacker puts in the collection path, the *output* path is always
    /// `<root>/<subdir>/<32-hex>/...`. This sweeps a battery of hostile inputs
    /// and asserts containment + a clean namespace component.
    #[test]
    fn hostile_collection_paths_cannot_escape_the_cache_root() {
        let root = Path::new("/cache");
        let hostile = [
            "../../../../etc/passwd",
            "/../../../../etc/shadow",
            "....//....//etc",
            "/a/../../../../../../tmp/evil",
            "foo/../../../../bar",
            "/cache/index/../../escape", // try to climb back out of our own tree
            "..",
            "/..",
            "/",
            "",
            "a\nb/c.anki2", // newline in a segment
            "seg with spaces/c.anki2",
            "weird:colon/c.anki2",
            "*?[]glob/c.anki2",
            "\u{0}embedded-nul", // NUL: representable in &str, lossy-rendered
            "/adv/\u{0}/c.anki2",
        ];
        for h in hostile {
            let idx = index_dir("/cache", h);
            let der = derived_db_path("/cache", h);

            // Containment: both stay strictly under the cache root.
            assert!(
                idx.starts_with(root),
                "index_dir escaped root for {h:?}: {}",
                idx.display()
            );
            assert!(
                der.starts_with(root),
                "derived path escaped root for {h:?}: {}",
                der.display()
            );

            // No `..` component ever appears in the derived output — the
            // structural proof there is no traversal in the produced path.
            for comp in idx.components() {
                assert_ne!(
                    comp,
                    std::path::Component::ParentDir,
                    "index_dir for {h:?} contains a `..` component"
                );
            }

            // The namespace leaf is always clean 32-hex, regardless of input.
            let ns = index_namespace(h);
            assert_eq!(ns.len(), 32, "namespace not 32 chars for {h:?}");
            assert!(
                ns.chars().all(|c| c.is_ascii_hexdigit()),
                "namespace for {h:?} contains a non-hex char: {ns}"
            );
            // And that leaf is exactly the dir name the layout used.
            assert_eq!(idx.file_name().unwrap().to_str().unwrap(), ns);
        }
    }

    /// The namespace digest, on its own, can never contain a path separator or
    /// `..` regardless of input — the hash output alphabet is hex. This is the
    /// load-bearing reason `index_dir`/`derived_db_path` cannot be steered out
    /// of the namespace by crafting the collection path. Fuzz it to be sure no
    /// input ever produces a non-hex namespace.
    #[test]
    fn namespace_alphabet_is_hex_only_under_fuzz() {
        let mut rng = Rng::new(0xCACE_F00D_1234_5678);
        for _ in 0..5000 {
            // Mix structured random paths with raw random byte-ish strings.
            let p = if rng.next_u64() & 1 == 0 {
                random_abs_path(&mut rng)
            } else {
                // Build a string from arbitrary (but valid UTF-8) chars,
                // including separators and dots, to stress the alphabet.
                let len = (rng.next_u64() % 40) as usize;
                let pool: &[char] = &['/', '.', '.', '\\', ':', 'a', 'Z', '9', '-', ' ', '\u{e9}'];
                let mut s = String::new();
                for _ in 0..len {
                    let i = (rng.next_u64() as usize) % pool.len();
                    s.push(pool[i]);
                }
                s
            };
            let ns = index_namespace(&p);
            assert_eq!(ns.len(), 32);
            assert!(
                ns.chars().all(|c| c.is_ascii_hexdigit()),
                "non-hex namespace {ns:?} for input {p:?}"
            );
        }
    }

    // -- 6. Generative injectivity + containment fuzz -----------------------

    /// Feed thousands of distinct random absolute paths through the full
    /// derivation. Asserts, together: (a) panic-freedom, (b) injectivity —
    /// distinct lexical identities map to distinct namespaces (no collision
    /// across the corpus), and (c) containment — every produced index/derived
    /// path nests under the cache root. This is the broad net for the
    /// cache-sharing and path-escape hazards combined.
    #[test]
    fn generative_paths_are_injective_and_contained() {
        let root = Path::new("/cache");
        let mut rng = Rng::new(0x5EED_1357_9BDF_2468);
        // Map namespace -> the canonical identity string that produced it, so a
        // collision (two different identities, one namespace) is caught while a
        // legitimate same-identity repeat (two spellings of one file) is not.
        let mut by_ns: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        for _ in 0..8000 {
            let p = random_abs_path(&mut rng);
            let ns = index_namespace(&p);
            let identity = owner_identity(&p); // the canonicalized string hashed

            // Containment.
            let idx = index_dir("/cache", &p);
            let der = derived_db_path("/cache", &p);
            assert!(idx.starts_with(root.join(INDEX_SUBDIR)));
            assert!(der.starts_with(root.join(DERIVED_SUBDIR)));

            // Injectivity: same namespace must mean same identity.
            if let Some(prev_identity) = by_ns.insert(ns.clone(), identity.clone()) {
                assert_eq!(
                    prev_identity, identity,
                    "namespace collision: distinct identities {prev_identity:?} and \
                     {identity:?} share namespace {ns}"
                );
            }
        }
    }

    // -- 5. Canonicalization / symlink edges --------------------------------

    /// A symlink to a real collection file resolves (via `canonicalize`) to the
    /// same identity as the target's real path — so opening a collection
    /// through a symlink reuses the same cache, not a second orphaned one.
    /// Beyond the existing dot-segment test: this exercises the symlink leg.
    #[test]
    fn symlink_resolves_to_the_targets_identity() {
        let tmp = std::env::temp_dir().join(format!(
            "shrike-symlink-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let target = tmp.join("real.anki2");
        std::fs::write(&target, b"x").unwrap();
        let link = tmp.join("link.anki2");

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&target, &link).unwrap();
            assert_eq!(
                index_namespace(target.to_str().unwrap()),
                index_namespace(link.to_str().unwrap()),
                "a symlink must resolve to the target's cache identity",
            );
        }
        #[cfg(not(unix))]
        {
            let _ = &link; // symlink creation is privileged on Windows; skip.
        }
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// A collection accessed once before it exists (fresh, lexical fallback) and
    /// again after it is created on disk (canonicalize succeeds) keeps a STABLE
    /// identity, AS LONG AS the pre-creation spelling was already the real path
    /// (no symlinks, already absolute, no `..`). This pins that the two
    /// derivation legs agree for the common fresh-collection lifecycle, so the
    /// index built before first save is not orphaned the moment the file lands.
    #[test]
    fn lexical_and_canonical_legs_agree_for_a_plain_absolute_path() {
        let tmp = std::env::temp_dir().join(format!(
            "shrike-fresh-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let file = tmp.join("collection.anki2");
        let spelling = file.to_str().unwrap();

        // Pre-creation: file absent → lexical_absolute leg.
        assert!(!file.exists());
        let before = index_namespace(spelling);

        // Create it → canonicalize leg now succeeds.
        std::fs::write(&file, b"x").unwrap();
        let after = index_namespace(spelling);

        assert_eq!(
            before, after,
            "a plain absolute path's identity must not shift when the file lands",
        );
        std::fs::remove_dir_all(&tmp).ok();
    }
}
