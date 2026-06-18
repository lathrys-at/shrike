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
}
