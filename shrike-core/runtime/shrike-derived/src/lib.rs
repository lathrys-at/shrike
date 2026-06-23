//! Native derived-text engine: the FTS5-trigram store under the
//! `DerivedTextStore` facade, on rusqlite's **bundled** SQLite.
//!
//! The ONE implementation of the sidecar (`idx` FTS5 trigram + `rowmap`
//! provenance + `segments` + `gated` below-gate markers + `meta` in
//! `shrike.db`). On a schema-version mismatch — or an `idx`↔`rowmap` pairing
//! inconsistency found at open — the derived data is dropped and the next
//! drift rebuilds: the derived-cache answer, no migrations. The bundled
//! SQLite always has FTS5 + the trigram tokenizer, so the facade's
//! availability probe stops being load-bearing.
//!
//! **The single WRITE connection is a correctness invariant, not just
//! thread-safety**: `idx` is an FTS5 virtual table, so nothing at the schema
//! level (no FK, no trigger) ties its rowids to `rowmap` — the pairing holds
//! because every write rides ONE connection's `last_insert_rowid()` under
//! [`DerivedEngine::lock`]. Reads run on a separate pool of connections under
//! WAL, so the fanned lexical reads (substring ∥ fuzzy) run concurrently rather
//! than taking turns on one mutexed connection. A pool connection is opened
//! `OPEN_READ_ONLY`, so SQLite ENFORCES that it only ever runs SELECTs (plus its
//! own per-connection TEMP scope staging) and never writes: it cannot touch
//! `last_insert_rowid()`, so the write↔rowmap pairing invariant is owned wholly
//! by the single write connection and is untouched by the pool.
//!
//! MATCH-expression building, trigram filtering, and the state machine stay
//! facade-side; this crate is storage + queries only. Pure Rust — no pyo3.

#![deny(missing_docs)]
#![deny(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc
)]

use std::borrow::Cow;
use std::sync::Mutex;

use rusqlite::{Connection, OptionalExtension};
use shrike_error::{ErrorKind, NativeError, NativeResult};
use unicode_normalization::{is_nfc, UnicodeNormalization};

/// NFC-normalize text, so canonically-equivalent forms (a precomposed `é` versus
/// an `e` followed by a combining accent) produce the same trigrams and match.
/// Applied to BOTH indexed text and queries — the single canonical form that makes
/// the FTS index and the query agree (and keeps the Rust-side trigram overlap
/// consistent with what FTS5 indexed).
///
/// Already-NFC text (the common case — Anki normalizes fields on write) is borrowed
/// without allocating, so the per-row index hot path pays nothing for identity work.
pub fn nfc(text: &str) -> Cow<'_, str> {
    if is_nfc(text) {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(text.nfc().collect())
    }
}

/// Mirrors `shrike.derived.SNIPPET_TOKENS` (the facade doesn't pass it — it's
/// part of the pinned engine behaviour).
const SNIPPET_TOKENS: i64 = 12;

/// Per-connection memory-map ceiling. 1 GiB covers the derived store at every
/// standard scale (≤50k notes) and stays well under the platform `mmap_size` cap
/// (~2 GiB on 64-bit), so the whole store maps and reads bypass the page cache.
const MMAP_SIZE_BYTES: i64 = 1024 * 1024 * 1024;

pub use shrike_store::MatchRow;

/// Whether this build statically links rusqlite's bundled SQLite.
/// Bundled guarantees FTS5 + trigram; a platform-linked build must rely on
/// [`fts5_trigram_available`] instead.
pub const fn sqlite_bundled() -> bool {
    cfg!(feature = "bundled")
}

/// One-time, process-global SQLite tuning. MUST run before the first connection
/// opens (i.e. before SQLite initializes), so a host calls it at startup
/// ([`shrike_native`'s module init]) and every connection-open entry point below
/// calls it too, behind a `Once`, for hosts that don't (the C ABI, a standalone
/// embed, tests).
///
/// Disables memory-statistics bookkeeping: with it on (the bundled default), every
/// `sqlite3_malloc`/`free` takes the global `SQLITE_MUTEX_STATIC_MEM` mutex to
/// update stats. FTS5 segment iteration allocates per page, so pooled concurrent
/// reads serialize on that one lock — a profile of the parallel search showed
/// nearly all its time in SQLite lock/unlock. We never read the stats, so this is
/// behaviour-transparent. Best-effort: a no-op (`SQLITE_MISUSE`) if a connection
/// already opened, recorded at DEBUG.
pub fn configure_sqlite_perf() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // SAFETY: `sqlite3_config` is the variadic C configuration API;
        // `SQLITE_CONFIG_MEMSTATUS` consumes exactly one C int (the on/off flag),
        // which is what is passed. Calling it before SQLite is initialized is the
        // documented contract.
        let rc =
            unsafe { rusqlite::ffi::sqlite3_config(rusqlite::ffi::SQLITE_CONFIG_MEMSTATUS, 0) };
        if rc != rusqlite::ffi::SQLITE_OK {
            tracing::debug!(rc, "sqlite memstatus tuning skipped (already initialized)");
        }
    });
}

/// Whether the linked SQLite has FTS5 with the trigram tokenizer.
///
/// Probed on a throwaway in-memory connection — the same check the stdlib
/// engine's probe performs. Trivially true under the bundled default; genuinely
/// load-bearing when linked against a platform SQLite.
pub fn fts5_trigram_available() -> bool {
    configure_sqlite_perf();
    let Ok(conn) = Connection::open_in_memory() else {
        return false;
    };
    conn.execute_batch("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='trigram')")
        .is_ok()
}

/// The derived-text store: the FTS5 trigram sidecar over note/recognized
/// text, backing the lexical search signals plus the recognition bookkeeping.
pub struct DerivedEngine {
    /// The single WRITE connection. Every write rides this one connection's
    /// `last_insert_rowid()` under [`DerivedEngine::lock`] — the idx↔rowmap
    /// pairing invariant (see the module docs). Writes serialize on it.
    conn: Mutex<Connection>,
    /// Pooled READ connections, opened on demand, for the fanned lexical reads.
    /// Under WAL these read concurrently with each other and with the write
    /// connection without blocking.
    read_pool: ReadPool,
    /// The per-query rare-trigram cap policy ([`FuzzyCapPolicy`]). Behind a `Mutex`
    /// for interior mutability — the engine is shared `&self` (and `frozen` in the
    /// pyo3 binding), but the fuzzy-recall eval sets the policy to A/B a floor sweep
    /// and the log-growth curve. Read once per fuzzy batch and applied per query
    /// from that query's own gram count, so a mid-flight set never splits one batch.
    /// Default: fixed-6, no behaviour change.
    fuzzy_cap: Mutex<FuzzyCapPolicy>,
    /// The document-frequency ceiling `C` below which a trigram's posting is
    /// MATERIALIZED as a base bitmap ([`MATERIALIZE_DF_CEILING`]); commoner trigrams
    /// fall to the live posting read. Behind a `Mutex` for the same reason as
    /// `fuzzy_cap`: the perf harness sweeps `C` to find the per-write-cost /
    /// query-coverage knee without recompiling. Read once per build and per fold.
    materialize_ceiling: Mutex<usize>,
}

/// A pool of read-only (`OPEN_READ_ONLY`) connections to the derived store, so the
/// fanned lexical reads (substring ∥ fuzzy) run on distinct connections rather
/// than serializing on one mutexed handle. Connections are opened on demand and
/// returned on [`ReadGuard`] drop; the pool self-bounds at the live read
/// concurrency (at most one checkout per compute thread), so it grows to the
/// compute-pool width and no further without an explicit cap.
struct ReadPool {
    /// The sidecar path — read connections open against the same file as the
    /// write connection (WAL is in the file header, so they inherit it).
    path: String,
    /// Idle connections available for checkout. A connection lives on exactly
    /// one of: this vector, or a live [`ReadGuard`].
    idle: Mutex<Vec<Connection>>,
}

impl ReadPool {
    fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            idle: Mutex::new(Vec::new()),
        }
    }

    /// Open a read connection against the sidecar.
    ///
    /// `SQLITE_OPEN_READ_ONLY` ENFORCES the read-only contract: SQLite rejects any
    /// write to the main database on a pool connection, so the WRITE invariant (the
    /// idx↔rowmap pairing rides the single write connection's `last_insert_rowid()`)
    /// holds structurally, not just by discipline. It still permits the read path's
    /// TEMP scope staging — `temp` is a separate, always-writable database — and it
    /// reads the WAL store via the `-shm` the write connection created at open. Not
    /// `OPEN_CREATE`: a read connection must never conjure a missing sidecar. (Note
    /// `PRAGMA query_only` was the alternative and is unusable: it rejects the
    /// `CREATE TEMP TABLE` the staging needs, failing as `readonly`.)
    fn open_read_conn(path: &str) -> NativeResult<Connection> {
        configure_sqlite_perf();
        let conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                | rusqlite::OpenFlags::SQLITE_OPEN_URI
                | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(db_err)?;
        // Match the write connection's wait budget: a read overlapping the
        // single writer's commit waits the lock out rather than taking an
        // instant SQLITE_BUSY (with_busy_retry is the belt past this).
        conn.busy_timeout(std::time::Duration::from_secs(5))
            .map_err(db_err)?;
        // Memory-mapped reads serve pages straight from the mapped file, bypassing
        // the page cache entirely — so FTS5 segment reads skip both the global
        // page-cache mutex (the bundled SQLite forces ONE shared cache group across
        // all connections) and the per-page buffer malloc. That is the bulk of the
        // lock traffic the pooled concurrent readers otherwise serialize on.
        conn.pragma_update(None, "mmap_size", MMAP_SIZE_BYTES)
            .map_err(db_err)?;
        // Register the `rarray` virtual table on this connection so candidate-set
        // reads (e.g. seg_meta hydration) can bind a whole id list to ONE
        // prepare-cached `IN rarray(?1)` statement.
        rusqlite::vtab::array::load_module(&conn).map_err(db_err)?;
        Ok(conn)
    }

    /// Check out a connection: reuse an idle one or open a fresh one on demand.
    ///
    /// # Errors
    ///
    /// Returns an error if a new connection cannot be opened.
    fn checkout(&self) -> NativeResult<ReadGuard<'_>> {
        let popped = self.idle.lock().expect("derived read pool poisoned").pop();
        let conn = match popped {
            Some(conn) => conn,
            None => Self::open_read_conn(&self.path)?,
        };
        Ok(ReadGuard {
            pool: self,
            conn: Some(conn),
        })
    }

    fn checkin(&self, conn: Connection) {
        self.idle
            .lock()
            .expect("derived read pool poisoned")
            .push(conn);
    }
}

/// A borrowed read connection that returns itself to its [`ReadPool`] on drop.
/// `Deref`s to the [`Connection`] so the read helpers use it as a plain handle.
struct ReadGuard<'p> {
    pool: &'p ReadPool,
    /// `Some` until drop; `take`n in `Drop` to hand the connection back.
    conn: Option<Connection>,
}

impl Drop for ReadGuard<'_> {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            self.pool.checkin(conn);
        }
    }
}

impl std::ops::Deref for ReadGuard<'_> {
    type Target = Connection;
    fn deref(&self) -> &Connection {
        self.conn.as_ref().expect("read guard connection taken")
    }
}

fn db_err(e: rusqlite::Error) -> NativeError {
    // Every SQLite failure here is a runtime-resource fault, not a bug — the
    // explicit Unavailable kind, carrying the rusqlite error as the recoverable
    // `#[source]` cause rather than flattening it into the message.
    NativeError::with_source(ErrorKind::Unavailable, "sqlite", e)
}

/// True for a transient SQLite lock contention (`SQLITE_BUSY`/`SQLITE_LOCKED`,
/// incl. the `*_SNAPSHOT` variants). Two engines share one `shrike.db` file
/// (the kernel's write connection + the Python facade's read connection),
/// so a read can momentarily lose the file lock to a concurrent write even with
/// `busy_timeout` set — that case is RETRYABLE, not a real failure.
fn is_busy(e: &rusqlite::Error) -> bool {
    matches!(
        e,
        rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error {
                code: rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked,
                ..
            },
            _,
        )
    )
}

/// True for a `SQLITE_SCHEMA` fault, which on the read path is the transient a
/// POOL read hits while a rebuild's atomic shadow-swap is committing on the write
/// connection (`DROP TABLE idx; DROP TABLE rowmap; ALTER idx_shadow RENAME TO idx;
/// ALTER rowmap_shadow RENAME TO rowmap`, all in ONE transaction —
/// [`DerivedEngine::swap_shadow_and_stamp`]).
///
/// The swap is atomic for ROW VISIBILITY — an already-open statement sees the
/// whole old index or the whole new one, and the table is never *committed-absent*
/// (so no `no such table`). But the commit bumps the schema cookie, so a FRESH
/// statement on a *separate* pool connection must re-prepare, and constructing the
/// FTS5 `idx` vtable against the changing schema can fail — surfaced as
/// `SQLITE_SCHEMA` ("vtable constructor failed: idx"). The singular read path is
/// immune: it shares the write mutex with the swap, so it never overlaps it and
/// its own connection's schema cache is the one the swap updated. Only the pooled
/// reads race it, which is why this is needed.
///
/// Matched BY CODE ([`ErrorCode::SchemaChanged`]) — robust and version-stable,
/// unlike a message substring. A genuine query error (a bad MATCH, a missing
/// column) carries the disjoint generic `SQLITE_ERROR` code, so it is NOT retried
/// — it still surfaces immediately. `SQLITE_SCHEMA` is inherently transient (the
/// schema is settling); a retry resolves against the settled new index. A
/// *persistent* `SQLITE_SCHEMA` (schema thrashing forever — it cannot happen from
/// a momentary swap) would exhaust the retries and surface as `unavailable`.
fn is_transient_swap_fault(e: &rusqlite::Error) -> bool {
    matches!(
        e,
        rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error {
                code: rusqlite::ErrorCode::SchemaChanged,
                ..
            },
            _,
        )
    )
}

/// Whether a failed read should be retried: a transient lock-contention busy
/// ([`is_busy`]) OR a transient rebuild-swap-window fault ([`is_transient_swap_fault`]).
/// Both are absorbed by [`with_busy_retry`] / the read helpers' retry arms.
fn is_retryable(e: &rusqlite::Error) -> bool {
    is_busy(e) || is_transient_swap_fault(e)
}

/// How many extra times a read retries past a transient busy before surfacing
/// it. The connection's `busy_timeout` (5s) absorbs the common lock-acquisition
/// wait; this is the belt for a busy that surfaces despite it (e.g. a snapshot
/// conflict). A surviving busy surfaces as `unavailable` — the caller (kernel
/// search) propagates it rather than silently degrading to a fallback that
/// can't serve OCR/ASR-only text.
const BUSY_RETRIES: usize = 5;

/// Run a fallible read, retrying a transient `SQLITE_BUSY`/`LOCKED` up to
/// [`BUSY_RETRIES`] times with a short backoff. Non-busy errors and success
/// pass straight through.
fn with_busy_retry<T>(mut read: impl FnMut() -> rusqlite::Result<T>) -> NativeResult<T> {
    let mut attempt = 0;
    loop {
        match read() {
            Ok(v) => return Ok(v),
            Err(e) if is_retryable(&e) && attempt < BUSY_RETRIES => {
                attempt += 1;
                std::thread::sleep(std::time::Duration::from_millis(10 * attempt as u64));
            }
            Err(e) => return Err(db_err(e)),
        }
    }
}

/// Whether a [`NativeError`] wraps a transient SQLite busy as its recoverable
/// `#[source]` (the leaf [`db_err`] attaches). Lets the write path retry the same
/// contention the read path does, without flattening the busy into the message.
fn native_busy(e: &NativeError) -> bool {
    std::error::Error::source(e)
        .and_then(|s| s.downcast_ref::<rusqlite::Error>())
        .is_some_and(is_busy)
}

/// Retry a fallible WRITE (one that opens and commits its own transaction) on a
/// transient busy, mirroring [`with_busy_retry`] for reads. The write rides the
/// single write connection under `self.lock()`, but a second ENGINE can share the
/// one `shrike.db` file (the kernel writer + the Python facade's reader), and the
/// per-write delta maintenance lengthens the transaction — widening the window
/// where a commit momentarily loses the file lock. That busy is spurious; the
/// transaction rolled back, so a clean retry from the top is safe (atomic: all or
/// nothing). A persistent busy surfaces after [`BUSY_RETRIES`].
fn with_busy_retry_write<T>(mut write: impl FnMut() -> NativeResult<T>) -> NativeResult<T> {
    let mut attempt = 0;
    loop {
        match write() {
            Err(e) if attempt < BUSY_RETRIES && native_busy(&e) => {
                attempt += 1;
                std::thread::sleep(std::time::Duration::from_millis(10 * attempt as u64));
            }
            other => return other,
        }
    }
}

/// One `trigram_delta` row as read off the wire: `(term, added blob, removed blob)`.
type DeltaBlobRow = (Trigram, Vec<u8>, Vec<u8>);

impl DerivedEngine {
    /// Open (or create) the store and ensure the schema, resetting the derived
    /// data on a schema-version mismatch (no migrations — it's a rebuildable
    /// cache). Errors are `unavailable` — the facade recovers by discarding.
    /// The current sidecar schema. v3: the incremental bitmap tier (the
    /// `trigram_delta` and `trigram_dirty` tables) alongside the `segments`
    /// recognition structure. A bump drops everything — the field index re-derives on
    /// the next drift rebuild and recognition rows re-derive via the pending sweep.
    pub const SCHEMA_VERSION: i64 = 3;

    /// Open (or create) the sidecar database at `path`, migrating to
    /// `schema_version`.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or its schema migrated.
    pub fn open(path: &str, schema_version: i64) -> NativeResult<Self> {
        configure_sqlite_perf();
        let conn = Connection::open(path).map_err(db_err)?;
        // Register the `rarray` virtual table on the write connection too (the read
        // pool registers its own): the per-write delta maintenance and the fold bind
        // the touched/materialized trigram sets through `IN rarray(?1)`.
        rusqlite::vtab::array::load_module(&conn).map_err(db_err)?;
        // WAL + synchronous=NORMAL. Reads run on a separate connection pool
        // ([`ReadPool`]); WAL is what lets those pooled reads proceed
        // concurrently with each other and with this single writer without
        // blocking (a rollback journal would serialize a read against an
        // overlapping write). The -wal/-shm sidecars are the cost of that
        // concurrency. NORMAL may lose the last transaction(s) on power loss
        // (never integrity), which a rebuildable cache absorbs: the col_mod
        // watermark lags, reads as drift, rebuilds. WAL is persistent in the
        // file header, so the pool's later read connections inherit it.
        let mode: String = conn
            .query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))
            .map_err(db_err)?;
        if !mode.eq_ignore_ascii_case("wal") {
            return Err(NativeError::unavailable(format!(
                "derived store could not enter WAL mode (got {mode:?})"
            )));
        }
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(db_err)?;
        // The write connection plus the read pool are per ENGINE, but two engines
        // can share the file (the kernel's + the Python facade's read surface).
        // With the default busy_timeout of 0, a read overlapping the OTHER engine's
        // write transaction gets an instant SQLITE_BUSY instead of waiting out a
        // brief lock. (The pool's read connections set the same timeout.)
        conn.busy_timeout(std::time::Duration::from_secs(5))
            .map_err(db_err)?;
        // Memory-mapped reads (see [`MMAP_SIZE_BYTES`] / the read pool): serves the
        // write connection's own singular reads from the map, bypassing the page
        // cache. Writes still ride the WAL — mmap is a read path.
        conn.pragma_update(None, "mmap_size", MMAP_SIZE_BYTES)
            .map_err(db_err)?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value)",
            [],
        )
        .map_err(db_err)?;
        let version: Option<i64> = conn
            .query_row(
                "SELECT value FROM meta WHERE key='schema_version'",
                [],
                |r| r.get(0),
            )
            .ok();
        if let Some(v) = version {
            if v != schema_version {
                Self::reset_tables(&conn)?;
            }
        }
        Self::create_tables(&conn)?;
        // The idx↔rowmap pairing has no schema-level enforcement (see the
        // module docs) — verify it at open and treat a mismatch exactly like
        // corruption: drop the derived data so the next drift rebuild restores
        // a consistent store (recognition rows re-derive via the pending sweep).
        // A silent mismatch would serve provenance for the wrong notes.
        let idx_rows: i64 = conn
            .query_row("SELECT count(*) FROM idx", [], |r| r.get(0))
            .map_err(db_err)?;
        let map_rows: i64 = conn
            .query_row("SELECT count(*) FROM rowmap", [], |r| r.get(0))
            .map_err(db_err)?;
        if idx_rows != map_rows {
            tracing::warn!(
                idx_rows,
                map_rows,
                "idx/rowmap desync — resetting derived data"
            );
            Self::reset_tables(&conn)?;
            Self::create_tables(&conn)?;
        }
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?1)",
            [schema_version],
        )
        .map_err(db_err)?;
        Ok(Self {
            conn: Mutex::new(conn),
            read_pool: ReadPool::new(path),
            fuzzy_cap: Mutex::new(FuzzyCapPolicy::default()),
            materialize_ceiling: Mutex::new(MATERIALIZE_DF_CEILING),
        })
    }

    /// Drop the derived tables + the col_mod watermark (schema bump or
    /// integrity failure) — the next drift detection rebuilds from scratch.
    fn reset_tables(conn: &Connection) -> NativeResult<()> {
        for sql in [
            "DROP TABLE IF EXISTS idx_vocab",
            "DROP TABLE IF EXISTS idx",
            "DROP TABLE IF EXISTS rowmap",
            "DROP TABLE IF EXISTS segments",
            "DROP TABLE IF EXISTS gated",
            "DROP TABLE IF EXISTS trigram_df",
            // The bitmap tier keys on idx rowids; dropping idx invalidates it. With
            // no global freshness gate, a stale base bitmap over a wiped idx would
            // serve rowids that no longer mean what they did — drop all three so the
            // next rebuild re-materializes from the fresh index.
            "DROP TABLE IF EXISTS trigram_bitmap",
            "DROP TABLE IF EXISTS trigram_delta",
            "DROP TABLE IF EXISTS trigram_dirty",
            "DELETE FROM meta WHERE key='col_mod'",
        ] {
            conn.execute(sql, []).map_err(db_err)?;
        }
        Ok(())
    }

    /// One DDL string for the FTS5 table — `create_tables` and the rebuild's
    /// drop-and-recreate reset must stay byte-identical.
    const IDX_DDL: &'static str =
        "CREATE VIRTUAL TABLE IF NOT EXISTS idx USING fts5(txt, tokenize='trigram')";

    /// A read-only view over `idx`'s vocabulary: `(term, doc, cnt)` per trigram,
    /// where `doc` is the document frequency. fts5vocab resolves `idx` by name at
    /// query time, so it survives the rebuild's drop-and-rename of `idx`. It
    /// computes `doc` by walking each term's doclist, so the fuzzy prune reads DF
    /// from the materialized `trigram_df` snapshot instead; this view is the SOURCE
    /// of that snapshot (see [`Self::refresh_trigram_df`]).
    const IDX_VOCAB_DDL: &'static str =
        "CREATE VIRTUAL TABLE IF NOT EXISTS idx_vocab USING fts5vocab('idx', 'row')";

    /// Materialized trigram document frequency: a plain table snapshot of
    /// `idx_vocab`'s `(term, doc)`, refreshed at the rebuild tail
    /// ([`Self::refresh_trigram_df`]). The fuzzy prune reads DF from HERE — a
    /// primary-key lookup on the mmap'd main file — rather than re-counting doclists
    /// through fts5vocab on every query. A plain table (not a view), so a `term IN
    /// (…)`/`term = ?` lookup is a real index seek, not a vocabulary scan.
    const TRIGRAM_DF_DDL: &'static str =
        "CREATE TABLE IF NOT EXISTS trigram_df(term TEXT PRIMARY KEY, df INTEGER NOT NULL)";

    /// One serialized roaring bitmap per trigram — the BASE posting over idx
    /// rowids, materialized for trigrams with `DF < C` ([`Self::materialize_ceiling`])
    /// at build by [`Self::materialize_trigram_bitmaps`] and maintained incrementally
    /// by the fold ([`Self::fold_trigram_bitmaps`]). The fuzzy candidate read loads a
    /// materialized trigram's effective posting as `(base ∪ added) \ removed` from
    /// here + [`Self::TRIGRAM_DELTA_DDL`]; an unmaterialized trigram falls to the live
    /// posting read.
    const TRIGRAM_BITMAP_DDL: &'static str =
        "CREATE TABLE IF NOT EXISTS trigram_bitmap(term TEXT PRIMARY KEY, bm BLOB NOT NULL)";

    /// Per-materialized-trigram pending changes since the last fold: two roaring
    /// bitmaps of idx rowids ADDED and REMOVED, written transactionally with the
    /// `idx` rows themselves ([`Self::apply_trigram_writes`]). A query reads a
    /// materialized trigram as `(base ∪ added) \ removed`, so the base+delta is
    /// always fresh without a global generation gate. Last-writer-wins per
    /// `(term, rowid)` — applying REMOVE then ADD ops within a write — keeps a
    /// delete-then-reuse of a freed FTS5 rowid in the correct tier. Rows exist ONLY
    /// for materialized trigrams; the fold ([`Self::fold_trigram_bitmaps`]) folds
    /// each into its base and clears it.
    const TRIGRAM_DELTA_DDL: &'static str = "CREATE TABLE IF NOT EXISTS trigram_delta(\
         term TEXT PRIMARY KEY, added BLOB NOT NULL, removed BLOB NOT NULL)";

    /// Every trigram touched by a write since the last fold (a term-only set). The
    /// fold's candidate set: bounding promote/demote (and the delta fold) to the
    /// touched trigrams keeps the fold `O(touched)`, never an `O(vocab)` rescan. A
    /// trigram's materialization can only flip if a write changed its DF, so a
    /// promote/demote candidate is necessarily here. Cleared at the end of each fold
    /// and at build.
    const TRIGRAM_DIRTY_DDL: &'static str =
        "CREATE TABLE IF NOT EXISTS trigram_dirty(term TEXT PRIMARY KEY)";

    fn create_tables(conn: &Connection) -> NativeResult<()> {
        conn.execute(Self::IDX_DDL, []).map_err(db_err)?;
        conn.execute(Self::IDX_VOCAB_DDL, []).map_err(db_err)?;
        conn.execute(Self::TRIGRAM_DF_DDL, []).map_err(db_err)?;
        conn.execute(Self::TRIGRAM_BITMAP_DDL, []).map_err(db_err)?;
        conn.execute(Self::TRIGRAM_DELTA_DDL, []).map_err(db_err)?;
        conn.execute(Self::TRIGRAM_DIRTY_DDL, []).map_err(db_err)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rowmap(\
             rowid INTEGER PRIMARY KEY, note_id INTEGER NOT NULL, \
             source TEXT NOT NULL, ref TEXT NOT NULL)",
            [],
        )
        .map_err(db_err)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS segments(\
             note_id INTEGER NOT NULL, source TEXT NOT NULL, ref TEXT NOT NULL, \
             json TEXT NOT NULL, PRIMARY KEY(note_id, source, ref))",
            [],
        )
        .map_err(db_err)?;
        // Below-gate recognition markers: a (note, source, ref) the recognizer
        // judged and the gate dropped — no text row exists, but the pending
        // sweep must count it DONE (or it re-recognizes forever). Invalidated
        // with the recognized rows on a recognizer-fingerprint change
        // ([`Self::clear_gated`]). IF NOT EXISTS + create_tables-on-open adds
        // this to an existing store with no schema bump (an absent/empty table
        // just means nothing is marked yet).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS gated(\
             note_id INTEGER NOT NULL, source TEXT NOT NULL, ref TEXT NOT NULL, \
             PRIMARY KEY(note_id, source, ref))",
            [],
        )
        .map_err(db_err)?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS rowmap_note ON rowmap(note_id, source)",
            [],
        )
        .map_err(db_err)?;
        // (source, note_id): refs_for_source/texts_for_source filter on
        // `source` alone — without this leading-column index they full-scan
        // rowmap, and texts_for_source(OCR) sits on every upsert's tail.
        // IF NOT EXISTS + create_tables-on-open adds it to an existing store.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS rowmap_source ON rowmap(source, note_id)",
            [],
        )
        .map_err(db_err)?;
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().expect("derived conn lock poisoned")
    }

    fn fuzzy_cap_lock(&self) -> std::sync::MutexGuard<'_, FuzzyCapPolicy> {
        self.fuzzy_cap.lock().expect("fuzzy cap policy poisoned")
    }

    fn materialize_ceiling_lock(&self) -> std::sync::MutexGuard<'_, usize> {
        self.materialize_ceiling
            .lock()
            .expect("materialize ceiling poisoned")
    }

    /// The stored drift watermark (`col.mod` at last reconcile), or `None`.
    pub fn get_col_mod(&self) -> Option<i64> {
        let conn = self.lock();
        conn.query_row("SELECT value FROM meta WHERE key='col_mod'", [], |r| {
            r.get(0)
        })
        .ok()
    }

    /// Stamp the derived-store drift watermark.
    ///
    /// INVARIANT: set `value` ONLY after the rows for every write up to
    /// `value`'s `col.mod` are DURABLY COMMITTED to this store. The watermark is
    /// the sole drift signal (`rebuild_derived` reconciles iff `get_col_mod() !=
    /// live col.mod`), so stamping it past an un-ingested write certifies that
    /// write as searchable when it is not — the heal gate then goes quiet and
    /// the note is permanently invisible to substring/fuzzy. The kernel enforces
    /// this via its [`crate`]-external watermark tracker: a failed/partial
    /// ingest, or a value covering a concurrent in-flight write, leaves the
    /// watermark behind for the next drift to heal — never advances it here.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn set_col_mod(&self, value: i64) -> NativeResult<()> {
        let conn = self.lock();
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('col_mod', ?1)",
            [value],
        )
        .map_err(db_err)?;
        Ok(())
    }

    /// Indexed row count. Errors are surfaced (`unavailable`), never folded
    /// to 0 — a locked/corrupt store must not read as an empty one.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn count(&self) -> NativeResult<i64> {
        let conn = self.lock();
        conn.query_row("SELECT count(*) FROM rowmap", [], |r| r.get(0))
            .map_err(db_err)
    }

    /// Above this, an id set is staged in a TEMP table instead of an inline
    /// `IN (…)` list — SQLite's parser caps long expression lists, and a
    /// collection-scale set would otherwise build a multi-megabyte statement.
    const INLINE_ID_MAX: usize = 500;

    /// Stage `ids` in a per-connection TEMP table (never touches the store
    /// file) so membership rides `IN (SELECT id FROM {table})`. Dropped and
    /// recreated each call, so a failed prior use can't leak state.
    fn stage_id_set(conn: &Connection, table: &str, ids: &[i64]) -> NativeResult<()> {
        conn.execute(&format!("DROP TABLE IF EXISTS temp.{table}"), [])
            .map_err(db_err)?;
        conn.execute(
            &format!("CREATE TEMP TABLE {table}(id INTEGER PRIMARY KEY)"),
            [],
        )
        .map_err(db_err)?;
        for chunk in ids.chunks(Self::INLINE_ID_MAX) {
            let marks = vec!["(?)"; chunk.len()].join(",");
            conn.execute(
                &format!("INSERT OR IGNORE INTO {table}(id) VALUES {marks}"),
                rusqlite::params_from_iter(chunk.iter()),
            )
            .map_err(db_err)?;
        }
        Ok(())
    }

    /// The membership clause for `note_id` against `ids`: inline placeholders
    /// below [`Self::INLINE_ID_MAX`] (returning the params to bind), a staged
    /// TEMP-table subquery above it.
    fn note_id_clause(
        conn: &Connection,
        column: &str,
        ids: &[i64],
    ) -> NativeResult<(String, Vec<i64>)> {
        if ids.len() <= Self::INLINE_ID_MAX {
            let marks = vec!["?"; ids.len()].join(",");
            Ok((format!("{column} IN ({marks})"), ids.to_vec()))
        } else {
            Self::stage_id_set(conn, "shrike_id_set", ids)?;
            Ok((
                format!("{column} IN (SELECT id FROM temp.shrike_id_set)"),
                Vec::new(),
            ))
        }
    }

    /// The `AND m.note_id IN (...)` scope filter for a deck/tag-scoped read: inline
    /// integer literals at or below [`Self::INLINE_ID_MAX`], a staged TEMP-table
    /// subquery above it.
    ///
    /// Inlining (not staging) is load-bearing on the PARALLEL read path: a temp
    /// table's btree pages bypass mmap (which maps only the main DB file), so they
    /// fault through `pcache1`'s `STATIC_LRU` mutex and serialize the parallel
    /// substring/fuzzy chunks. The ids are our own `i64`s (no injection surface), so
    /// they go straight into the SQL; the inline form costs a statement-cache entry
    /// per distinct scope (vs. one stable entry for the staged form), far cheaper
    /// than the mutex it avoids. Only a scope past the inline ceiling stages a temp
    /// set, where the parser's expression-list cap would otherwise bite.
    fn scope_clause(conn: &Connection, scope: Option<&[i64]>) -> NativeResult<String> {
        match scope {
            Some(ids) if !ids.is_empty() => {
                if ids.len() <= Self::INLINE_ID_MAX {
                    let csv = ids
                        .iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    Ok(format!("AND m.note_id IN ({csv}) "))
                } else {
                    Self::stage_id_set(conn, "shrike_scope_ids", ids)?;
                    Ok("AND m.note_id IN (SELECT id FROM temp.shrike_scope_ids) ".to_string())
                }
            }
            Some(_) => Ok("AND 0 ".to_string()), // an empty scope matches nothing
            None => Ok(String::new()),
        }
    }

    /// Drop the idx/rowmap/segments rows named by `note_ids` (and `source`),
    /// returning the `(rowid, txt)` of the idx rows it dropped — the input the
    /// delta maintenance needs to know which rowids each trigram LOST
    /// ([`Self::apply_trigram_writes`]). The text is read BEFORE the delete (it's
    /// gone after).
    fn delete_rows(
        conn: &Connection,
        note_ids: &[i64],
        source: Option<&str>,
    ) -> NativeResult<Vec<(i64, String)>> {
        if note_ids.is_empty() {
            return Ok(Vec::new());
        }
        let (id_clause, id_params) = Self::note_id_clause(conn, "note_id", note_ids)?;
        let clause = match source {
            Some(_) => format!("{id_clause} AND source=?"),
            None => id_clause,
        };
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = id_params
            .iter()
            .map(|n| Box::new(*n) as Box<dyn rusqlite::ToSql>)
            .collect();
        if let Some(s) = source {
            params.push(Box::new(s.to_string()));
        }
        // Read the idx rows this delete is about to drop, BEFORE dropping them, so the
        // delta can remove their rowids from the trigrams their text contained.
        let removed: Vec<(i64, String)> = {
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT i.rowid, i.txt FROM idx i \
                     WHERE i.rowid IN (SELECT rowid FROM rowmap WHERE {clause})"
                ))
                .map_err(db_err)?;
            let rows = stmt
                .query_map(
                    rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())),
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .map_err(db_err)?;
            rows.collect::<rusqlite::Result<_>>().map_err(db_err)?
        };
        // idx rows go first, named through rowmap — the subquery reads the
        // pairing this delete is about to drop.
        conn.execute(
            &format!("DELETE FROM idx WHERE rowid IN (SELECT rowid FROM rowmap WHERE {clause})"),
            rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())),
        )
        .map_err(db_err)?;
        conn.execute(
            &format!("DELETE FROM rowmap WHERE {clause}"),
            rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())),
        )
        .map_err(db_err)?;
        // Segments share the row keys: drop them with their rows.
        conn.execute(
            &format!("DELETE FROM segments WHERE {clause}"),
            rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())),
        )
        .map_err(db_err)?;
        Ok(removed)
    }

    /// Insert a note's text rows for one source, returning the `(rowid, NFC text)`
    /// of each row inserted — the input the delta maintenance needs to know which
    /// rowids each trigram GAINED ([`Self::apply_trigram_writes`]). The text is the
    /// NFC-normalized form actually indexed, so the caller trigrams the same bytes
    /// FTS5 did. Blank rows are skipped (no idx row, none returned).
    fn insert_rows(
        conn: &Connection,
        note_id: i64,
        source: &str,
        refs_text: &[(String, String)],
    ) -> NativeResult<Vec<(i64, String)>> {
        // prepare_cached: the two insert statements parse once per connection,
        // not once per row (a rebuild would otherwise pay ~2 prepares per field
        // row; the cache also serves every later ingest).
        let mut ins_idx = conn
            .prepare_cached("INSERT INTO idx(txt) VALUES(?1)")
            .map_err(db_err)?;
        let mut ins_map = conn
            .prepare_cached("INSERT INTO rowmap(rowid, note_id, source, ref) VALUES(?1,?2,?3,?4)")
            .map_err(db_err)?;
        let mut inserted = Vec::new();
        for (reference, text) in refs_text {
            if text.trim().is_empty() {
                continue;
            }
            // The idx→rowmap pairing rides last_insert_rowid() on THIS
            // connection — sound only under the engine's single mutexed
            // connection (the module-docs invariant; verified at open). The text
            // is NFC-normalized so the index agrees with NFC-normalized queries.
            let normalized = nfc(text).into_owned();
            ins_idx.execute([normalized.as_str()]).map_err(db_err)?;
            let rowid = conn.last_insert_rowid();
            ins_map
                .execute(rusqlite::params![rowid, note_id, source, reference])
                .map_err(db_err)?;
            inserted.push((rowid, normalized));
        }
        Ok(inserted)
    }

    /// Serialize a roaring bitmap to a blob for storage.
    fn serialize_bitmap(bm: &roaring::RoaringBitmap) -> NativeResult<Vec<u8>> {
        let mut buf = Vec::new();
        bm.serialize_into(&mut buf)
            .map_err(|e| NativeError::internal(e.to_string()))?;
        Ok(buf)
    }

    /// An `Rc`'d carray of `terms` for an `IN rarray(?1)` bind (mirrors
    /// [`Self::trigram_dfs`]).
    fn term_array<'a, I>(terms: I) -> std::rc::Rc<Vec<rusqlite::types::Value>>
    where
        I: IntoIterator<Item = &'a str>,
    {
        std::rc::Rc::new(
            terms
                .into_iter()
                .map(|t| rusqlite::types::Value::Text(t.to_string()))
                .collect(),
        )
    }

    /// Maintain the incremental bitmap tier for one write's `idx` changes, INSIDE the
    /// write transaction. `added`/`removed` are the `(rowid, NFC text)` rows
    /// [`Self::insert_rows`]/[`Self::delete_rows`] just wrote — trigrammed here the
    /// same way FTS5 tokenized them. For each MATERIALIZED trigram they touch the
    /// delta is updated so a query's `(base ∪ added) \ removed` reflects the live
    /// index with no global freshness gate; every touched trigram is recorded
    /// `trigram_dirty` for the fold's promote/demote candidate set.
    ///
    /// Per `(trigram, rowid)` it is last-writer-wins: REMOVE ops are applied before
    /// ADD ops, and `delete_rows` runs before `insert_rows` in every write path, so a
    /// freed FTS5 rowid reused by a new row in the SAME write lands in the correct
    /// tier (added iff the new text shares the trigram) regardless of which old text
    /// held it.
    ///
    /// A no-op before the first build materializes a base tier (`trigram_bitmap`
    /// empty): the materialization check finds nothing and `trigram_dirty` is left
    /// alone, so pre-build writes cost nothing here and are captured whole by the
    /// build's full materialization.
    fn apply_trigram_writes(
        conn: &Connection,
        added: &[(i64, String)],
        removed: &[(i64, String)],
    ) -> NativeResult<()> {
        use roaring::RoaringBitmap;
        if added.is_empty() && removed.is_empty() {
            return Ok(());
        }
        // No materialized tier yet (fresh store / C=0): nothing to maintain. The
        // build will materialize from the live index, which already holds this write.
        let has_base: bool = conn
            .query_row("SELECT EXISTS(SELECT 1 FROM trigram_bitmap)", [], |r| {
                r.get(0)
            })
            .map_err(db_err)?;
        if !has_base {
            return Ok(());
        }
        // Per trigram, the rowids this write gained and lost.
        let mut add_by_term: std::collections::HashMap<Trigram, Vec<u32>> =
            std::collections::HashMap::new();
        let mut rm_by_term: std::collections::HashMap<Trigram, Vec<u32>> =
            std::collections::HashMap::new();
        for (rowid, text) in removed {
            for t in trigrams(text) {
                rm_by_term.entry(t).or_default().push(*rowid as u32);
            }
        }
        for (rowid, text) in added {
            for t in trigrams(text) {
                add_by_term.entry(t).or_default().push(*rowid as u32);
            }
        }
        let touched: std::collections::BTreeSet<Trigram> = add_by_term
            .keys()
            .chain(rm_by_term.keys())
            .copied()
            .collect();
        if touched.is_empty() {
            return Ok(());
        }
        // Record every touched trigram dirty — the fold's promote/demote candidates.
        {
            let mut ins = conn
                .prepare_cached("INSERT OR IGNORE INTO trigram_dirty(term) VALUES(?1)")
                .map_err(db_err)?;
            for t in &touched {
                ins.execute(rusqlite::params![t]).map_err(db_err)?;
            }
        }
        // The materialized subset of the touched trigrams — only these carry a delta.
        let materialized: std::collections::HashSet<Trigram> = {
            let want = Self::term_array(touched.iter().map(Trigram::as_str));
            let mut stmt = conn
                .prepare_cached("SELECT term FROM trigram_bitmap WHERE term IN rarray(?1)")
                .map_err(db_err)?;
            let mut q = stmt.query(rusqlite::params![want]).map_err(db_err)?;
            let mut set = std::collections::HashSet::new();
            while let Some(r) = q.next().map_err(db_err)? {
                set.insert(r.get::<_, Trigram>(0).map_err(db_err)?);
            }
            set
        };
        if materialized.is_empty() {
            return Ok(());
        }
        // Existing deltas for the materialized touched trigrams (absent → empty).
        let mut deltas: std::collections::HashMap<Trigram, (RoaringBitmap, RoaringBitmap)> = {
            let want = Self::term_array(materialized.iter().map(Trigram::as_str));
            let mut stmt = conn
                .prepare_cached(
                    "SELECT term, added, removed FROM trigram_delta WHERE term IN rarray(?1)",
                )
                .map_err(db_err)?;
            let mut q = stmt.query(rusqlite::params![want]).map_err(db_err)?;
            let mut m = std::collections::HashMap::new();
            while let Some(r) = q.next().map_err(db_err)? {
                let term: Trigram = r.get(0).map_err(db_err)?;
                let added_blob: Vec<u8> = r.get(1).map_err(db_err)?;
                let removed_blob: Vec<u8> = r.get(2).map_err(db_err)?;
                let added_bm = RoaringBitmap::deserialize_from(&added_blob[..])
                    .map_err(|e| NativeError::internal(e.to_string()))?;
                let removed_bm = RoaringBitmap::deserialize_from(&removed_blob[..])
                    .map_err(|e| NativeError::internal(e.to_string()))?;
                m.insert(term, (added_bm, removed_bm));
            }
            m
        };
        let mut upsert = conn
            .prepare_cached(
                "INSERT OR REPLACE INTO trigram_delta(term, added, removed) VALUES(?1, ?2, ?3)",
            )
            .map_err(db_err)?;
        for term in &materialized {
            let (added_bm, removed_bm) = deltas.entry(*term).or_default();
            // REMOVE ops first, then ADD ops → last-writer-wins per (term, rowid).
            if let Some(rids) = rm_by_term.get(term) {
                for &r in rids {
                    removed_bm.insert(r);
                    added_bm.remove(r);
                }
            }
            if let Some(rids) = add_by_term.get(term) {
                for &r in rids {
                    added_bm.insert(r);
                    removed_bm.remove(r);
                }
            }
            let added_blob = Self::serialize_bitmap(added_bm)?;
            let removed_blob = Self::serialize_bitmap(removed_bm)?;
            upsert
                .execute(rusqlite::params![term, added_blob, removed_blob])
                .map_err(db_err)?;
        }
        Ok(())
    }

    /// Replace a note's text rows for one source (incremental upsert), in one
    /// transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn ingest(
        &self,
        note_id: i64,
        source: &str,
        refs_text: &[(String, String)],
    ) -> NativeResult<()> {
        let mut conn = self.lock();
        with_busy_retry_write(|| {
            let tx = conn.transaction().map_err(db_err)?;
            let removed = Self::delete_rows(&tx, &[note_id], Some(source))?;
            let added = Self::insert_rows(&tx, note_id, source, refs_text)?;
            Self::apply_trigram_writes(&tx, &added, &removed)?;
            tx.commit().map_err(db_err)
        })
    }

    /// Replace MANY notes' text rows for one source in ONE transaction: callers
    /// hold whole upsert batches, and one-commit-per-note under DELETE
    /// journaling is a journal create+fsync+delete per note. The delete half
    /// batches across all ids; inserts pair idx↔rowmap per row exactly like
    /// `ingest`.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn ingest_many(
        &self,
        notes: &[(i64, Vec<(String, String)>)],
        source: &str,
    ) -> NativeResult<()> {
        if notes.is_empty() {
            return Ok(());
        }
        // Duplicate ids: LAST entry wins. The batch deletes each id's rows ONCE
        // up front, so without this guard a duplicate would double-insert.
        let mut last: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        for (i, (id, _)) in notes.iter().enumerate() {
            last.insert(*id, i);
        }
        let mut conn = self.lock();
        let ids: Vec<i64> = notes.iter().map(|(id, _)| *id).collect();
        with_busy_retry_write(|| {
            let tx = conn.transaction().map_err(db_err)?;
            let removed = Self::delete_rows(&tx, &ids, Some(source))?;
            let mut added: Vec<(i64, String)> = Vec::new();
            for (i, (note_id, refs_text)) in notes.iter().enumerate() {
                if last.get(note_id) == Some(&i) {
                    added.extend(Self::insert_rows(&tx, *note_id, source, refs_text)?);
                }
            }
            Self::apply_trigram_writes(&tx, &added, &removed)?;
            tx.commit().map_err(db_err)
        })
    }

    /// Drop the below-gate markers for a set of notes (one source, or all).
    /// Deliberately NOT part of [`Self::delete_rows`]: `ingest` replaces a
    /// note's *text rows* and must leave its judgement markers standing (a
    /// note's newly stored image must not put its sibling gated image back
    /// in the pending set).
    fn delete_gated(conn: &Connection, note_ids: &[i64], source: Option<&str>) -> NativeResult<()> {
        if note_ids.is_empty() {
            return Ok(());
        }
        let (id_clause, id_params) = Self::note_id_clause(conn, "note_id", note_ids)?;
        let clause = match source {
            Some(_) => format!("{id_clause} AND source=?"),
            None => id_clause,
        };
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = id_params
            .iter()
            .map(|n| Box::new(*n) as Box<dyn rusqlite::ToSql>)
            .collect();
        if let Some(s) = source {
            params.push(Box::new(s.to_string()));
        }
        conn.execute(
            &format!("DELETE FROM gated WHERE {clause}"),
            rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())),
        )
        .map_err(db_err)?;
        Ok(())
    }

    /// Drop notes' rows (all sources, or just one), in one transaction.
    /// Note REMOVAL (deletion / invalidation) also drops the notes' below-gate
    /// markers — unlike `ingest`'s internal replace, which preserves them.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn remove(&self, note_ids: &[i64], source: Option<&str>) -> NativeResult<()> {
        let mut conn = self.lock();
        with_busy_retry_write(|| {
            let tx = conn.transaction().map_err(db_err)?;
            let removed = Self::delete_rows(&tx, note_ids, source)?;
            Self::delete_gated(&tx, note_ids, source)?;
            Self::apply_trigram_writes(&tx, &[], &removed)?;
            tx.commit().map_err(db_err)
        })
    }

    /// Full (re)build from (note_id, source, ref, text) rows; stamps col_mod.
    /// One transaction — a failure rolls everything back.
    ///
    /// The rebuild is **collection-derived-sources-scoped**: it
    /// replaces `field` rows (cheap — re-read from the collection) but
    /// PRESERVES recognition-derived rows (`ocr`/`asr` — expensive, with
    /// their own fingerprint-keyed invalidation), so a boot-drift rebuild
    /// never forces re-recognition. Recognition rows whose note vanished
    /// from the new row set are pruned (the note was deleted).
    ///
    /// # Errors
    ///
    /// Returns an error if a schema/table statement (the FTS5 `idx`
    /// drop+recreate or the row deletes), a row insert, the stale-row prune, or
    /// the wrapping build transaction (including its commit) fails — the whole
    /// build rolls back.
    pub fn build(
        &self,
        rows: &[(i64, String, String, String)],
        live_notes: &[i64],
        col_mod: i64,
    ) -> NativeResult<()> {
        // One-shot over the streaming core: yield the whole slice once.
        let mut taken = false;
        let mut next = || {
            if taken {
                None
            } else {
                taken = true;
                Some(Ok(rows.to_vec()))
            }
        };
        self.build_inner(&mut next, live_notes, col_mod).map(|_| ())
    }

    /// Streaming rebuild: pull `(note_id, source, ref, text)` row chunks via
    /// `next` and ingest them within ONE transaction (O(chunk) memory). Returns
    /// the total rows seen. See [`shrike_store::DerivedStore::build_streamed`].
    ///
    /// # Errors
    ///
    /// Returns an error if a chunk read or the rebuild transaction fails.
    #[allow(clippy::type_complexity)]
    pub fn build_streamed(
        &self,
        next: &mut dyn FnMut() -> Option<NativeResult<Vec<(i64, String, String, String)>>>,
        live_notes: &[i64],
        col_mod: i64,
    ) -> NativeResult<usize> {
        self.build_inner(next, live_notes, col_mod)
    }

    const IDX_SHADOW_DDL: &'static str =
        "CREATE VIRTUAL TABLE idx_shadow USING fts5(txt, tokenize='trigram')";

    #[allow(clippy::type_complexity)]
    fn build_inner(
        &self,
        next: &mut dyn FnMut() -> Option<NativeResult<Vec<(i64, String, String, String)>>>,
        live_notes: &[i64],
        col_mod: i64,
    ) -> NativeResult<usize> {
        // BUILD-AND-SWAP. The streaming producer reads each chunk through the
        // collection actor (the single drive_collection thread). The connection lock
        // MUST NOT be held across `next()`: a concurrent `search` runs
        // `search_substring`/`search_fuzzy` THROUGH that same actor and takes
        // THIS lock, so holding it across the actor-dependent pull would wedge
        // the actor on the lock while the build waits on the actor — a circular
        // deadlock.
        //
        // So the new index is built into SHADOW tables (`idx_shadow`/
        // `rowmap_shadow`) entirely OFF the live `idx`/`rowmap`, per chunk under
        // a short lock dropped around every `next()`. The live tables are
        // untouched throughout the build, so a concurrent search serves the FULL
        // OLD index the whole time — no empty/partial window, no recall cliff. A
        // single short swap transaction then renames the shadow over the live
        // tables atomically: readers see the complete old index until the swap
        // commits, the complete new one after. A mid-stream error or a crash
        // before the swap discards the shadow and leaves the live store intact at
        // the OLD col_mod (stamped only in the swap) — drift rebuilds next boot.
        self.reset_shadow()?;
        // Carry the recognition rows (`ocr`/`asr`/… — NOT rebuilt) into the
        // shadow so the swap is a clean whole-table replace, not a field-only
        // graft. Bounded by recognized-content volume, chunked under short locks.
        self.copy_recognition_rows_to_shadow()?;
        // Stream the new field rows into the shadow, off the live tables.
        let mut total = 0usize;
        while let Some(chunk) = next() {
            let chunk = chunk?;
            total += self.insert_field_chunk_to_shadow(&chunk)?;
        }
        // Merge the shadow down to one segment BEFORE the swap. The chunk-per-
        // transaction stream leaves it fragmented across many segments, and this is
        // the cheapest place to compact it: on the shadow OFF the live tables (no
        // recall window) and OUTSIDE the swap transaction (the swap stays short), so
        // the index a search hammers lands compact, not fragmented.
        self.optimize_shadow()?;
        // Atomic swap: prune dead-note rows, rename the shadow over the live
        // tables, stamp col_mod — all in ONE short transaction.
        self.swap_shadow_and_stamp(live_notes, col_mod)?;
        // Re-materialize the DF snapshot the fuzzy prune reads. Synchronous, so a
        // finished rebuild leaves it fresh; best-effort, because a refresh failure
        // costs only prune quality — a ranking drift, see the refresh_trigram_df doc
        // — never the index, so it must not fail an otherwise-successful rebuild.
        if let Err(e) = self.refresh_trigram_df() {
            tracing::warn!(
                error = %e,
                "trigram_df refresh failed; fuzzy prune will use a stale DF snapshot"
            );
        }
        // Re-materialize the base posting bitmaps (the rare, DF<C trigrams) the fuzzy
        // candidate read uses, from the freshly-swapped index, and reset the
        // incremental delta/dirty tier (the prior rowids are invalid after the swap);
        // best-effort like the DF refresh.
        if let Err(e) = self.materialize_trigram_bitmaps() {
            tracing::warn!(error = %e, "trigram_bitmap materialize failed");
        }
        Ok(total)
    }

    /// Re-materialize [`Self::TRIGRAM_DF_DDL`]'s `trigram_df` from the live index's
    /// `idx_vocab`. fts5vocab computes `doc` by walking each term's doclist, so doing
    /// it once here lets the fuzzy prune read DF as a cheap primary-key lookup
    /// instead of re-counting doclists per query. Its own short transaction, so a
    /// concurrent reader sees the prior snapshot until it commits — never an empty
    /// table. Between rebuilds the snapshot may lag the live index, but that lag is a
    /// RANKING drift, not a recall loss: the prune scans every absent (DF-0) trigram
    /// (see [`Self::prune_to_rare_terms`]), so a trigram written since the snapshot is
    /// still scanned and a match through it is still found. What the lag affects is
    /// the rarest-KNOWN selection — a stale DF can mis-order which present trigrams
    /// the prune scans. (Additions only ever make a trigram likelier to be kept, so
    /// the one residual recall risk is a DELETE drifting a present trigram's DF high
    /// enough to drop it from the rarest set; the refresh and the next rebuild both
    /// close that too.)
    ///
    /// # Errors
    ///
    /// Returns an error if the vocabulary read or the table rewrite fails.
    fn refresh_trigram_df(&self) -> NativeResult<()> {
        let mut conn = self.lock();
        let tx = conn.transaction().map_err(db_err)?;
        Self::refresh_trigram_df_in(&tx)?;
        tx.commit().map_err(db_err)?;
        Ok(())
    }

    /// The `trigram_df` rewrite, on a caller-supplied connection/transaction so the
    /// fold can refresh DF and re-tier the bitmaps in ONE transaction (no window for
    /// a write to land between the DF snapshot and the promote/demote it drives).
    fn refresh_trigram_df_in(conn: &Connection) -> NativeResult<()> {
        conn.execute("DELETE FROM trigram_df", []).map_err(db_err)?;
        conn.execute(
            "INSERT INTO trigram_df(term, df) SELECT term, doc FROM idx_vocab",
            [],
        )
        .map_err(db_err)?;
        Ok(())
    }

    /// Fully (re)materialize the BASE posting bitmaps from the live index — by
    /// trigramming each `idx` row's text with `trigrams()` — and RESET the incremental
    /// tier. Keying the base by the SAME tokenizer the delta and the query use (NOT
    /// FTS5's own fold, which diverges on e.g. Greek final sigma) is what keeps the
    /// materialized tier self-consistent without a global freshness gate.
    /// Materializes only the RARE trigrams — `DF < C` ([`Self::materialize_ceiling`])
    /// — where `DF` is the posting's cardinality, read straight off the accumulated
    /// bitmap with no separate lookup. Commoner trigrams get no base row and fall to
    /// the live posting read; the prune keeps only the rarest trigrams per query, so
    /// the query-relevant ones are materialized and the bound stays off the
    /// `O(collection)` common postings.
    ///
    /// Clears `trigram_delta`/`trigram_dirty`: the build rebuilt `idx` rowids from
    /// scratch (the shadow swap), so any prior delta keys stale rowids — the fresh
    /// base IS the live truth, with an empty delta. This is the one place a full
    /// scan of every row's text is acceptable (the heavy, infrequent rebuild); the
    /// steady-state path folds incrementally ([`Self::fold_trigram_bitmaps`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the index scan or the table rewrite fails.
    fn materialize_trigram_bitmaps(&self) -> NativeResult<()> {
        use roaring::RoaringBitmap;
        let ceiling = self.materialize_ceiling() as u64;
        let mut conn = self.lock();
        let tx = conn.transaction().map_err(db_err)?;
        tx.execute("DELETE FROM trigram_bitmap", [])
            .map_err(db_err)?;
        tx.execute("DELETE FROM trigram_delta", [])
            .map_err(db_err)?;
        tx.execute("DELETE FROM trigram_dirty", [])
            .map_err(db_err)?;
        {
            // Accumulate every trigram's posting keyed by `trigrams()` — the SAME fold
            // the delta and the query use. Building from FTS5's own `idx_vocab_inst`
            // would key the base by FTS5's context-free fold, which diverges from
            // `str::to_lowercase` on e.g. Greek final sigma / Turkish İ; the
            // `trigrams()`-keyed delta could not then maintain it, and with no global
            // freshness gate that desync surfaces as wrong results. Keying base, delta,
            // and query by one tokenizer makes the tier self-consistent regardless of
            // how `trigrams()` relates to FTS5's fold.
            let mut postings: std::collections::HashMap<Trigram, RoaringBitmap> =
                std::collections::HashMap::new();
            {
                let mut sel = tx.prepare("SELECT rowid, txt FROM idx").map_err(db_err)?;
                let mut rows = sel.query([]).map_err(db_err)?;
                while let Some(r) = rows.next().map_err(db_err)? {
                    let rowid: i64 = r.get(0).map_err(db_err)?;
                    let txt: String = r.get(1).map_err(db_err)?;
                    for t in trigrams(&txt) {
                        postings.entry(t).or_default().insert(rowid as u32);
                    }
                }
            }
            let mut ins = tx
                .prepare("INSERT INTO trigram_bitmap(term, bm) VALUES(?1, ?2)")
                .map_err(db_err)?;
            // Materialize only the rare trigrams (DF = posting cardinality < C);
            // commoner trigrams get no base row and fall to the live posting read.
            for (term, bm) in &postings {
                if bm.len() < ceiling {
                    let buf = Self::serialize_bitmap(bm)?;
                    ins.execute(rusqlite::params![term, buf]).map_err(db_err)?;
                }
            }
        }
        tx.commit().map_err(db_err)?;
        Ok(())
    }

    /// The incremental fold the debounced refresher runs between builds — the
    /// `O(touched)` replacement for the full `O(vocab)` bitmap rebuild (#998). In ONE
    /// transaction (so a concurrent write can't split the work):
    ///
    /// 1. Refresh `trigram_df` from the live index (the cheap row-vocab read the query
    ///    prune ranks on — independent of the bitmap tier).
    /// 2. For each DIRTY trigram that is MATERIALIZED, fold its pending delta into its
    ///    base (`base = (base ∪ added) \ removed`) and clear the delta — then DEMOTE it
    ///    (drop base + delta, serve live) if the folded base is now empty (every row
    ///    gone) or has grown common (`>= C`). The materialization decision rides the
    ///    folded base's OWN cardinality — the `trigrams()`-fold DF — never `trigram_df`
    ///    (FTS5's fold), so the two tokenizers can't fight over a trigram's tier.
    /// 3. Clear `trigram_dirty`.
    ///
    /// A newly-RARE unmaterialized trigram is NOT promoted here: rebuilding its
    /// `trigrams()`-fold posting would mean reading the FTS5-folded live index, whose
    /// fold can diverge from `trigrams()` (Greek final sigma, Turkish İ) — exactly the
    /// desync the single-tokenizer base avoids. It stays correct on the live posting
    /// read and is re-materialized at the next full rebuild. Demote (the delta-growth
    /// bound) is the tier move that matters for steady-state hygiene; promote is a pure
    /// read-latency optimization, deferred to the rebuild.
    ///
    /// # Errors
    ///
    /// Returns an error if the DF refresh or a table write fails.
    fn fold_trigram_bitmaps(&self) -> NativeResult<()> {
        use roaring::RoaringBitmap;
        let ceiling = self.materialize_ceiling() as u64;
        let mut conn = self.lock();
        let tx = conn.transaction().map_err(db_err)?;
        // Keep trigram_df fresh for the query prune (it ranks query trigrams by DF).
        Self::refresh_trigram_df_in(&tx)?;
        // Candidate set: only touched trigrams can have a pending delta or a tier flip.
        let dirty: Vec<Trigram> = {
            let mut stmt = tx
                .prepare("SELECT term FROM trigram_dirty")
                .map_err(db_err)?;
            let rows = stmt
                .query_map([], |r| r.get(0))
                .map_err(db_err)?
                .collect::<rusqlite::Result<_>>()
                .map_err(db_err)?;
            rows
        };
        if dirty.is_empty() {
            tx.commit().map_err(db_err)?;
            return Ok(());
        }
        for term in &dirty {
            // Only a materialized trigram carries a base + delta to reconcile.
            let base: Option<Vec<u8>> = tx
                .query_row(
                    "SELECT bm FROM trigram_bitmap WHERE term = ?1",
                    [term],
                    |r| r.get(0),
                )
                .optional()
                .map_err(db_err)?;
            let Some(base_blob) = base else {
                continue;
            };
            // Fold the pending delta (if any) into the base.
            let mut bm = RoaringBitmap::deserialize_from(&base_blob[..])
                .map_err(|e| NativeError::internal(e.to_string()))?;
            let delta: Option<(Vec<u8>, Vec<u8>)> = tx
                .query_row(
                    "SELECT added, removed FROM trigram_delta WHERE term = ?1",
                    [term],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()
                .map_err(db_err)?;
            if let Some((added_blob, removed_blob)) = delta {
                let added = RoaringBitmap::deserialize_from(&added_blob[..])
                    .map_err(|e| NativeError::internal(e.to_string()))?;
                let removed = RoaringBitmap::deserialize_from(&removed_blob[..])
                    .map_err(|e| NativeError::internal(e.to_string()))?;
                bm |= &added;
                bm -= &removed;
            }
            if bm.is_empty() || bm.len() >= ceiling {
                // DEMOTE: every row gone, or grown common — drop base + delta, serve live.
                tx.execute("DELETE FROM trigram_bitmap WHERE term = ?1", [term])
                    .map_err(db_err)?;
            } else {
                let buf = Self::serialize_bitmap(&bm)?;
                tx.execute(
                    "UPDATE trigram_bitmap SET bm = ?2 WHERE term = ?1",
                    rusqlite::params![term, buf],
                )
                .map_err(db_err)?;
            }
            tx.execute("DELETE FROM trigram_delta WHERE term = ?1", [term])
                .map_err(db_err)?;
        }
        tx.execute("DELETE FROM trigram_dirty", [])
            .map_err(db_err)?;
        tx.commit().map_err(db_err)?;
        Ok(())
    }

    /// Drop any leftover shadow tables (a prior aborted rebuild) and create
    /// fresh empty ones. Its own short transaction.
    fn reset_shadow(&self) -> NativeResult<()> {
        let conn = self.lock();
        conn.execute_batch(
            "DROP TABLE IF EXISTS idx_shadow; \
             DROP TABLE IF EXISTS rowmap_shadow;",
        )
        .map_err(db_err)?;
        conn.execute(Self::IDX_SHADOW_DDL, []).map_err(db_err)?;
        conn.execute(
            "CREATE TABLE rowmap_shadow(\
             rowid INTEGER PRIMARY KEY, note_id INTEGER NOT NULL, \
             source TEXT NOT NULL, ref TEXT NOT NULL)",
            [],
        )
        .map_err(db_err)?;
        Ok(())
    }

    /// Merge `idx_shadow` down to one segment (FTS5 `optimize`), on the write
    /// connection. Run BEFORE the swap so a freshly-built index lands compact with
    /// no recall window (the shadow is off the live tables). FTS5 `optimize` is
    /// physical-only — rowids, content, and the idx↔rowmap pairing are preserved —
    /// so the col_mod watermark and DF-based fuzzy ranking are untouched.
    fn optimize_shadow(&self) -> NativeResult<()> {
        let conn = self.lock();
        conn.execute("INSERT INTO idx_shadow(idx_shadow) VALUES('optimize')", [])
            .map_err(db_err)?;
        Ok(())
    }

    /// Copy the live recognition rows (every non-`field` source) into the shadow,
    /// re-pairing idx↔rowmap on the shadow connection. Chunked under short locks
    /// (the lock is released between chunks) so a concurrent search interleaves;
    /// the volume is bounded by recognized-content size, not the collection.
    fn copy_recognition_rows_to_shadow(&self) -> NativeResult<()> {
        // Read the live recognition (idx.txt, note_id, source, ref) rows once.
        let rows: Vec<(String, i64, String, String)> = {
            let conn = self.lock();
            let mut stmt = conn
                .prepare(
                    "SELECT idx.txt, m.note_id, m.source, m.ref FROM idx \
                     JOIN rowmap m ON m.rowid = idx.rowid WHERE m.source != 'field'",
                )
                .map_err(db_err)?;
            let out = stmt
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
                .map_err(db_err)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(db_err)?;
            out
        };
        for chunk in rows.chunks(Self::INLINE_ID_MAX) {
            let mut conn = self.lock();
            let tx = conn.transaction().map_err(db_err)?;
            Self::insert_shadow_rows(
                &tx,
                chunk
                    .iter()
                    .map(|(txt, nid, source, r)| (*nid, source.as_str(), r.as_str(), txt.as_str())),
            )?;
            tx.commit().map_err(db_err)?;
        }
        Ok(())
    }

    /// Insert ONE chunk of new `field` rows into the shadow under a short lock
    /// (released on return, so the actor-dependent pulls between chunks never
    /// block on it). Returns the rows seen (blank text skipped). The shadow's
    /// idx↔rowmap pairing rides `last_insert_rowid()` on this connection.
    fn insert_field_chunk_to_shadow(
        &self,
        chunk: &[(i64, String, String, String)],
    ) -> NativeResult<usize> {
        let mut conn = self.lock();
        let tx = conn.transaction().map_err(db_err)?;
        let seen = chunk.len();
        Self::insert_shadow_rows(
            &tx,
            chunk
                .iter()
                .map(|(nid, source, r, txt)| (*nid, source.as_str(), r.as_str(), txt.as_str())),
        )?;
        tx.commit().map_err(db_err)?;
        Ok(seen)
    }

    /// Insert `(note_id, source, ref, text)` rows into `idx_shadow`/
    /// `rowmap_shadow`, pairing the FTS5 rowid via `last_insert_rowid()` (blank
    /// text skipped, exactly like the live insert).
    fn insert_shadow_rows<'a>(
        tx: &rusqlite::Transaction<'_>,
        rows: impl Iterator<Item = (i64, &'a str, &'a str, &'a str)>,
    ) -> NativeResult<()> {
        let mut ins_idx = tx
            .prepare_cached("INSERT INTO idx_shadow(txt) VALUES(?1)")
            .map_err(db_err)?;
        let mut ins_map = tx
            .prepare_cached(
                "INSERT INTO rowmap_shadow(rowid, note_id, source, ref) VALUES(?1,?2,?3,?4)",
            )
            .map_err(db_err)?;
        for (note_id, source, reference, text) in rows {
            if text.trim().is_empty() {
                continue;
            }
            // NFC-normalize (also re-normalizes recognition rows carried from a
            // pre-normalization index on rebuild — idempotent for already-NFC text).
            ins_idx.execute([nfc(text).as_ref()]).map_err(db_err)?;
            let rowid = tx.last_insert_rowid();
            ins_map
                .execute(rusqlite::params![rowid, note_id, source, reference])
                .map_err(db_err)?;
        }
        Ok(())
    }

    /// The atomic swap (ONE short transaction): prune the shadow's recognition
    /// rows for notes no longer in the collection (and the live segments/gated
    /// markers for them — those tables are not swapped), then RENAME the shadow
    /// over the live `idx`/`rowmap`, and stamp `col_mod` LAST. SQLite renames an
    /// FTS5 table cleanly; the whole thing is one transaction, so a reader sees
    /// the complete old index until commit and the complete new one after — never
    /// a partial. That atomicity is for ROW VISIBILITY: the commit bumps the schema
    /// cookie, so a *fresh* statement on a separate pool connection that re-prepares
    /// and reconstructs the FTS5 `idx` vtable against the changing schema can
    /// transiently fail as `SQLITE_SCHEMA`; that is absorbed by the read path's
    /// retry ([`is_transient_swap_fault`]), not surfaced to callers.
    /// A crash before this leaves the live tables + old col_mod intact.
    /// `live_notes` is the authoritative set: a note can be live yet have no field
    /// rows, and its recognition rows must survive.
    fn swap_shadow_and_stamp(&self, live_notes: &[i64], col_mod: i64) -> NativeResult<()> {
        let mut conn = self.lock();
        let tx = conn.transaction().map_err(db_err)?;
        let live: std::collections::HashSet<i64> = live_notes.iter().copied().collect();
        // Recognition rows that were copied into the shadow but whose note is gone.
        let stale: Vec<i64> = {
            let mut stmt = tx
                .prepare(
                    "SELECT DISTINCT note_id FROM rowmap_shadow WHERE source != 'field' \
                     UNION SELECT DISTINCT note_id FROM gated",
                )
                .map_err(db_err)?;
            let ids: Vec<i64> = stmt
                .query_map([], |r| r.get(0))
                .map_err(db_err)?
                .collect::<Result<_, _>>()
                .map_err(db_err)?;
            ids.into_iter().filter(|n| !live.contains(n)).collect()
        };
        if !stale.is_empty() {
            // Drop the dead notes' recognition rows from the SHADOW (idx_shadow +
            // rowmap_shadow), and their segments + gated markers from the live
            // tables (not part of the idx/rowmap swap).
            let (id_clause, id_params) = Self::note_id_clause(&tx, "note_id", &stale)?;
            let params: Vec<&dyn rusqlite::ToSql> = id_params
                .iter()
                .map(|n| n as &dyn rusqlite::ToSql)
                .collect();
            tx.execute(
                &format!(
                    "DELETE FROM idx_shadow WHERE rowid IN \
                     (SELECT rowid FROM rowmap_shadow WHERE {id_clause})"
                ),
                rusqlite::params_from_iter(params.iter().copied()),
            )
            .map_err(db_err)?;
            tx.execute(
                &format!("DELETE FROM rowmap_shadow WHERE {id_clause}"),
                rusqlite::params_from_iter(params.iter().copied()),
            )
            .map_err(db_err)?;
            // The live segments + gated tables aren't part of the idx/rowmap
            // swap, so prune the dead notes from them directly.
            tx.execute(
                &format!("DELETE FROM segments WHERE {id_clause}"),
                rusqlite::params_from_iter(params.iter().copied()),
            )
            .map_err(db_err)?;
            Self::delete_gated(&tx, &stale, None)?;
        }
        // Atomic whole-table swap: the new index replaces the old in one step.
        tx.execute_batch(
            "DROP TABLE idx; \
             DROP TABLE rowmap; \
             ALTER TABLE idx_shadow RENAME TO idx; \
             ALTER TABLE rowmap_shadow RENAME TO rowmap;",
        )
        .map_err(db_err)?;
        // The rowmap indexes rode the old table; recreate them on the new one.
        tx.execute(
            "CREATE INDEX IF NOT EXISTS rowmap_note ON rowmap(note_id, source)",
            [],
        )
        .map_err(db_err)?;
        tx.execute(
            "CREATE INDEX IF NOT EXISTS rowmap_source ON rowmap(source, note_id)",
            [],
        )
        .map_err(db_err)?;
        tx.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('col_mod', ?1)",
            [col_mod],
        )
        .map_err(db_err)?;
        tx.commit().map_err(db_err)?;
        // Fold the WAL into the main file so the freshly-rebuilt index sits where
        // memory-mapped reads can serve it (mmap sees only the main file, never WAL
        // frames). Best-effort: a TRUNCATE checkpoint can return BUSY against a
        // concurrent reader, in which case the next writer / the autocheckpoint
        // cadence folds it in — a miss here only delays the mmap benefit. (The
        // steady-state checkpoint cadence is the index-maintenance issue, #938.)
        let _ = conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
        Ok(())
    }

    /// A free-form meta value (e.g. the recognizer fingerprint).
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn meta_get(&self, key: &str) -> NativeResult<Option<String>> {
        let conn = self.lock();
        Ok(conn
            .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| {
                r.get::<_, String>(0)
            })
            .ok())
    }

    /// Write a free-form meta key/value.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn meta_set(&self, key: &str, value: &str) -> NativeResult<()> {
        let conn = self.lock();
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?1, ?2)",
            rusqlite::params![key, value],
        )
        .map_err(db_err)?;
        Ok(())
    }

    /// Store one item's recognition structure (segments JSON — boxes
    /// for OCR, time spans for ASR) alongside its text row, keyed like the
    /// row. One pass, many consumers: occlusion reads these back.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn put_segments(
        &self,
        note_id: i64,
        source: &str,
        reference: &str,
        json: &str,
    ) -> NativeResult<()> {
        let conn = self.lock();
        conn.execute(
            "INSERT OR REPLACE INTO segments(note_id, source, ref, json) VALUES(?1,?2,?3,?4)",
            rusqlite::params![note_id, source, reference, json],
        )
        .map_err(db_err)?;
        Ok(())
    }

    /// The stored per-segment JSON for one `(note, source, ref)`, or `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn get_segments(
        &self,
        note_id: i64,
        source: &str,
        reference: &str,
    ) -> NativeResult<Option<String>> {
        let conn = self.lock();
        Ok(conn
            .query_row(
                "SELECT json FROM segments WHERE note_id=?1 AND source=?2 AND ref=?3",
                rusqlite::params![note_id, source, reference],
                |r| r.get::<_, String>(0),
            )
            .ok())
    }

    /// All (note_id, ref) pairs for one source — the pending sweep's "what
    /// has already been recognized" set. Deliberately a full-set read:
    /// the sweep's pending diff needs the complete set, and the pairs are
    /// small. Bounding belongs to the sweep's batching, not this query.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn refs_for_source(&self, source: &str) -> NativeResult<Vec<(i64, String)>> {
        let conn = self.lock();
        let mut stmt = conn
            .prepare("SELECT note_id, ref FROM rowmap WHERE source = ?1")
            .map_err(db_err)?;
        let rows = stmt
            .query_map([source], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(db_err)?
            .collect::<Result<Vec<(i64, String)>, _>>()
            .map_err(db_err)?;
        Ok(rows)
    }

    /// Record below-gate outcomes: each (note_id, ref) was recognized
    /// and the gate dropped it — no text row, but the pending sweep counts it
    /// done. One transaction per batch.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn mark_gated(&self, source: &str, pairs: &[(i64, String)]) -> NativeResult<()> {
        if pairs.is_empty() {
            return Ok(());
        }
        let mut conn = self.lock();
        let tx = conn.transaction().map_err(db_err)?;
        for (note_id, reference) in pairs {
            tx.execute(
                "INSERT OR REPLACE INTO gated(note_id, source, ref) VALUES(?1,?2,?3)",
                rusqlite::params![note_id, source, reference],
            )
            .map_err(db_err)?;
        }
        tx.commit().map_err(db_err)
    }

    /// All below-gate (note_id, ref) markers for one source — unioned with
    /// [`Self::refs_for_source`] by the pending sweep's done-set diff.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn gated_refs_for_source(&self, source: &str) -> NativeResult<Vec<(i64, String)>> {
        let conn = self.lock();
        let mut stmt = conn
            .prepare("SELECT note_id, ref FROM gated WHERE source = ?1")
            .map_err(db_err)?;
        let rows = stmt
            .query_map([source], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(db_err)?
            .collect::<Result<Vec<(i64, String)>, _>>()
            .map_err(db_err)?;
        Ok(rows)
    }

    /// Drop ALL below-gate markers for one source — the recognizer-fingerprint
    /// invalidation path: a new engine re-judges everything, gated
    /// items included, exactly like stored rows re-derive.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn clear_gated(&self, source: &str) -> NativeResult<()> {
        let conn = self.lock();
        conn.execute("DELETE FROM gated WHERE source = ?1", [source])
            .map_err(db_err)?;
        Ok(())
    }

    /// All (note_id, ref, text) rows for one source — the embed-input
    /// composition reads recognized text back for vector minting.
    /// Deliberately a full-set read (the composition consumes the whole set);
    /// volume is bounded by recognized-text size, not media size.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn texts_for_source(&self, source: &str) -> NativeResult<Vec<(i64, String, String)>> {
        let conn = self.lock();
        let mut stmt = conn
            .prepare(
                "SELECT m.note_id, m.ref, idx.txt FROM idx \
                 JOIN rowmap m ON m.rowid = idx.rowid WHERE m.source = ?1",
            )
            .map_err(db_err)?;
        let rows = stmt
            .query_map([source], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .map_err(db_err)?
            .collect::<Result<Vec<(i64, String, String)>, _>>()
            .map_err(db_err)?;
        Ok(rows)
    }

    /// `texts_for_source` scoped to a note set: the per-upsert embed
    /// composition needs only the WRITTEN notes' recognized texts — the
    /// full-set read belongs to rebuild/reconcile, not the op tail.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn texts_for_source_for_notes(
        &self,
        source: &str,
        note_ids: &[i64],
    ) -> NativeResult<Vec<(i64, String, String)>> {
        if note_ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.lock();
        let (id_clause, id_params) = Self::note_id_clause(&conn, "m.note_id", note_ids)?;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT m.note_id, m.ref, idx.txt FROM idx \
                 JOIN rowmap m ON m.rowid = idx.rowid WHERE m.source = ?1 AND {id_clause}"
            ))
            .map_err(db_err)?;
        let mut params: Vec<Box<dyn rusqlite::ToSql>> =
            vec![Box::new(source.to_string()) as Box<dyn rusqlite::ToSql>];
        params.extend(
            id_params
                .iter()
                .map(|n| Box::new(*n) as Box<dyn rusqlite::ToSql>),
        );
        let rows = stmt
            .query_map(
                rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())),
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .map_err(db_err)?
            .collect::<Result<Vec<(i64, String, String)>, _>>()
            .map_err(db_err)?;
        Ok(rows)
    }

    /// One FTS5 MATCH (rank-ordered), returning provenance + snippet rows.
    /// A bad expression is `invalid_input` — the facade maps it to its
    /// OperationalError fallback path. `scope`, when given, restricts the
    /// match to those note ids INSIDE the query (the scoped-search path:
    /// the id set comes from anki's indexed deck:/tag: search, so scoped
    /// literal search needs no over-fetch and no post-hoc recall gamble).
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn match_rows(
        &self,
        expr: &str,
        limit: i64,
        with_text: bool,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Vec<MatchRow>> {
        let span = tracing::debug_span!("derived.match", limit, with_text);
        let _enter = span.enter();
        let conn = self.lock();
        let txt_col = if with_text { "idx.txt" } else { "NULL" };
        let scope_clause = Self::scope_clause(&conn, scope)?;
        // Hidden-source exclusion: a VectorOnly recognition source is dropped
        // BEFORE ranking/limiting, so its rows never surface on a lexical query
        // yet stay stored for provenance + reconcile. Bound as positional params
        // (starting after the three fixed ones below) — the source strings are
        // kernel-controlled, but binding keeps the path injection-safe by
        // construction.
        let exclude_clause = if exclude_sources.is_empty() {
            String::new()
        } else {
            let placeholders = (0..exclude_sources.len())
                .map(|i| format!("?{}", i + 4))
                .collect::<Vec<_>>()
                .join(",");
            format!("AND m.source NOT IN ({placeholders}) ")
        };
        let sql = format!(
            "SELECT m.note_id, m.source, m.ref, {txt_col}, \
             snippet(idx, 0, '', '', '…', ?1) \
             FROM idx JOIN rowmap m ON m.rowid = idx.rowid \
             WHERE idx MATCH ?2 {scope_clause}{exclude_clause}ORDER BY rank LIMIT ?3"
        );
        // Retry a transient busy: two engines share the file, so a read can lose
        // the lock to a concurrent write even with `busy_timeout`. The closure
        // re-prepares + re-runs per attempt; a busy at prepare or step bubbles
        // as a `rusqlite::Error` for `with_busy_retry` to retry, while a
        // non-busy FTS5/MATCH fault returns `Ok(Err(invalid_input))` (a real
        // query error, not a lock — surfaced without retry). A busy surviving
        // the retries surfaces as `unavailable`; the kernel caller propagates it
        // rather than silently degrading to a fallback that can't serve OCR/ASR.
        let run = || -> rusqlite::Result<Result<Vec<MatchRow>, NativeError>> {
            let mut params: Vec<&dyn rusqlite::ToSql> = vec![&SNIPPET_TOKENS, &expr, &limit];
            params.extend(exclude_sources.iter().map(|s| s as &dyn rusqlite::ToSql));
            let mut stmt = conn.prepare(&sql)?;
            let mut q = stmt.query(rusqlite::params_from_iter(params))?;
            let mut rows: Vec<MatchRow> = Vec::new();
            loop {
                let row = match q.next() {
                    Ok(Some(r)) => r,
                    Ok(None) => break,
                    Err(e) if is_retryable(&e) => return Err(e), // retried
                    Err(e) => {
                        return Ok(Err(NativeError::invalid_input(format!("fts5 match: {e}"))))
                    }
                };
                match (|| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                })() {
                    Ok(tuple) => rows.push(tuple),
                    Err(e) if is_retryable(&e) => return Err(e),
                    Err(e) => {
                        return Ok(Err(NativeError::invalid_input(format!("fts5 match: {e}"))))
                    }
                }
            }
            Ok(Ok(rows))
        };
        with_busy_retry(run)?
    }

    /// [`Self::match_rows`] over many MATCH expressions on `conn`, sharing ONE
    /// compiled statement (through the connection's statement cache) across the
    /// whole batch. The fused-search lexical reads call this once for all query
    /// strings rather than re-resolving the scope and recompiling the statement per
    /// query. Returns one row vector per expression, in `exprs` order.
    ///
    /// `conn` is a checked-out [`ReadPool`] connection — read-only by discipline
    /// (SELECTs, plus a TEMP scope set only for a scope past the inline cap, never a
    /// store write).
    ///
    /// The scope is constant across the batch, so its clause — inline literals at or
    /// below the inline cap, a TEMP subquery above it (see [`Self::scope_clause`]) —
    /// leaves the SQL text, hence the statement-cache key, identical for every query
    /// in the batch, and the statement compiles once for the whole set.
    ///
    /// # Errors
    ///
    /// Returns an error if resolving the scope fails, or any expression's MATCH query
    /// fails (the batch stops at the first failure, like the singular read).
    fn match_rows_batch(
        conn: &Connection,
        exprs: &[String],
        limit: i64,
        with_text: bool,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Vec<Vec<MatchRow>>> {
        if exprs.is_empty() {
            return Ok(Vec::new()); // nothing to match — skip the staging
        }
        let span = tracing::debug_span!("derived.match_batch", n = exprs.len(), limit, with_text);
        let _enter = span.enter();
        let txt_col = if with_text { "idx.txt" } else { "NULL" };
        // The scope is constant across the batch, so inlining it as literals leaves
        // the SQL — the prepare_cached key — stable across the batch's queries (it
        // compiles once), while keeping the parallel chunks off the temp-table
        // pcache mutex a staged set would re-introduce. See [`Self::scope_clause`].
        let scope_clause = Self::scope_clause(conn, scope)?;
        let exclude_clause = if exclude_sources.is_empty() {
            String::new()
        } else {
            let placeholders = (0..exclude_sources.len())
                .map(|i| format!("?{}", i + 4))
                .collect::<Vec<_>>()
                .join(",");
            format!("AND m.source NOT IN ({placeholders}) ")
        };
        let sql = format!(
            "SELECT m.note_id, m.source, m.ref, {txt_col}, \
             snippet(idx, 0, '', '', '…', ?1) \
             FROM idx JOIN rowmap m ON m.rowid = idx.rowid \
             WHERE idx MATCH ?2 {scope_clause}{exclude_clause}ORDER BY rank LIMIT ?3"
        );
        let mut out: Vec<Vec<MatchRow>> = Vec::with_capacity(exprs.len());
        for expr in exprs {
            // prepare_cached: the statement compiles on the first expression and
            // every later one in the batch (and later searches of the same shape)
            // reuses it. The MATCH expression is the only thing that varies per
            // query, bound as ?2. Busy-retry stays per query — a surviving busy
            // surfaces as `unavailable`, exactly like the singular read.
            let run = || -> rusqlite::Result<Result<Vec<MatchRow>, NativeError>> {
                let mut params: Vec<&dyn rusqlite::ToSql> = vec![&SNIPPET_TOKENS, expr, &limit];
                params.extend(exclude_sources.iter().map(|s| s as &dyn rusqlite::ToSql));
                let mut stmt = conn.prepare_cached(&sql)?;
                let mut q = stmt.query(rusqlite::params_from_iter(params))?;
                let mut rows: Vec<MatchRow> = Vec::new();
                loop {
                    let row = match q.next() {
                        Ok(Some(r)) => r,
                        Ok(None) => break,
                        Err(e) if is_retryable(&e) => return Err(e), // retried
                        Err(e) => {
                            return Ok(Err(NativeError::invalid_input(format!("fts5 match: {e}"))))
                        }
                    };
                    match (|| {
                        Ok((
                            row.get(0)?,
                            row.get(1)?,
                            row.get(2)?,
                            row.get(3)?,
                            row.get(4)?,
                        ))
                    })() {
                        Ok(tuple) => rows.push(tuple),
                        Err(e) if is_retryable(&e) => return Err(e),
                        Err(e) => {
                            return Ok(Err(NativeError::invalid_input(format!("fts5 match: {e}"))))
                        }
                    }
                }
                Ok(Ok(rows))
            };
            out.push(with_busy_retry(run)??);
        }
        Ok(out)
    }

    /// Document frequency for each of `terms`, from the materialized `trigram_df`
    /// snapshot — the fuzzy path reads it to prune common trigrams. A term absent
    /// from the result is absent from the SNAPSHOT (treated as DF 0, sorting last in
    /// the prune); see [`Self::refresh_trigram_df`] for the freshness contract.
    /// ONE prepare-cached `term IN rarray(?1)` primary-key seek over the plain table
    /// (the in-memory carray binds the whole trigram set; mmap-served), NOT a doclist
    /// count through fts5vocab. Reads on the passed [`ReadPool`] connection.
    ///
    /// # Errors
    ///
    /// Returns an error if the lookup fails.
    fn trigram_dfs(
        conn: &Connection,
        terms: &[Trigram],
    ) -> NativeResult<std::collections::HashMap<Trigram, i64>> {
        if terms.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        let run = || -> rusqlite::Result<std::collections::HashMap<Trigram, i64>> {
            // The whole trigram set as an in-memory carray, so the term→df lookup is
            // ONE prepare-cached `IN rarray(?1)` statement over one implicit
            // transaction, versus a `query`/`reset` per trigram. Terms absent from
            // the table never come back — the caller reads a missing term as DF 0.
            let want = Self::term_array(terms.iter().map(Trigram::as_str));
            let mut stmt =
                conn.prepare_cached("SELECT term, df FROM trigram_df WHERE term IN rarray(?1)")?;
            let mut m = std::collections::HashMap::new();
            let mut q = stmt.query(rusqlite::params![want])?;
            while let Some(r) = q.next()? {
                m.insert(r.get::<_, Trigram>(0)?, r.get::<_, i64>(1)?);
            }
            Ok(m)
        };
        with_busy_retry(run)
    }
}

/// The store contract: every method forwards to the inherent impl, so
/// the concrete engine keeps its full API while the kernel consumes
/// `Arc<dyn DerivedStore>`.
impl shrike_store::DerivedStore for DerivedEngine {
    fn build(
        &self,
        rows: &[(i64, String, String, String)],
        live_notes: &[i64],
        col_mod: i64,
    ) -> NativeResult<()> {
        Self::build(self, rows, live_notes, col_mod)
    }
    fn build_streamed(
        &self,
        next: &mut dyn FnMut() -> Option<NativeResult<Vec<(i64, String, String, String)>>>,
        live_notes: &[i64],
        col_mod: i64,
    ) -> NativeResult<usize> {
        Self::build_streamed(self, next, live_notes, col_mod)
    }
    fn ingest(
        &self,
        note_id: i64,
        source: &str,
        refs_text: &[(String, String)],
    ) -> NativeResult<()> {
        Self::ingest(self, note_id, source, refs_text)
    }
    fn ingest_many(
        &self,
        notes: &[(i64, Vec<(String, String)>)],
        source: &str,
    ) -> NativeResult<()> {
        Self::ingest_many(self, notes, source)
    }
    fn refresh_derived_snapshots(&self) -> NativeResult<()> {
        // The fold refreshes trigram_df itself (in its own txn, so promote/demote sees
        // a consistent DF), then re-tiers the bitmaps incrementally over the dirty set.
        Self::fold_trigram_bitmaps(self)
    }
    fn remove(&self, note_ids: &[i64], source: Option<&str>) -> NativeResult<()> {
        Self::remove(self, note_ids, source)
    }
    fn count(&self) -> NativeResult<i64> {
        Self::count(self)
    }
    fn get_col_mod(&self) -> Option<i64> {
        Self::get_col_mod(self)
    }
    fn set_col_mod(&self, value: i64) -> NativeResult<()> {
        Self::set_col_mod(self, value)
    }
    fn meta_get(&self, key: &str) -> NativeResult<Option<String>> {
        Self::meta_get(self, key)
    }
    fn meta_set(&self, key: &str, value: &str) -> NativeResult<()> {
        Self::meta_set(self, key, value)
    }
    fn refs_for_source(&self, source: &str) -> NativeResult<Vec<(i64, String)>> {
        Self::refs_for_source(self, source)
    }
    fn texts_for_source(&self, source: &str) -> NativeResult<Vec<(i64, String, String)>> {
        Self::texts_for_source(self, source)
    }
    fn texts_for_source_for_notes(
        &self,
        source: &str,
        note_ids: &[i64],
    ) -> NativeResult<Vec<(i64, String, String)>> {
        Self::texts_for_source_for_notes(self, source, note_ids)
    }
    fn mark_gated(&self, source: &str, pairs: &[(i64, String)]) -> NativeResult<()> {
        Self::mark_gated(self, source, pairs)
    }
    fn gated_refs_for_source(&self, source: &str) -> NativeResult<Vec<(i64, String)>> {
        Self::gated_refs_for_source(self, source)
    }
    fn clear_gated(&self, source: &str) -> NativeResult<()> {
        Self::clear_gated(self, source)
    }
    fn put_segments(
        &self,
        note_id: i64,
        source: &str,
        reference: &str,
        json: &str,
    ) -> NativeResult<()> {
        Self::put_segments(self, note_id, source, reference, json)
    }
    fn get_segments(
        &self,
        note_id: i64,
        source: &str,
        reference: &str,
    ) -> NativeResult<Option<String>> {
        Self::get_segments(self, note_id, source, reference)
    }
    fn match_rows(
        &self,
        expr: &str,
        limit: i64,
        with_text: bool,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Vec<MatchRow>> {
        Self::match_rows(self, expr, limit, with_text, scope, exclude_sources)
    }
    fn search_substring(
        &self,
        query: &str,
        limit: i64,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Option<Vec<LexicalRow>>> {
        Self::search_substring(self, query, limit, scope, exclude_sources)
    }
    fn search_fuzzy(
        &self,
        query: &str,
        top_k: i64,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Vec<LexicalRow>> {
        Self::search_fuzzy(self, query, top_k, scope, exclude_sources)
    }
    fn search_substring_batch(
        &self,
        queries: &[&str],
        limit: i64,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Vec<Option<Vec<LexicalRow>>>> {
        Self::search_substring_batch(self, queries, limit, scope, exclude_sources)
    }
    fn search_fuzzy_batch(
        &self,
        queries: &[&str],
        top_k: i64,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Vec<Vec<LexicalRow>>> {
        Self::search_fuzzy_batch(self, queries, top_k, scope, exclude_sources)
    }
}

/// `build` with the live note set taken from the snapshot's own rows — the
/// "no notes vanished and none are missing from the snapshot" case most tests
/// want. Tests exercising the prune pass an explicit `live_notes`. Shared by
/// every test module below.
#[cfg(test)]
fn build_snapshot_live(
    e: &DerivedEngine,
    rows: &[(i64, String, String, String)],
    col_mod: i64,
) -> NativeResult<()> {
    let live: Vec<i64> = rows.iter().map(|r| r.0).collect();
    e.build(rows, &live, col_mod)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (DerivedEngine, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "shrike-derived-{}-{}",
            std::process::id(),
            rand_suffix()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shrike.db");
        (DerivedEngine::open(path.to_str().unwrap(), 1).unwrap(), dir)
    }

    fn rand_suffix() -> u64 {
        // Unique per call (tests run in parallel — a timestamp alone can collide).
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }

    #[test]
    fn fts5_trigram_always_available() {
        // The bundled-SQLite win: creating the trigram FTS5 table just works.
        let (_e, dir) = store();
        std::fs::remove_dir_all(dir).ok();
    }

    fn busy_err() -> rusqlite::Error {
        rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            Some("database is locked".into()),
        )
    }

    #[test]
    fn busy_retry_succeeds_after_transient_busy() {
        // A read that hits a transient SQLITE_BUSY a few times
        // then succeeds is RETRIED to success — it never surfaces as an error.
        use std::cell::Cell;
        let calls = Cell::new(0usize);
        let out: i32 = with_busy_retry(|| {
            let n = calls.get();
            calls.set(n + 1);
            if n < 3 {
                Err(busy_err())
            } else {
                Ok(42)
            }
        })
        .expect("a transient busy is retried to success");
        assert_eq!(out, 42);
        assert_eq!(calls.get(), 4, "3 busies + 1 success");
    }

    #[test]
    fn busy_retry_surfaces_a_persistent_busy_as_unavailable() {
        // A busy that outlives the retries surfaces (as `unavailable`) — the
        // kernel caller then propagates it rather than silently degrading to a
        // fallback that can't serve OCR/ASR text.
        let err = with_busy_retry::<i32>(|| Err(busy_err())).unwrap_err();
        assert_eq!(err.kind(), shrike_error::ErrorKind::Unavailable);
    }

    #[test]
    fn busy_retry_does_not_retry_a_non_busy_error() {
        // A genuine (non-busy) error is NOT retried — it surfaces immediately,
        // so a real fault isn't masked by the busy retry.
        use std::cell::Cell;
        let calls = Cell::new(0usize);
        let err = with_busy_retry::<i32>(|| {
            calls.set(calls.get() + 1);
            Err(rusqlite::Error::InvalidQuery)
        })
        .unwrap_err();
        assert_eq!(calls.get(), 1, "a non-busy error is not retried");
        assert_eq!(err.kind(), shrike_error::ErrorKind::Unavailable); // db_err maps all to unavailable
    }

    fn sqlite_err(code: i32, msg: &str) -> rusqlite::Error {
        rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(code), Some(msg.to_string()))
    }

    #[test]
    fn is_retryable_covers_schema_change_but_not_genuine_errors() {
        use rusqlite::ffi;
        // The rebuild shadow-swap, committing on the write connection, bumps the
        // schema cookie, so a FRESH statement on a separate pool connection that
        // re-prepares + reconstructs the FTS5 vtable transiently fails as
        // SQLITE_SCHEMA ("vtable constructor failed: idx"). It MUST be retried —
        // else a valid search racing a rebuild fails outright (the High this guards).
        // Matched BY CODE: the message is incidental.
        assert!(is_retryable(&sqlite_err(
            ffi::SQLITE_SCHEMA,
            "vtable constructor failed: idx"
        )));
        assert!(is_retryable(&sqlite_err(
            ffi::SQLITE_SCHEMA,
            "database schema has changed"
        )));
        // Busy/locked stay retryable (unchanged from before this fix).
        assert!(is_retryable(&sqlite_err(
            ffi::SQLITE_BUSY,
            "database is locked"
        )));
        assert!(is_retryable(&sqlite_err(
            ffi::SQLITE_LOCKED,
            "database table is locked"
        )));

        // A GENUINE error must NOT be retried — masking it would hide a real fault
        // and add a retry-budget latency tail. Genuine query errors carry the
        // generic SQLITE_ERROR code, DISJOINT from SQLITE_SCHEMA — even a message
        // that *looks* swap-related does not retry (the predicate keys on code, not
        // text), so the old brittle message-match can't misfire.
        assert!(!is_retryable(&sqlite_err(
            ffi::SQLITE_ERROR,
            "near \"slect\": syntax error"
        )));
        assert!(!is_retryable(&sqlite_err(
            ffi::SQLITE_ERROR,
            "no such column: idx"
        )));
        assert!(!is_retryable(&sqlite_err(
            ffi::SQLITE_ERROR,
            "no such table: idx" // SQLITE_ERROR, not SCHEMA — a real missing table is permanent
        )));
        assert!(!is_retryable(&sqlite_err(
            ffi::SQLITE_CONSTRAINT,
            "UNIQUE constraint failed: idx.x"
        )));
        assert!(!is_retryable(&rusqlite::Error::InvalidQuery));
    }

    #[test]
    fn scoped_match_restricts_to_the_id_set() {
        // The scoped-search path: the id set rides INSIDE the FTS5
        // query, so a scoped literal/fuzzy search has exact recall within
        // scope and zero hits outside it.
        let (e, _dir) = store();
        build_snapshot_live(
            &e,
            &[
                (1, "field".into(), "Front".into(), "the krebs cycle".into()),
                (
                    2,
                    "field".into(),
                    "Front".into(),
                    "the krebs cycle too".into(),
                ),
                (3, "field".into(), "Front".into(), "unrelated text".into()),
            ],
            100,
        )
        .unwrap();

        // Unscoped: both literal hits.
        let all = e.search_substring("krebs", 10, None, &[]).unwrap().unwrap();
        let ids: Vec<i64> = all.iter().map(|r| r.0).collect();
        assert!(ids.contains(&1) && ids.contains(&2));

        // Scoped to note 2 only.
        let scoped = e
            .search_substring("krebs", 10, Some(&[2]), &[])
            .unwrap()
            .unwrap();
        assert_eq!(scoped.iter().map(|r| r.0).collect::<Vec<_>>(), vec![2]);

        // An empty scope matches nothing (never falls open).
        let none = e
            .search_substring("krebs", 10, Some(&[]), &[])
            .unwrap()
            .unwrap();
        assert!(none.is_empty());

        // Fuzzy honors the same scope.
        let fz = e.search_fuzzy("kreps cycle", 10, Some(&[1]), &[]).unwrap();
        assert_eq!(fz.iter().map(|r| r.0).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn probe_reports_linkage_capability() {
        // Under the bundled default the probe MUST pass (the bundled-SQLite
        // guarantee); under platform linkage it reports whatever the host
        // library has — and since this test only runs when the store above
        // worked, the probe must agree.
        assert!(fts5_trigram_available());
        if sqlite_bundled() {
            assert!(fts5_trigram_available());
        }
    }

    #[test]
    fn ingest_many_matches_per_note_ingest() {
        // One-transaction batch ingest is behavior-identical to the
        // per-note loop it replaces: rows replaced per (note, source), blank
        // texts skipped, other notes untouched.
        let (e, _dir) = store();
        e.ingest(1, "field", &[("Front".into(), "old text one".into())])
            .unwrap();
        e.ingest_many(
            &[
                (1, vec![("Front".into(), "new text one".into())]),
                (
                    2,
                    vec![
                        ("Front".into(), "text two".into()),
                        ("Back".into(), "  ".into()),
                    ],
                ),
                (3, vec![]),
            ],
            "field",
        )
        .unwrap();
        // Note 1 replaced (old gone), note 2 has exactly its non-blank row,
        // note 3 has none.
        let hits = e
            .search_substring("new text one", 10, None, &[])
            .unwrap()
            .unwrap();
        assert_eq!(hits.iter().map(|r| r.0).collect::<Vec<_>>(), vec![1]);
        let old = e
            .search_substring("old text one", 10, None, &[])
            .unwrap()
            .unwrap();
        assert!(old.is_empty(), "the pre-batch row must be replaced");
        assert_eq!(e.count().unwrap(), 2);
    }

    #[test]
    fn texts_for_source_for_notes_scopes_to_the_id_set() {
        let (e, _dir) = store();
        e.ingest(1, "ocr", &[("a.png".into(), "alpha text".into())])
            .unwrap();
        e.ingest(2, "ocr", &[("b.png".into(), "beta text".into())])
            .unwrap();
        e.ingest(3, "field", &[("Front".into(), "gamma".into())])
            .unwrap();

        let scoped = e.texts_for_source_for_notes("ocr", &[2, 3]).unwrap();
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].0, 2);
        assert_eq!(scoped[0].2, "beta text");
        assert!(e.texts_for_source_for_notes("ocr", &[]).unwrap().is_empty());
        // Agrees with the full read, filtered.
        let full = e.texts_for_source("ocr").unwrap();
        assert_eq!(full.len(), 2);
    }

    #[test]
    fn ingest_many_and_scoped_texts_work_beyond_the_inline_cap() {
        // The staged-temp-table branch (> INLINE_ID_MAX ids) changes the SQL
        // shape (id params go empty; only `source` stays bound) — pin it for
        // both new APIs.
        let (e, _dir) = store();
        let n = DerivedEngine::INLINE_ID_MAX + 7;
        let batch: Vec<(i64, Vec<(String, String)>)> = (0..n as i64)
            .map(|i| {
                (
                    i + 1,
                    vec![("Front".to_string(), format!("text number {i}"))],
                )
            })
            .collect();
        e.ingest_many(&batch, "ocr").unwrap();
        assert_eq!(e.count().unwrap(), n as i64);

        let all_ids: Vec<i64> = (1..=n as i64).collect();
        let scoped = e.texts_for_source_for_notes("ocr", &all_ids).unwrap();
        let full = e.texts_for_source("ocr").unwrap();
        assert_eq!(scoped.len(), full.len());
        assert_eq!(scoped.len(), n);

        // Re-ingesting the same ids beyond the cap REPLACES (the staged
        // delete half), never accumulates.
        e.ingest_many(&batch, "ocr").unwrap();
        assert_eq!(e.count().unwrap(), n as i64);
    }

    #[test]
    fn ingest_many_duplicate_ids_last_entry_wins() {
        let (e, _dir) = store();
        e.ingest_many(
            &[
                (1, vec![("Front".into(), "first version".into())]),
                (1, vec![("Front".into(), "second version".into())]),
            ],
            "field",
        )
        .unwrap();
        assert_eq!(e.count().unwrap(), 1);
        let hits = e
            .search_substring("second version", 10, None, &[])
            .unwrap()
            .unwrap();
        assert_eq!(hits.iter().map(|r| r.0).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn build_ingest_remove_count_round_trip() {
        let (e, dir) = store();
        build_snapshot_live(
            &e,
            &[
                (1, "field".into(), "Front".into(), "the mitochondria".into()),
                (
                    2,
                    "field".into(),
                    "Front".into(),
                    "powerhouse of the cell".into(),
                ),
                (3, "field".into(), "Back".into(), "   ".into()), // blank → skipped
            ],
            100,
        )
        .unwrap();
        assert_eq!(e.count().unwrap(), 2);
        assert_eq!(e.get_col_mod(), Some(100));

        e.ingest(1, "field", &[("Front".into(), "the chloroplast".into())])
            .unwrap();
        assert_eq!(e.count().unwrap(), 2);
        let hits = e
            .match_rows("\"chloroplast\"", 10, false, None, &[])
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1);
        assert_eq!(hits[0].1, "field");
        assert_eq!(hits[0].2, "Front");

        e.remove(&[1], None).unwrap();
        assert_eq!(e.count().unwrap(), 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rebuild_over_field_only_rows_resets_and_reindexes() {
        // With no recognition rows to preserve, the rebuild swaps the
        // row-by-row FTS5 delete for a drop-and-recreate. A second build
        // over an already-populated store must leave exactly the new rows
        // searchable (the rowid↔rowmap pairing is rebuilt from scratch).
        let (e, dir) = store();
        build_snapshot_live(
            &e,
            &[
                (1, "field".into(), "F".into(), "alpha alpha".into()),
                (2, "field".into(), "F".into(), "beta beta".into()),
            ],
            1,
        )
        .unwrap();
        assert_eq!(e.count().unwrap(), 2);
        build_snapshot_live(
            &e,
            &[(2, "field".into(), "F".into(), "gamma gamma".into())],
            2,
        )
        .unwrap();
        assert_eq!(e.count().unwrap(), 1);
        assert!(e
            .match_rows("\"alpha\"", 10, false, None, &[])
            .unwrap()
            .is_empty());
        let hits = e.match_rows("\"gamma\"", 10, false, None, &[]).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 2);
        assert_eq!(e.get_col_mod(), Some(2));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn streamed_rebuild_never_empties_the_index_mid_build() {
        // The no-recall-cliff guarantee: a streamed rebuild builds the new index
        // into a SHADOW and swaps it over the live one atomically, never touching
        // the live field rows until the swap — so a search landing mid-rebuild
        // always finds the FULL OLD index (then the full new one after the swap).
        // Probe the store from INSIDE the chunk stream (between chunk
        // transactions, the lock is free) and assert every note remains
        // searchable at every step.
        let (e, dir) = store();
        // Seed three notes, each its own searchable term.
        build_snapshot_live(
            &e,
            &[
                (1, "field".into(), "F".into(), "alpha unique".into()),
                (2, "field".into(), "F".into(), "beta unique".into()),
                (3, "field".into(), "F".into(), "gamma unique".into()),
            ],
            1,
        )
        .unwrap();

        // Rebuild streaming ONE note per chunk, probing between chunks. The new
        // text keeps each note's term so it stays findable across the swap.
        let chunks = std::cell::RefCell::new(
            vec![
                vec![(
                    1i64,
                    "field".to_string(),
                    "F".to_string(),
                    "alpha unique".to_string(),
                )],
                vec![(
                    2,
                    "field".to_string(),
                    "F".to_string(),
                    "beta unique".to_string(),
                )],
                vec![(
                    3,
                    "field".to_string(),
                    "F".to_string(),
                    "gamma unique".to_string(),
                )],
            ]
            .into_iter(),
        );
        #[allow(clippy::type_complexity)]
        let mut next = || -> Option<NativeResult<Vec<(i64, String, String, String)>>> {
            // BEFORE handing the next chunk, every seeded note must still match —
            // proof that no prior step emptied the index.
            for term in ["alpha", "beta", "gamma"] {
                let hits = e
                    .search_substring(term, 10, None, &[])
                    .unwrap()
                    .unwrap_or_default();
                assert!(
                    !hits.is_empty(),
                    "'{term}' was not searchable mid-rebuild — recall cliff"
                );
            }
            chunks.borrow_mut().next().map(Ok)
        };
        let live = [1i64, 2, 3];
        e.build_streamed(&mut next, &live, 2).unwrap();

        // After the rebuild all three remain, at the new col_mod.
        for term in ["alpha", "beta", "gamma"] {
            assert!(!e
                .search_substring(term, 10, None, &[])
                .unwrap()
                .unwrap_or_default()
                .is_empty());
        }
        assert_eq!(e.get_col_mod(), Some(2));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn build_streamed_swap_clobbers_a_mid_build_ingest_but_stamps_the_snapshot_col_mod() {
        // #1002: an `ingest_many` landing in the LIVE store AFTER `build_streamed`'s
        // snapshot but BEFORE its swap is discarded by the swap — the shadow was built
        // from the snapshot, which didn't include it. The recovery hook is the col_mod
        // stamp: the swap stamps the SNAPSHOT col_mod it was given, NEVER a value
        // covering the concurrent ingest, so the derived watermark stays BELOW the
        // collection `col.mod` that ingest advanced → the kernel's drift check re-fires
        // and a later rebuild re-includes it (rather than the stamp masking the gap).
        //
        // The production kernel path serializes `rebuild_derived` on the sole-writer
        // ingest actor, so this interleaving can't occur there (pinned by
        // `rebuild_derived_serializes_a_concurrent_upsert` in the kernel crate); this
        // pins the build_streamed-level stamping the recovery rides on if it ever could.
        let (e, dir) = store();
        build_snapshot_live(
            &e,
            &[(1, "field".into(), "F".into(), "alpha unique".into())],
            100,
        )
        .unwrap();

        let injected = std::cell::Cell::new(false);
        let yielded = std::cell::Cell::new(false);
        #[allow(clippy::type_complexity)]
        let mut next = || -> Option<NativeResult<Vec<(i64, String, String, String)>>> {
            if !injected.get() {
                injected.set(true);
                // Note B lands in the LIVE index mid-build (the lock is free between
                // chunks), after the snapshot the producer is replaying below.
                e.ingest_many(
                    &[(2, vec![("F".to_string(), "bravo unique".to_string())])],
                    "field",
                )
                .unwrap();
                assert!(
                    !e.search_substring("bravo", 10, None, &[])
                        .unwrap()
                        .unwrap_or_default()
                        .is_empty(),
                    "B is in the LIVE index mid-build"
                );
            }
            if yielded.get() {
                None
            } else {
                yielded.set(true);
                // The snapshot the producer captured: note 1 only (B landed after it).
                Some(Ok(vec![(
                    1i64,
                    "field".to_string(),
                    "F".to_string(),
                    "alpha unique".to_string(),
                )]))
            }
        };
        // Live set + col_mod are the SNAPSHOT's (note 1 @ 100); B is not in either.
        e.build_streamed(&mut next, &[1], 100).unwrap();

        // The swap replaced live with the snapshot-built shadow → B's mid-build rows
        // are gone (the hazard).
        assert!(
            e.search_substring("bravo", 10, None, &[])
                .unwrap()
                .unwrap_or_default()
                .is_empty(),
            "the swap discarded B's mid-build ingest"
        );
        // …but the stamp is the SNAPSHOT col_mod (100), never advanced to cover B — so
        // the derived watermark trails the live col.mod and drift catches the gap.
        assert_eq!(
            e.get_col_mod(),
            Some(100),
            "the swap stamps the snapshot col_mod, not a value masking the concurrent ingest"
        );
        // The snapshot's own note survived the swap.
        assert!(
            !e.search_substring("alpha", 10, None, &[])
                .unwrap()
                .unwrap_or_default()
                .is_empty(),
            "alpha (in the snapshot) survived the swap"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn streamed_rebuild_error_leaves_the_live_index_untouched() {
        // Build-and-swap atomicity: a mid-stream chunk error aborts the rebuild
        // BEFORE the swap, so the live index + col_mod are exactly as they were —
        // the failed rebuild reads as drift next boot and retries (no partial
        // state, the property the old single-transaction rebuild had).
        let (e, dir) = store();
        build_snapshot_live(
            &e,
            &[
                (1, "field".into(), "F".into(), "alpha original".into()),
                (2, "field".into(), "F".into(), "beta original".into()),
            ],
            7,
        )
        .unwrap();

        // Stream new text, then ERROR before the stream completes.
        let mut step = 0;
        #[allow(clippy::type_complexity)]
        let mut next = || -> Option<NativeResult<Vec<(i64, String, String, String)>>> {
            step += 1;
            match step {
                1 => Some(Ok(vec![(
                    1i64,
                    "field".to_string(),
                    "F".to_string(),
                    "alpha rebuilt".to_string(),
                )])),
                _ => Some(Err(NativeError::internal(
                    "simulated mid-stream read failure",
                ))),
            }
        };
        let live = [1i64, 2];
        let err = e.build_streamed(&mut next, &live, 99).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Internal);

        // The live index is UNCHANGED: the OLD text still matches, the new text
        // never landed, and col_mod stayed at 7 (so drift re-fires).
        assert!(!e
            .search_substring("alpha original", 10, None, &[])
            .unwrap()
            .unwrap_or_default()
            .is_empty());
        assert!(e
            .search_substring("alpha rebuilt", 10, None, &[])
            .unwrap()
            .unwrap_or_default()
            .is_empty());
        assert_eq!(e.get_col_mod(), Some(7));

        // A subsequent clean rebuild still works (the leftover shadow is reset).
        build_snapshot_live(
            &e,
            &[(1, "field".into(), "F".into(), "alpha healed".into())],
            8,
        )
        .unwrap();
        assert!(!e
            .search_substring("alpha healed", 10, None, &[])
            .unwrap()
            .unwrap_or_default()
            .is_empty());
        assert_eq!(e.get_col_mod(), Some(8));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn match_returns_snippet_and_text_when_asked() {
        let (e, dir) = store();
        build_snapshot_live(
            &e,
            &[(7, "field".into(), "F".into(), "alpha beta gamma".into())],
            1,
        )
        .unwrap();
        let rows = e.match_rows("\"beta\"", 10, true, None, &[]).unwrap();
        assert_eq!(rows[0].3.as_deref(), Some("alpha beta gamma"));
        assert!(rows[0].4.as_deref().unwrap().contains("beta"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn bad_match_expression_is_invalid_input() {
        let (e, dir) = store();
        build_snapshot_live(&e, &[(1, "field".into(), "F".into(), "abc".into())], 1).unwrap();
        let err = e.match_rows("AND AND (", 10, false, None, &[]).unwrap_err();
        assert_eq!(err.kind(), shrike_error::ErrorKind::InvalidInput);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rebuild_preserves_recognition_rows_and_prunes_orphans() {
        // A drift rebuild replaces `field` rows but never discards
        // recognition-derived rows (re-recognition is expensive) — except for
        // notes that vanished from the collection.
        let (e, _dir) = store();
        build_snapshot_live(
            &e,
            &[(1, "field".into(), "Front".into(), "the mitochondria".into())],
            100,
        )
        .unwrap();
        e.ingest(
            1,
            "ocr",
            &[("diagram.png".into(), "electron transport chain".into())],
        )
        .unwrap();
        e.put_segments(
            1,
            "ocr",
            "diagram.png",
            r#"[{"text":"electron","confidence":0.9}]"#,
        )
        .unwrap();
        e.ingest(2, "ocr", &[("gone.png".into(), "orphaned text".into())])
            .unwrap();

        // Rebuild with note 1 present, note 2 gone.
        build_snapshot_live(
            &e,
            &[(
                1,
                "field".into(),
                "Front".into(),
                "the mitochondria EDITED".into(),
            )],
            200,
        )
        .unwrap();

        // Note 1's OCR row + segments survived; note 2's is pruned.
        let ocr = e.refs_for_source("ocr").unwrap();
        assert_eq!(ocr, vec![(1, "diagram.png".to_string())]);
        assert!(e.get_segments(1, "ocr", "diagram.png").unwrap().is_some());
        assert!(e.get_segments(2, "ocr", "gone.png").unwrap().is_none());

        // The OCR text is still searchable; the edited field text too.
        let hits = e
            .search_substring("electron transport", 10, None, &[])
            .unwrap()
            .unwrap();
        assert_eq!(hits[0].0, 1);
        assert_eq!(hits[0].1, "ocr");
        let field_hits = e
            .search_substring("EDITED", 10, None, &[])
            .unwrap()
            .unwrap();
        assert_eq!(field_hits[0].1, "field");

        // texts_for_source reads recognized text back for vector minting.
        let texts = e.texts_for_source("ocr").unwrap();
        assert_eq!(
            texts,
            vec![(
                1,
                "diagram.png".to_string(),
                "electron transport chain".to_string()
            )]
        );

        // remove(note, Some("ocr")) clears the row AND its segments.
        e.remove(&[1], Some("ocr")).unwrap();
        assert!(e.refs_for_source("ocr").unwrap().is_empty());
        assert!(e.get_segments(1, "ocr", "diagram.png").unwrap().is_none());

        // Meta keys round-trip (the recognizer fingerprint home).
        assert!(e.meta_get("recognizer_fingerprint").unwrap().is_none());
        e.meta_set("recognizer_fingerprint", "vision:1").unwrap();
        assert_eq!(
            e.meta_get("recognizer_fingerprint").unwrap().as_deref(),
            Some("vision:1")
        );
    }

    /// A rebuild whose field-row snapshot is STALE — it predates a note the
    /// collection now has — must keep that note's recognition rows. The prune
    /// keys off `live_notes` (the collection), not the field snapshot: a live
    /// note can contribute no field rows (all-blank fields, or a snapshot taken
    /// before the note was written). Without this, the boot rebuild raced a
    /// recognition sweep and dropped the row; the kernel's converge loop could
    /// not heal it because a recognition ingest does not bump col.mod.
    #[test]
    fn rebuild_with_stale_field_snapshot_keeps_a_live_notes_recognition_rows() {
        let (e, _dir) = store();
        // Note 1's OCR row is in the store (a recognition sweep ingested it).
        e.ingest(
            1,
            "ocr",
            &[("diagram.png".into(), "electron transport chain".into())],
        )
        .unwrap();

        // The field-row snapshot carries no rows for note 1 (taken before the
        // note was written), but note 1 IS live in the collection.
        e.build(&[], &[1], 200).unwrap();

        // The recognition row survived the stale-snapshot rebuild.
        let ocr = e.refs_for_source("ocr").unwrap();
        assert_eq!(
            ocr,
            vec![(1, "diagram.png".to_string())],
            "a live note's OCR row must survive a rebuild whose field snapshot \
             predates the note"
        );
        let hits = e
            .search_substring("electron transport", 10, None, &[])
            .unwrap()
            .unwrap();
        assert_eq!(hits[0].0, 1);
        assert_eq!(hits[0].1, "ocr");

        // A note genuinely absent from `live_notes` is still pruned.
        e.ingest(2, "ocr", &[("gone.png".into(), "orphaned".into())])
            .unwrap();
        e.build(&[], &[1], 201).unwrap();
        assert_eq!(
            e.refs_for_source("ocr").unwrap(),
            vec![(1, "diagram.png".to_string())],
            "note 2 — not in live_notes — is pruned"
        );
    }

    #[test]
    fn gated_markers_persist_survive_ingest_and_invalidate() {
        // Below-gate markers round-trip, survive a sibling image's
        // ingest (the replace must not put a gated item back in the pending
        // set), drop with note removal, prune with dead notes on rebuild,
        // and clear wholesale on fingerprint invalidation.
        let (e, _dir) = store();
        assert!(e.gated_refs_for_source("ocr").unwrap().is_empty());
        e.mark_gated(
            "ocr",
            &[(1, "tiny.png".to_string()), (2, "logo.png".to_string())],
        )
        .unwrap();
        // Re-marking is idempotent (INSERT OR REPLACE on the keyed table).
        e.mark_gated("ocr", &[(1, "tiny.png".to_string())]).unwrap();
        let mut got = e.gated_refs_for_source("ocr").unwrap();
        got.sort();
        assert_eq!(
            got,
            vec![(1, "tiny.png".to_string()), (2, "logo.png".to_string())]
        );
        // Markers are source-scoped.
        assert!(e.gated_refs_for_source("asr").unwrap().is_empty());

        // ingest (note 1 stores a DIFFERENT image's text) preserves markers.
        e.ingest(1, "ocr", &[("big.png".into(), "substantive text".into())])
            .unwrap();
        assert!(e
            .gated_refs_for_source("ocr")
            .unwrap()
            .contains(&(1, "tiny.png".to_string())));

        // remove (note deletion) drops the note's markers with its rows.
        e.remove(&[1], None).unwrap();
        assert_eq!(
            e.gated_refs_for_source("ocr").unwrap(),
            vec![(2, "logo.png".to_string())]
        );

        // A rebuild prunes markers of notes gone from the collection — even
        // marker-only notes (note 2 has no text rows at all).
        build_snapshot_live(
            &e,
            &[(3, "field".into(), "Front".into(), "still here".into())],
            100,
        )
        .unwrap();
        assert!(e.gated_refs_for_source("ocr").unwrap().is_empty());

        // clear_gated drops the whole source (fingerprint invalidation).
        e.mark_gated("ocr", &[(3, "x.png".to_string())]).unwrap();
        e.clear_gated("ocr").unwrap();
        assert!(e.gated_refs_for_source("ocr").unwrap().is_empty());
    }

    #[test]
    fn schema_version_bump_resets_data() {
        let (e, dir) = store();
        build_snapshot_live(&e, &[(1, "field".into(), "F".into(), "abc".into())], 9).unwrap();
        drop(e);
        let path = dir.join("shrike.db");
        let e2 = DerivedEngine::open(path.to_str().unwrap(), 2).unwrap();
        assert_eq!(e2.count().unwrap(), 0);
        assert_eq!(e2.get_col_mod(), None);
        std::fs::remove_dir_all(dir).ok();
    }
}

// ── lexical search policy ────────────────────────────────────────────────────
// MATCH-expression building + result filtering. One implementation: the Python
// facade delegates here through the binding, and the kernel's search assembly
// calls it directly.

/// FTS5's trigram tokenizer can't match a term shorter than 3 chars.
pub const MIN_TRIGRAM: usize = 3;
/// A fuzzy candidate must share at least this many query trigrams (noise floor).
pub const FUZZY_MIN_SHARED: usize = 2;
/// Cap on the trigrams a fuzzy `OR` generates candidates from: the rarest (most
/// discriminative) N of the query's trigrams. Bounds the match set `ORDER BY rank`
/// (bm25) scans — the search hotspot — without dropping typo recall (a typo'd word
/// has only a handful of trigrams, all kept). A perf/recall dial.
pub const FUZZY_MAX_TRIGRAMS: usize = 6;

/// Default document-frequency ceiling `C` for base-bitmap materialization
/// ([`DerivedEngine::materialize_ceiling`]): a trigram present in FEWER than `C` idx
/// rows is materialized as a base bitmap and maintained incrementally; commoner
/// trigrams fall to the live posting read. `C` is a pure PERFORMANCE dial, not a
/// recall one — both read paths return the same rowids (a materialized trigram has
/// `DF < C ≪ FUZZY_POSTING_CEILING`, so the live path's downsample never fires on it
/// either), so moving a trigram between tiers cannot change results. It trades
/// per-write delta cost (higher `C` materializes more trigrams → more delta
/// read-modify-writes per write) against query coverage (more trigrams served by the
/// `O(containers)` bitmap rather than a live FTS5 scan). The query-relevant trigrams
/// are the RAREST (the prune keeps `FuzzyCapPolicy::cap` of them), which sit far
/// below any sensible `C`, so a moderate ceiling captures them cheaply. Provisional;
/// the per-write-cost / coverage knee is tuned on the perf harness at 50k.
pub const MATERIALIZE_DF_CEILING: usize = 4096;

/// The per-query rare-trigram cap as a clamped log-growth curve in the query's OWN
/// trigram count `n`: `clamp(floor + round(k·ln(n/floor)), floor, ceiling)`.
///
/// The cap has two jobs and the curve serves both. It keeps the RAREST (most
/// discriminative) trigrams and drops the common ones — the perf win (a common
/// trigram's posting is huge) and a precision win. On a LONG query a fixed cap can
/// under-count overlap by ranking on too few of its rare trigrams; growing the cap
/// recovers that recall. But growth must stay a small FRACTION of `n` or a long
/// query re-admits the common trigrams the prune deliberately drops — so the
/// ceiling is an ABSOLUTE constant, not a fraction, and growth is sub-linear (log):
/// the kept fraction shrinks as the query grows.
///
/// The cap derives strictly from `n = grams.len()` for THIS query, never from a
/// batch-wide aggregate, so `search_fuzzy_batch([…,q,…]) == search_fuzzy(q)`: a
/// trigram's DF is a collection property, batch-independent, and the cap depends
/// only on `q`'s own gram count. A batch-wide cap (by the longest query) would
/// break that invariant and is deliberately not used.
///
/// The engine's default is fixed-6 ([`FuzzyCapPolicy::default`], `floor == ceiling
/// == 6`), so every `n` clamps to 6 and there is no behaviour change from the
/// historical `FUZZY_MAX_TRIGRAMS` truncate. The `(floor, k, ceiling)` are settable
/// ([`DerivedEngine::set_fuzzy_cap_policy`]) so the fuzzy-recall eval can A/B a
/// floor sweep and the log-growth curve against fixed-6 without recompiling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FuzzyCapPolicy {
    /// The floor (and, by default, also the ceiling) on the cap — today's fixed
    /// value. The `ln` term is `<= 0` for `n <= floor`, so the cap pins here for
    /// short queries.
    pub floor: usize,
    /// The log-growth rate of the cap in `n`. Larger `k` admits the next-rarest
    /// trigrams faster. Inert when `floor == ceiling` (the default).
    pub k: f64,
    /// The absolute ceiling on the cap — a constant, NOT a fraction of `n`, so a
    /// long query can never re-admit the common trigrams the prune drops.
    pub ceiling: usize,
}

impl Default for FuzzyCapPolicy {
    /// Fixed-6: `floor == ceiling == 6`, so [`Self::cap`] returns 6 for every `n`
    /// — byte-identical to the historical [`FUZZY_MAX_TRIGRAMS`] truncate. `k` is
    /// the documented default the eval sweeps up from; it is inert while the floor
    /// and ceiling coincide.
    fn default() -> Self {
        Self {
            floor: FUZZY_MAX_TRIGRAMS,
            k: 2.7,
            ceiling: FUZZY_MAX_TRIGRAMS,
        }
    }
}

impl FuzzyCapPolicy {
    /// The cap for a query of `n` trigrams: `clamp(floor + round(k·ln(n/floor)),
    /// floor, ceiling)`. For `n <= floor` the `ln` term is `<= 0` so the result
    /// pins at `floor`. A `ceiling < floor` mis-config degrades to fixed-at-floor:
    /// the upper bound is `max(floor, ceiling)`, so the cap never returns below the
    /// floor and `clamp` never sees an inverted range.
    #[must_use]
    pub fn cap(&self, n: usize) -> usize {
        if n <= self.floor {
            return self.floor;
        }
        let grown = self.floor as f64 + (self.k * ((n as f64) / self.floor as f64).ln()).round();
        // `grown >= floor` here (the `ln` term is positive for `n > floor`), so the
        // `as usize` cast cannot underflow. `max(floor, ceiling)` keeps the clamp
        // range valid even if a caller sets a ceiling below the floor.
        (grown as usize).clamp(self.floor, self.ceiling.max(self.floor))
    }
}

/// Ceiling on the rowids one trigram's posting contributes to the overlap basis.
/// `prune_to_rare_terms` keeps the rarest trigrams, but a query of all-common words
/// (every trigram is frequent) leaves the pruned set with O(collection)-length
/// postings, and the batch bounds the COUNT of queries (≤50), not their length — so a
/// pathological batch reads, accumulates, and hydrates a collection-scale rowid set
/// per term. Past this many rowids a posting is deterministically downsampled to it
/// ([`DerivedEngine::sample_posting`]).
///
/// Sized above the worst single-trigram posting at the heaviest standard scale (50k
/// notes, ~2 indexed text rows each → a top-frequency trigram appears in ≈10^5 rows)
/// so it never fires there; it only caps the 100k+ pathological case the bound exists
/// for. A power of two for a clean stride and to read as "deliberately large, not a
/// tuned recall cut".
pub const FUZZY_POSTING_CEILING: usize = 1 << 18;

// The integer-keyed fast-hash types live in `shrike-store` (the only crate the
// index, derived, and kernel impls all share, so the vector index can use the same
// hasher); re-exported so existing `shrike_derived::{FxI64Hasher, FxI64Map}` users
// keep resolving.
pub use shrike_store::{FxI64Hasher, FxI64Map, FxI64Set};

pub use shrike_store::LexicalRow;

/// A single trigram — exactly [`MIN_TRIGRAM`] (3) Unicode code points, so at most 12
/// UTF-8 bytes — stored INLINE with no heap allocation. The fuzzy path materializes
/// one per 3-char window of every indexed and queried string, so a `String` per
/// window was a hot allocation on both writes and queries; `Trigram` is a `Copy`
/// value that lives on the stack and keys the bitmap maps directly.
///
/// `Hash`/`Eq`/`Ord` delegate to the string slice, so they agree with `str` — which
/// is what makes the [`Borrow<str>`] impl sound and lets `HashMap<Trigram, _>` /
/// `BTreeSet<Trigram>` be probed with a plain `&str`.
#[derive(Clone, Copy)]
pub struct Trigram {
    /// UTF-8 bytes of the trigram, left-aligned; only `buf[..len]` is meaningful.
    buf: [u8; Self::MAX_LEN],
    len: u8,
}

impl Trigram {
    /// 3 code points × up to 4 UTF-8 bytes each.
    const MAX_LEN: usize = 4 * MIN_TRIGRAM;

    /// The trigram as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        // SAFETY: `buf[..len]` is only ever filled from valid UTF-8 — `from_chars`
        // encodes `char`s (always valid UTF-8), and `try_from_str` copies the bytes of
        // an existing `&str` after a length check — so the slice is always valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(&self.buf[..self.len as usize]) }
    }

    /// Build from exactly [`MIN_TRIGRAM`] code points, encoding them inline (no
    /// allocation). `MAX_LEN` holds 3 max-width (4-byte) code points exactly, so the
    /// encode always fits; a caller passing more than 3 would overflow the buffer slice
    /// and panic (a bug — the sole caller, `trigrams`, passes a 3-char window).
    #[inline]
    fn from_chars(chars: &[char]) -> Self {
        let mut buf = [0u8; Self::MAX_LEN];
        let mut len = 0usize;
        for &c in chars {
            len += c.encode_utf8(&mut buf[len..]).len();
        }
        Self {
            buf,
            len: len as u8,
        }
    }

    /// Build from a string slice that is one trigram (≤ [`MAX_LEN`] bytes). Returns
    /// `None` if it is longer — the bound the inline buffer guarantees.
    fn try_from_str(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() > Self::MAX_LEN {
            return None;
        }
        let mut buf = [0u8; Self::MAX_LEN];
        buf[..bytes.len()].copy_from_slice(bytes);
        Some(Self {
            buf,
            len: bytes.len() as u8,
        })
    }
}

impl std::ops::Deref for Trigram {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for Trigram {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::borrow::Borrow<str> for Trigram {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq for Trigram {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}
impl Eq for Trigram {}

impl std::hash::Hash for Trigram {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Delegate to the `str` hash so it matches a `&str` probe through `Borrow`.
        self.as_str().hash(state);
    }
}

impl PartialOrd for Trigram {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Trigram {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl std::fmt::Debug for Trigram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.as_str(), f)
    }
}
impl std::fmt::Display for Trigram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl rusqlite::ToSql for Trigram {
    fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
        Ok(rusqlite::types::ToSqlOutput::from(self.as_str()))
    }
}

impl rusqlite::types::FromSql for Trigram {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> rusqlite::types::FromSqlResult<Self> {
        let s = value.as_str()?;
        Trigram::try_from_str(s).ok_or(rusqlite::types::FromSqlError::InvalidType)
    }
}

/// Lowercased char-level trigrams (mirrors the Python `_trigrams`: code-point
/// windows over `text.lower()`), each an inline [`Trigram`] — no per-window
/// allocation.
pub fn trigrams(text: &str) -> Vec<Trigram> {
    let lowered: Vec<char> = text.to_lowercase().chars().collect();
    if lowered.len() < MIN_TRIGRAM {
        return Vec::new();
    }
    (0..=lowered.len() - MIN_TRIGRAM)
        .map(|i| Trigram::from_chars(&lowered[i..i + MIN_TRIGRAM]))
        .collect()
}

/// Quote a term as an FTS5 string literal (wrap in double quotes, double
/// internal ones) — the only safe way to feed arbitrary user text into MATCH.
pub fn fts_quote(term: &str) -> String {
    format!("\"{}\"", term.replace('"', "\"\""))
}

impl DerivedEngine {
    /// Set the per-query rare-trigram cap policy ([`FuzzyCapPolicy`]). The
    /// fuzzy-recall eval uses this to A/B a floor sweep and the log-growth curve
    /// against the fixed-6 default without recompiling. Production opens the engine
    /// with the default and never calls this.
    ///
    /// Applied from the NEXT fuzzy batch on (each batch reads the policy once), so
    /// it never splits a batch in flight. The cap is still derived per query from
    /// that query's own gram count, so batch==serial is preserved under any policy.
    pub fn set_fuzzy_cap_policy(&self, policy: FuzzyCapPolicy) {
        *self.fuzzy_cap_lock() = policy;
    }

    /// The current per-query rare-trigram cap policy.
    pub fn fuzzy_cap_policy(&self) -> FuzzyCapPolicy {
        *self.fuzzy_cap_lock()
    }

    /// Set the materialization DF ceiling `C` ([`MATERIALIZE_DF_CEILING`]). Takes
    /// effect at the NEXT build/fold (the per-write delta path reads materialization
    /// from the base table, so a `C` change is fully applied once the fold has
    /// re-tiered the trigrams). The perf harness sweeps this to find the per-write /
    /// coverage knee; production opens with the default and never calls it.
    pub fn set_materialize_ceiling(&self, ceiling: usize) {
        *self.materialize_ceiling_lock() = ceiling;
    }

    /// The current materialization DF ceiling `C`.
    pub fn materialize_ceiling(&self) -> usize {
        *self.materialize_ceiling_lock()
    }

    /// One query's trigram set, `None` when it has fewer than [`FUZZY_MIN_SHARED`]
    /// trigram windows (too short to rank). NFC-normalized so the trigrams match
    /// the NFC-normalized index. [`Self::prune_to_rare_terms`] derives the (smaller)
    /// rare-trigram set the overlap ranker actually queries and counts.
    fn fuzzy_grams(query: &str) -> Option<std::collections::BTreeSet<Trigram>> {
        let normalized = nfc(query);
        let grams = trigrams(normalized.trim());
        if grams.len() < FUZZY_MIN_SHARED {
            return None;
        }
        Some(grams.into_iter().collect())
    }

    /// The fuzzy candidate trigrams the overlap ranker queries and counts: the
    /// `policy.cap(grams.len())` rarest KNOWN (present in the DF snapshot) trigrams,
    /// PLUS every trigram ABSENT from the snapshot. Common trigrams bloat the match
    /// set and discriminate least, so the rarest known ones are kept; but
    /// absent-from-the-snapshot does NOT mean globally rare. `df` is read from the
    /// materialized `trigram_df`, which lags the live index between rebuilds, so an
    /// absent trigram can mean "written since the snapshot" — and truncating those
    /// away would drop a fuzzy match whose overlap is dominated by just-written
    /// trigrams. So absent trigrams are KEPT, not pruned:
    /// on a FRESH snapshot they are genuine typos with empty postings (recall-safe,
    /// results-neutral, near-free to scan), and on a STALE one they are exactly the
    /// new trigrams the prune must scan to keep recall. The floor is over this kept
    /// set, not the full query gram set.
    fn prune_to_rare_terms(
        grams: &std::collections::BTreeSet<Trigram>,
        df: &std::collections::HashMap<Trigram, i64>,
        policy: FuzzyCapPolicy,
    ) -> Vec<Trigram> {
        // The cap is derived from THIS query's own gram count (`grams.len()`), never
        // a batch-wide count, so the pruned set is byte-identical whether the query
        // runs alone or in a batch — the batch==serial invariant.
        let cap = policy.cap(grams.len());
        // The `cap` rarest PRESENT trigrams (DF ascending, term as a deterministic
        // tie-break)...
        let mut present: Vec<&Trigram> = grams
            .iter()
            .filter(|g| df.get(*g).copied().unwrap_or(0) > 0)
            .collect();
        present.sort_by(|a, b| {
            let d = |g: &Trigram| df.get(g).copied().unwrap_or(0);
            d(a).cmp(&d(b)).then_with(|| a.cmp(b))
        });
        present.truncate(cap);
        // ...then EVERY absent (DF-0) trigram appended. `grams` is a BTreeSet, so the
        // absent filter yields them in deterministic term order.
        let absent = grams
            .iter()
            .filter(|g| df.get(*g).copied().unwrap_or(0) == 0);
        present.into_iter().chain(absent).copied().collect()
    }

    /// Accumulate one query's per-segment trigram overlap: how many of its pruned
    /// (rare) trigrams each indexed rowid shares, from the per-term posting sets
    /// gathered rowid-only by [`Self::fuzzy_term_rowids`]. Provenance-free, so it
    /// runs before the survivors' `(note_id, source, ref)` is known.
    #[allow(dead_code)] // reference for the bit-sliced fuzzy_rank_query cross-check + revert.
    fn accumulate_overlap(
        pruned_terms: &[Trigram],
        term_bitmaps: &std::collections::HashMap<Trigram, roaring::RoaringBitmap>,
    ) -> FxI64Map<usize> {
        // Preallocate to the sum of the rare trigrams' posting lengths — the upper
        // bound on distinct rowids (a rowid shared across trigrams just over-counts
        // the bound). Building from empty otherwise grows-and-rehashes the table
        // repeatedly (the `reserve_rehash` frame in the profile); one sized
        // allocation skips all of it. The over-allocation is transient — the map is
        // dropped at the end of the query.
        let cap: usize = pruned_terms
            .iter()
            .filter_map(|t| term_bitmaps.get(t))
            .map(|b| b.len() as usize)
            .sum();
        let mut overlap: FxI64Map<usize> =
            std::collections::HashMap::with_capacity_and_hasher(cap, Default::default());
        for term in pruned_terms {
            if let Some(bm) = term_bitmaps.get(term) {
                for rid in bm.iter() {
                    *overlap.entry(rid as i64).or_insert(0) += 1;
                }
            }
        }
        overlap
    }

    /// Load the pruned trigrams' precomputed posting bitmaps from
    /// `trigram_bitmap` in ONE query (inline `IN` over the ≤cap distinct terms),
    /// keyed by term. A term with no row (never indexed) is simply absent from the
    /// map — the overlap accumulation skips it, exactly as an empty FTS5 posting
    /// would.
    ///
    /// # Errors
    ///
    /// Returns an error if the read or a bitmap deserialize fails.
    fn load_trigram_bitmaps(
        conn: &Connection,
        terms: &[Trigram],
    ) -> NativeResult<std::collections::HashMap<Trigram, roaring::RoaringBitmap>> {
        let mut out = std::collections::HashMap::new();
        if terms.is_empty() {
            return Ok(out);
        }
        let placeholders = (0..terms.len())
            .map(|i| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!("SELECT term, bm FROM trigram_bitmap WHERE term IN ({placeholders})");
        let run = || -> rusqlite::Result<Vec<(Trigram, Vec<u8>)>> {
            let mut stmt = conn.prepare_cached(&sql)?;
            let mut q = stmt.query(rusqlite::params_from_iter(terms.iter()))?;
            let mut rows = Vec::new();
            while let Some(r) = q.next()? {
                rows.push((r.get(0)?, r.get(1)?));
            }
            Ok(rows)
        };
        for (term, blob) in with_busy_retry(run)? {
            let bm = roaring::RoaringBitmap::deserialize_from(&blob[..])
                .map_err(|e| NativeError::internal(e.to_string()))?;
            out.insert(term, bm);
        }
        Ok(out)
    }

    /// Each pruned trigram's EFFECTIVE posting bitmap for the fuzzy ranker. A
    /// MATERIALIZED trigram (one with a base row) resolves to its always-fresh
    /// `(base ∪ added) \ removed` from `trigram_bitmap` + `trigram_delta`; the delta
    /// is written in the same transaction as the `idx` rows, so no global freshness
    /// gate is needed. An UNMATERIALIZED trigram (common, or a store with no base tier
    /// yet) falls to the live FTS5 posting read ([`Self::fuzzy_term_rowids`],
    /// downsampled past the ceiling). Keyed by term; a trigram with no posting at all
    /// is simply absent (the overlap accumulation skips it).
    ///
    /// # Errors
    ///
    /// Returns an error if a read or a bitmap deserialize fails.
    fn effective_term_bitmaps(
        conn: &Connection,
        terms: &[Trigram],
    ) -> NativeResult<std::collections::HashMap<Trigram, roaring::RoaringBitmap>> {
        use roaring::RoaringBitmap;
        // load_trigram_bitmaps returns a row ONLY for a materialized term, so its keys
        // ARE the materialized subset of `terms`.
        let mut out = Self::load_trigram_bitmaps(conn, terms)?;
        // Fold each materialized term's pending delta into its base.
        if !out.is_empty() {
            let materialized: Vec<Trigram> = out.keys().copied().collect();
            let want = Self::term_array(materialized.iter().map(Trigram::as_str));
            let run = || -> rusqlite::Result<Vec<DeltaBlobRow>> {
                let mut stmt = conn.prepare_cached(
                    "SELECT term, added, removed FROM trigram_delta WHERE term IN rarray(?1)",
                )?;
                let mut q = stmt.query(rusqlite::params![want])?;
                let mut rows = Vec::new();
                while let Some(r) = q.next()? {
                    rows.push((r.get(0)?, r.get(1)?, r.get(2)?));
                }
                Ok(rows)
            };
            for (term, added_blob, removed_blob) in with_busy_retry(run)? {
                if let Some(bm) = out.get_mut(&term) {
                    let added = RoaringBitmap::deserialize_from(&added_blob[..])
                        .map_err(|e| NativeError::internal(e.to_string()))?;
                    let removed = RoaringBitmap::deserialize_from(&removed_blob[..])
                        .map_err(|e| NativeError::internal(e.to_string()))?;
                    *bm |= &added;
                    *bm -= &removed;
                }
            }
        }
        // The unmaterialized terms fall to the live posting read.
        let live_terms: Vec<Trigram> = terms
            .iter()
            .copied()
            .filter(|t| !out.contains_key(t))
            .collect();
        if !live_terms.is_empty() {
            for (term, rids) in Self::fuzzy_term_rowids(conn, &live_terms)? {
                out.insert(term, rids.into_iter().map(|r| r as u32).collect());
            }
        }
        Ok(out)
    }

    /// Bit-sliced overlap counting. Returns the count "bit planes" `acc` where the
    /// overlap count of a rowid is `Σ_b 2^b · [rid ∈ acc[b]]`. Adds each input
    /// bitmap with a ripple-carry across the planes (XOR = sum bit, AND = carry), so
    /// the per-rowid overlap for EVERY rowid is computed in `O(k·log k)` bitmap ops
    /// — `O(containers)`, independent of posting size. That is the dense-posting win
    /// the per-rowid accumulation loop can't get: a trigram in tens of thousands of
    /// notes is one bitmap container, so adding it costs a few SIMD word ops, not
    /// one increment per rowid.
    fn bitsliced_overlap(bitmaps: &[&roaring::RoaringBitmap]) -> Vec<roaring::RoaringBitmap> {
        use roaring::RoaringBitmap;
        let mut acc: Vec<RoaringBitmap> = Vec::new();
        for &b in bitmaps {
            let mut carry = b.clone();
            let mut level = 0usize;
            while !carry.is_empty() {
                if level == acc.len() {
                    acc.push(RoaringBitmap::new());
                }
                let new_carry = &acc[level] & &carry; // carry-out = already-set AND incoming
                acc[level] ^= &carry; // sum bit at this plane
                carry = new_carry;
                level += 1;
            }
        }
        acc
    }

    /// The rowids whose bit-sliced overlap count is EXACTLY `c`: AND the planes `c`
    /// has set, then subtract every plane it has clear — a rowid survives iff its
    /// plane membership is exactly `c`'s bit pattern. `c == 0` selects nothing.
    fn count_eq(acc: &[roaring::RoaringBitmap], c: u32) -> roaring::RoaringBitmap {
        use roaring::RoaringBitmap;
        // A count whose highest set bit is beyond the planes can't exist (the loop
        // walks c up to the term count, which may exceed the realized max overlap).
        if 32 - c.leading_zeros() > acc.len() as u32 {
            return RoaringBitmap::new();
        }
        let set: Vec<usize> = (0..acc.len()).filter(|&b| (c >> b) & 1 == 1).collect();
        let Some((&first, rest)) = set.split_first() else {
            return RoaringBitmap::new();
        };
        let mut out = acc[first].clone();
        for &b in rest {
            out &= &acc[b];
        }
        for (b, plane) in acc.iter().enumerate() {
            if (c >> b) & 1 == 0 {
                out -= plane;
            }
        }
        out
    }

    /// One query's fuzzy ranking via the bit-sliced overlap planes, hydrating only
    /// as deep as the threshold top-k needs. Walks the overlap-count buckets high →
    /// low; each bucket hydrates its rowids (the same source/scope filter as
    /// [`Self::seg_meta_for_rowids`]) and records the best segment per note (highest
    /// count, lowest rowid). Processing high → low means a note's FIRST appearance
    /// is its highest-count (best) segment, and ascending rowid order within a
    /// bucket makes the first the lowest-rowid tiebreak — exactly
    /// [`Self::rank_overlap`]'s `best`. Once `top_k` notes are locked at a count, no
    /// lower bucket (a strictly smaller count) can enter the top-k, so it stops —
    /// bounding the hydration to the high-overlap head instead of every
    /// `count >= FUZZY_MIN_SHARED` candidate. The result is identical to
    /// `accumulate_overlap` + `rank_overlap` (pinned by a cross-check test); a
    /// fetch-all (`top_k` past the candidate count) never trips the stop and walks
    /// every bucket.
    fn fuzzy_rank_query(
        conn: &Connection,
        pruned_terms: &[Trigram],
        term_bitmaps: &std::collections::HashMap<Trigram, roaring::RoaringBitmap>,
        top_k: usize,
        exclude_sources: &[&str],
        scope: Option<&[i64]>,
    ) -> NativeResult<Vec<(i64, String, String, i64)>> {
        // top_k == 0 selects nothing — match rank_overlap's truncate-to-empty (a
        // huge fetch-all top_k, not 0, is the "return all" sentinel).
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let bitmaps: Vec<&roaring::RoaringBitmap> = pruned_terms
            .iter()
            .filter_map(|t| term_bitmaps.get(t))
            .collect();
        // Fewer present postings than the floor can never reach the overlap minimum.
        if bitmaps.len() < FUZZY_MIN_SHARED {
            return Ok(Vec::new());
        }
        let acc = Self::bitsliced_overlap(&bitmaps);
        let max_count = bitmaps.len() as u32;
        // note_id -> (count, rowid, source, ref) for its best segment.
        let mut best: FxI64Map<(usize, i64, String, String)> = FxI64Map::default();
        for c in (FUZZY_MIN_SHARED as u32..=max_count).rev() {
            let bucket = Self::count_eq(&acc, c);
            if !bucket.is_empty() {
                let rids: Vec<i64> = bucket.iter().map(i64::from).collect();
                let meta = Self::seg_meta_for_rowids(conn, &rids, exclude_sources, scope)?;
                // `rids` is ascending (bitmap iteration order), so the first segment
                // recorded for a note is its lowest rowid at this — its best — count.
                for &rid in &rids {
                    if let Some((nid, source, r)) = meta.get(&rid) {
                        best.entry(*nid)
                            .or_insert_with(|| (c as usize, rid, source.clone(), r.clone()));
                    }
                }
            }
            // Every note in `best` now has count >= c; the next bucket is c-1 < c, so
            // it can never displace a locked top-k. A fetch-all top_k never trips this.
            if top_k > 0 && best.len() >= top_k {
                break;
            }
        }
        let mut ranked: Vec<(i64, usize, i64, String, String)> = best
            .into_iter()
            .map(|(nid, (count, rid, source, r))| (nid, count, rid, source, r))
            .collect();
        // count desc, then note-id asc — the rank_overlap order.
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if top_k > 0 {
            ranked.truncate(top_k);
        }
        Ok(ranked
            .into_iter()
            .map(|(nid, _c, rid, source, r)| (nid, source, r, rid))
            .collect())
    }

    /// Rank one query's accumulated overlap into its survivors. A segment's overlap
    /// is how many pruned trigrams matched it; a note's overlap is its best
    /// segment's. Keep notes sharing at least [`FUZZY_MIN_SHARED`] pruned trigrams,
    /// one (best-overlap, lowest-rowid) segment per note, ordered overlap-desc then
    /// note-id-asc, capped at `top_k`. Returns `(note_id, source, ref, rowid)`; the
    /// rowid drives the deferred snippet read. Recall-safe: the cut is by overlap,
    /// never by rowid — every matched segment is ranked, unlike a bm25 `LIMIT` over
    /// the OR. A rowid present in `overlap` but ABSENT from `seg_meta` is dropped:
    /// [`Self::seg_meta_for_rowids`] omits the segments filtered out by
    /// `exclude_sources` or an out-of-scope `note_id`, so "absent" means "filtered
    /// out," and skipping it makes a note rank by its best surviving segment — the
    /// same result a `source NOT IN` / `note_id IN` MATCH predicate would give.
    #[allow(dead_code)] // reference for the bit-sliced fuzzy_rank_query cross-check + revert.
    fn rank_overlap(
        overlap: &FxI64Map<usize>,
        seg_meta: &FxI64Map<(i64, String, String)>,
        top_k: usize,
    ) -> Vec<(i64, String, String, i64)> {
        // Best segment per note: highest overlap, lowest rowid as the deterministic
        // tie-break (so the chosen snippet segment is stable run to run). Sized to
        // `overlap` (the upper bound on distinct notes) so it never rehashes.
        let mut best: FxI64Map<(usize, i64)> =
            std::collections::HashMap::with_capacity_and_hasher(overlap.len(), Default::default());
        for (&rid, &count) in overlap {
            if count < FUZZY_MIN_SHARED {
                continue;
            }
            let nid = match seg_meta.get(&rid) {
                Some((nid, _, _)) => *nid,
                None => continue, // source-excluded segment — not a candidate
            };
            let better = match best.get(&nid) {
                None => true,
                Some(&(bc, br)) => count > bc || (count == bc && rid < br),
            };
            if better {
                best.insert(nid, (count, rid));
            }
        }
        let mut ranked: Vec<(i64, usize, i64)> = best
            .into_iter()
            .map(|(nid, (c, rid))| (nid, c, rid))
            .collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(top_k);
        ranked
            .into_iter()
            .map(|(nid, _, rid)| {
                let (_, source, r) = &seg_meta[&rid];
                (nid, source.clone(), r.clone(), rid)
            })
            .collect()
    }

    /// Notes whose derived text literally contains `query` (case-insensitive),
    /// with `(source, ref, snippet)` provenance. `None` tells the caller to use
    /// the `find_notes` fallback (query shorter than a trigram); a MATCH error
    /// is a real error (the caller decides whether to degrade).
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn search_substring(
        &self,
        query: &str,
        limit: i64,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Option<Vec<LexicalRow>>> {
        // NFC-normalize so the phrase matches the NFC-normalized index.
        let normalized = nfc(query);
        let q = normalized.trim();
        if q.chars().count() < MIN_TRIGRAM {
            return Ok(None);
        }
        // A quoted phrase → contiguous (literal substring) match.
        let rows = self.match_rows(&fts_quote(q), limit, false, scope, exclude_sources)?;
        Ok(Some(
            rows.into_iter()
                .map(|(nid, source, r, _txt, snippet)| (nid, source, r, snippet))
                .collect(),
        ))
    }

    /// Notes sharing trigrams with `query` (typo/partial tolerant), ranked by how
    /// many of the query's rarest trigrams they share, deduped to one (best) row
    /// per note, requiring at least [`FUZZY_MIN_SHARED`] shared rare trigrams
    /// (drops single-trigram noise). Empty when the query is too short to rank.
    /// Delegates to [`Self::search_fuzzy_batch`] so the singular and batched paths
    /// share one ranking implementation.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the operation.
    pub fn search_fuzzy(
        &self,
        query: &str,
        top_k: i64,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Vec<LexicalRow>> {
        Ok(self
            .search_fuzzy_batch(&[query], top_k, scope, exclude_sources)?
            .into_iter()
            .next()
            .unwrap_or_default())
    }

    /// [`Self::search_substring`] over a batch of queries — one result per query
    /// in `queries` order, sharing one connection lock, one scope staging, and
    /// one compiled statement across the set. A sub-trigram query resolves to
    /// `None` (the caller's `find_notes` fallback) without reaching FTS5, exactly
    /// like the singular call.
    ///
    /// # Errors
    ///
    /// Returns an error if the batched MATCH query fails.
    pub fn search_substring_batch(
        &self,
        queries: &[&str],
        limit: i64,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Vec<Option<Vec<LexicalRow>>>> {
        // Servable queries (>= a trigram) contribute an expr + an FTS5 slot;
        // sub-trigram queries resolve to None without a query. `served[i]`
        // records which, so the batched rows reattach to the right queries.
        let mut exprs: Vec<String> = Vec::new();
        let mut served: Vec<bool> = Vec::with_capacity(queries.len());
        for q in queries {
            // NFC-normalize so the phrase matches the NFC-normalized index.
            let normalized = nfc(q);
            let trimmed = normalized.trim();
            if trimmed.chars().count() < MIN_TRIGRAM {
                served.push(false);
            } else {
                served.push(true);
                exprs.push(fts_quote(trimmed));
            }
        }
        // One pool connection for the whole batch's single MATCH read.
        let conn = self.read_pool.checkout()?;
        let mut batched =
            Self::match_rows_batch(&conn, &exprs, limit, false, scope, exclude_sources)?
                .into_iter();
        // `batched` yields one entry per served query, in order; reattach each to
        // its query. A served query maps to `Some(rows)` (possibly empty — the
        // store served it and found nothing); a sub-trigram query maps to `None`
        // (the caller's find_notes fallback). `map` over `next()` keeps this
        // panic-free even if the lengths ever disagree.
        let out = served
            .into_iter()
            .map(|s| {
                if s {
                    batched.next().map(|rows| {
                        rows.into_iter()
                            .map(|(nid, source, r, _txt, snippet)| (nid, source, r, snippet))
                            .collect()
                    })
                } else {
                    None
                }
            })
            .collect();
        Ok(out)
    }

    /// [`Self::search_fuzzy`] over a batch of queries — one result per query in
    /// `queries` order — the fuzzy counterpart to [`Self::search_substring_batch`].
    /// A query too short to rank yields an empty result without reaching FTS5.
    ///
    /// Ranks by trigram OVERLAP without bm25: each query's rarest trigrams are read
    /// as individual posting lists (one MATCH per DISTINCT trigram — a trigram's
    /// posting is identical for every query in the batch), rowid-ONLY
    /// ([`Self::fuzzy_term_rowids`]), then accumulated per query into per-note
    /// overlap. Provenance is hydrated once for the overlap candidates
    /// ([`Self::seg_meta_for_rowids`]), which also applies BOTH filters in Rust off
    /// the parallel hot path: `exclude_sources` and the deck/tag SCOPE drop their
    /// segments there, not as a `note_id IN (...)` MATCH predicate (which would make
    /// SQLite build an ephemeral index — a temp btree on the pcache mutex). bm25's
    /// `ORDER BY rank` over the pruned `OR` was the search hotspot; raw posting reads
    /// skip it and the overlap cut is recall-safe (by overlap, never rowid). Snippets
    /// are read once for the surviving top-k ([`Self::fuzzy_snippets_batch`]), not for
    /// every match.
    ///
    /// # Errors
    ///
    /// Returns an error if the batched MATCH query fails.
    pub fn search_fuzzy_batch(
        &self,
        queries: &[&str],
        top_k: i64,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Vec<Vec<LexicalRow>>> {
        let gram_sets: Vec<Option<std::collections::BTreeSet<Trigram>>> =
            queries.iter().map(|q| Self::fuzzy_grams(q)).collect();
        // One batched DF lookup over every distinct trigram, then prune each served
        // query to its rarest (most discriminative) trigrams.
        let distinct: std::collections::BTreeSet<Trigram> = gram_sets
            .iter()
            .flatten()
            .flat_map(|g| g.iter().copied())
            .collect();
        // One pool connection for the whole fuzzy read: the DF lookup, the posting
        // reads, and the snippet reads all run on it. Each sub-read is its own
        // statement (its own committed snapshot) — the same per-sub-read freshness
        // as before; the freshness bracket flags a write that lands mid-read.
        let conn = self.read_pool.checkout()?;
        let df = Self::trigram_dfs(&conn, &distinct.into_iter().collect::<Vec<Trigram>>())?;
        // Read the cap policy ONCE for the whole batch, so every query in this batch
        // is pruned under one snapshot of the policy (a concurrent set never splits a
        // batch). The cap is still derived per query from that query's own gram
        // count inside the map — never batch-wide — so batch==serial holds.
        let policy = self.fuzzy_cap_policy();
        let pruned: Vec<Option<Vec<Trigram>>> = gram_sets
            .iter()
            .map(|g| {
                g.as_ref()
                    .map(|gs| Self::prune_to_rare_terms(gs, &df, policy))
            })
            .collect();
        // Read each DISTINCT pruned trigram's posting once (shared across queries).
        let distinct_terms: std::collections::BTreeSet<Trigram> =
            pruned.iter().flatten().flatten().copied().collect();
        if distinct_terms.is_empty() {
            return Ok(queries.iter().map(|_| Vec::new()).collect());
        }
        let distinct_terms_vec: Vec<Trigram> = distinct_terms.into_iter().collect();
        // Read each pruned trigram's posting rowid-ONLY (no per-posting JOIN, no
        // string allocs over the full lists), accumulate per-query overlap, then
        // hydrate `(note_id, source, ref)` once for the candidate set (overlap >= the
        // floor). Both the `exclude_sources` hidden-source list and the deck/tag
        // SCOPE drop their segments in that one batched hydration, in Rust — off the
        // parallel hot path, where a `note_id IN scope` MATCH predicate would make
        // SQLite build an ephemeral index over the set (a temp btree on the pcache
        // mutex). rank_overlap skips a candidate absent from `seg_meta`, so a dropped
        // segment never ranks.
        // The pruned trigrams' posting bitmaps, PER TERM: a materialized (rare)
        // trigram resolves to its always-fresh `(base ∪ added) \ removed`; an
        // unmaterialized (common, or never-built) one falls to the live FTS5 posting
        // read. Both feed the same bit-sliced ranker.
        let term_bitmaps = Self::effective_term_bitmaps(&conn, &distinct_terms_vec)?;
        let survivors: Vec<Vec<(i64, String, String, i64)>> = pruned
            .iter()
            .map(|p| match p {
                Some(terms) => Self::fuzzy_rank_query(
                    &conn,
                    terms,
                    &term_bitmaps,
                    top_k as usize,
                    exclude_sources,
                    scope,
                ),
                None => Ok(Vec::new()),
            })
            .collect::<NativeResult<_>>()?;
        // Build snippets for the surviving rowids only, from text read by a plain
        // rowid lookup (no MATCH re-scan) and windowed in Rust around the query's
        // own rare trigrams.
        let snippet_jobs: Vec<(&[Trigram], Vec<i64>)> = pruned
            .iter()
            .zip(&survivors)
            .map(|(p, surv)| {
                let terms: &[Trigram] = match p {
                    Some(t) if !surv.is_empty() => t.as_slice(),
                    _ => &[],
                };
                (terms, surv.iter().map(|(_, _, _, rid)| *rid).collect())
            })
            .collect();
        let snippets = Self::fuzzy_snippets_batch(&conn, &snippet_jobs)?;
        let out = survivors
            .into_iter()
            .zip(snippets)
            .map(|(surv, snip)| {
                surv.into_iter()
                    .map(|(nid, source, r, rid)| (nid, source, r, snip.get(&rid).cloned()))
                    .collect()
            })
            .collect();
        Ok(out)
    }

    /// Bound one trigram's posting to [`FUZZY_POSTING_CEILING`] rowids, returning it
    /// unchanged when it already fits. Past the ceiling, keep an EVENLY-SPACED sample
    /// across the ascending posting (indices `i * len / ceiling`), not a prefix: the
    /// sample spans the whole rowid range, so a capped common-trigram posting still
    /// contributes overlap from low- AND high-rowid notes rather than dropping the
    /// newest (highest-rowid) ones a `LIMIT` would.
    ///
    /// Deterministic — same posting in, same sample out — and a pure function of THIS
    /// term's own rowids, so the batched and singular paths sample identically (the
    /// `batch_lexical_matches_loop_of_singular` parity invariant). `i = 0` keeps the
    /// first (lowest) rowid; the stride reaches into the final stride-window, so the
    /// high-rowid tail is sampled, never truncated away.
    fn sample_posting(rids: Vec<i64>) -> Vec<i64> {
        let len = rids.len();
        if len <= FUZZY_POSTING_CEILING {
            return rids;
        }
        (0..FUZZY_POSTING_CEILING)
            .map(|i| rids[i * len / FUZZY_POSTING_CEILING])
            .collect()
    }

    /// Each term's matching idx rowids, rowid-ONLY — no `rowmap` JOIN, no
    /// provenance. FTS5 yields rowids straight off the posting list with no table
    /// access, the cheapest posting read; provenance is deferred to the overlap
    /// candidates ([`Self::seg_meta_for_rowids`]), which also drops the
    /// `exclude_sources` and out-of-scope segments — so the posting scan needs
    /// neither `note_id` nor `source` and stays rowid-only for every fuzzy read,
    /// scoped or not. Reads on the passed [`ReadPool`] connection.
    ///
    /// No SQL `LIMIT`: a `LIMIT` cuts the posting positionally (FTS5 yields rowids
    /// ascending, so it drops the highest-rowid — newest — notes), a measured recall
    /// gap. Instead a posting past [`FUZZY_POSTING_CEILING`] is deterministically
    /// downsampled across its whole rowid range ([`Self::sample_posting`]) so an
    /// all-common-word query's collection-scale postings stay bounded while a capped
    /// term still contributes a representative overlap signal. The ceiling sits above
    /// the worst posting at every standard scale, so it never fires there.
    ///
    /// # Errors
    ///
    /// Returns an error if any term's MATCH query fails.
    fn fuzzy_term_rowids(
        conn: &Connection,
        terms: &[Trigram],
    ) -> NativeResult<std::collections::HashMap<Trigram, Vec<i64>>> {
        let mut term_rowids: std::collections::HashMap<Trigram, Vec<i64>> =
            std::collections::HashMap::new();
        if terms.is_empty() {
            return Ok(term_rowids);
        }
        let span = tracing::debug_span!("derived.fuzzy_terms", n = terms.len());
        let _enter = span.enter();
        for term in terms {
            let quoted = fts_quote(term);
            let run = || -> rusqlite::Result<Result<Vec<i64>, NativeError>> {
                let mut stmt = conn.prepare_cached("SELECT rowid FROM idx WHERE idx MATCH ?1")?;
                let mut q = stmt.query([&quoted])?;
                let mut rids: Vec<i64> = Vec::new();
                loop {
                    let row = match q.next() {
                        Ok(Some(r)) => r,
                        Ok(None) => break,
                        Err(e) if is_retryable(&e) => return Err(e),
                        Err(e) => {
                            return Ok(Err(NativeError::invalid_input(format!("fts5 match: {e}"))))
                        }
                    };
                    match row.get(0) {
                        Ok(rid) => rids.push(rid),
                        Err(e) if is_retryable(&e) => return Err(e),
                        Err(e) => {
                            return Ok(Err(NativeError::invalid_input(format!("fts5 match: {e}"))))
                        }
                    }
                }
                Ok(Ok(rids))
            };
            let rids = Self::sample_posting(with_busy_retry(run)??);
            term_rowids.insert(*term, rids);
        }
        Ok(term_rowids)
    }

    /// Provenance `(note_id, source, ref)` per idx rowid, for a set of rowids,
    /// dropping any whose `source` is in `exclude_sources` or whose `note_id` is
    /// outside `scope` (when a deck/tag scope is given). Hydrates the overlap
    /// candidates the posting scan deferred (it read postings rowid-only, every
    /// source and note included), applying BOTH filters HERE rather than in the
    /// per-posting scan. A dropped rowid is then absent from the returned map, so
    /// [`Self::rank_overlap`] skips it — the same effect a `source NOT IN` /
    /// `note_id IN` MATCH predicate would have, off the hot scan.
    ///
    /// Each rowid is read by a PREPARE-CACHED single-row PK seek (`rowid = ?1`), and
    /// the scope is dropped in Rust, NOT as a `note_id IN (...)` predicate. Both keep
    /// this read on mmap with a statement parsed ONCE: a temp table's btree pages —
    /// or the ephemeral RHS index SQLite builds for an `IN (…)` set (inline literals
    /// re-parse the statement every read; bound `?` placeholders rebuild the
    /// ephemeral) — bypass mmap (it maps only the main DB file), so they fault
    /// through `pcache1`'s `STATIC_LRU` mutex and serialize the parallel fuzzy
    /// chunks. `rowmap` lives in the mmap'd main file, so a cached `rowid = ?1` seek
    /// (rowid is its primary key) is a direct lookup with no shared-cache mutex and
    /// no per-read parse. Reads on the passed [`ReadPool`] connection.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn seg_meta_for_rowids(
        conn: &Connection,
        rowids: &[i64],
        exclude_sources: &[&str],
        scope: Option<&[i64]>,
    ) -> NativeResult<FxI64Map<(i64, String, String)>> {
        let scope_set: Option<FxI64Set> = scope.map(|ids| ids.iter().copied().collect());
        // Sized to the candidate set (minus any source-excluded or out-of-scope rows)
        // so the per-candidate inserts never grow-and-rehash the table.
        let mut seg_meta: FxI64Map<(i64, String, String)> =
            std::collections::HashMap::with_capacity_and_hasher(rowids.len(), Default::default());
        // Bind the whole candidate set to ONE prepare-cached `IN rarray(?1)`
        // statement — the `rarray` carray vtab is an in-memory view over `ids` — so
        // the read is a single statement (one `query`/`reset`) over one implicit
        // transaction, versus a `query`/`reset` per candidate rowid. The statement
        // text is constant, so it parses once (no per-arity re-parse an inline
        // `IN (literals)` would pay), and `rarray` is not a temp DB table, so it
        // stages no pages through the pcache `STATIC_LRU` mutex the parallel fuzzy
        // chunks otherwise serialize on.
        let run = || -> rusqlite::Result<Vec<(i64, i64, String, String)>> {
            let ids: std::rc::Rc<Vec<rusqlite::types::Value>> = std::rc::Rc::new(
                rowids
                    .iter()
                    .map(|&r| rusqlite::types::Value::Integer(r))
                    .collect(),
            );
            let mut stmt = conn.prepare_cached(
                "SELECT rowid, note_id, source, ref FROM rowmap WHERE rowid IN rarray(?1)",
            )?;
            let mut out = Vec::with_capacity(rowids.len());
            let mut q = stmt.query(rusqlite::params![ids])?;
            while let Some(r) = q.next()? {
                out.push((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?));
            }
            Ok(out)
        };
        for (rowid, note_id, source, r) in with_busy_retry(run)? {
            if exclude_sources.contains(&source.as_str()) {
                continue;
            }
            if let Some(set) = &scope_set {
                if !set.contains(&note_id) {
                    continue;
                }
            }
            seg_meta.insert(rowid, (note_id, source, r));
        }
        Ok(seg_meta)
    }

    /// One snippet per surviving rowid, per query. `jobs[i]` is `(pruned_terms,
    /// rowids)` for query `i`: the rare trigrams it ranked on and its surviving idx
    /// rowids. An empty job (unservable query, or no survivors) yields an empty map.
    ///
    /// Reads each survivor's text by a plain rowid lookup — NOT a `MATCH` — and
    /// windows it in Rust ([`Self::window_snippet`]). The `snippet()` builtin needs
    /// a `MATCH`, and re-running the OR over the index to snippet a handful of rows
    /// re-pays the posting scan the overlap path exists to avoid; a rowid lookup
    /// reads only the survivor pages. Snippets are off the hot path: reading text
    /// for every candidate is what made bm25 expensive. Reads on the passed
    /// [`ReadPool`] connection.
    ///
    /// # Errors
    ///
    /// Returns an error if a text lookup fails.
    fn fuzzy_snippets_batch(
        conn: &Connection,
        jobs: &[(&[Trigram], Vec<i64>)],
    ) -> NativeResult<Vec<std::collections::HashMap<i64, String>>> {
        let mut out: Vec<std::collections::HashMap<i64, String>> = Vec::with_capacity(jobs.len());
        if jobs.iter().all(|(_, rids)| rids.is_empty()) {
            return Ok(jobs
                .iter()
                .map(|_| std::collections::HashMap::new())
                .collect());
        }
        for (terms, rowids) in jobs {
            if rowids.is_empty() || terms.is_empty() {
                out.push(std::collections::HashMap::new());
                continue;
            }
            // Survivor rowids inline as literals (i64 — no injection surface); the
            // set is at most top_k. A plain rowid lookup, no MATCH.
            let csv = rowids
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!("SELECT idx.rowid, idx.txt FROM idx WHERE idx.rowid IN ({csv})");
            let run = || -> rusqlite::Result<Result<std::collections::HashMap<i64, String>, NativeError>> {
                let mut stmt = conn.prepare_cached(&sql)?;
                let mut q = stmt.query([])?;
                let mut m = std::collections::HashMap::new();
                loop {
                    let row = match q.next() {
                        Ok(Some(r)) => r,
                        Ok(None) => break,
                        Err(e) if is_retryable(&e) => return Err(e),
                        Err(e) => {
                            return Ok(Err(NativeError::invalid_input(format!("fts5 text: {e}"))))
                        }
                    };
                    match (|| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?)))() {
                        Ok((rid, Some(txt))) => {
                            if let Some(s) = Self::window_snippet(&txt, terms) {
                                m.insert(rid, s);
                            }
                        }
                        Ok((_, None)) => {}
                        Err(e) if is_retryable(&e) => return Err(e),
                        Err(e) => {
                            return Ok(Err(NativeError::invalid_input(format!("fts5 text: {e}"))))
                        }
                    }
                }
                Ok(Ok(m))
            };
            out.push(with_busy_retry(run)??);
        }
        Ok(out)
    }

    /// A `…`-delimited window of `txt` around its earliest match of any `terms`
    /// trigram — the fuzzy snippet, built without FTS5's `snippet()` (which needs a
    /// MATCH). Searches `txt`'s own char-trigrams (lowercased per window) so the
    /// match position indexes the ORIGINAL text and the slice preserves its case;
    /// `SNIPPET_TOKENS` chars of context flank the match. `None` if nothing matched
    /// (no snippet rather than a misleading head-of-text slice).
    fn window_snippet(txt: &str, terms: &[Trigram]) -> Option<String> {
        let chars: Vec<char> = txt.chars().collect();
        if chars.len() < MIN_TRIGRAM {
            return None;
        }
        let ctx = SNIPPET_TOKENS as usize;
        let hit = (0..=chars.len() - MIN_TRIGRAM).find(|&i| {
            let tri: String = chars[i..i + MIN_TRIGRAM]
                .iter()
                .collect::<String>()
                .to_lowercase();
            terms.iter().any(|t| t.as_str() == tri)
        })?;
        let start = hit.saturating_sub(ctx);
        let end = (hit + MIN_TRIGRAM + ctx).min(chars.len());
        let mut s = String::new();
        if start > 0 {
            s.push('…');
        }
        s.extend(&chars[start..end]);
        if end < chars.len() {
            s.push('…');
        }
        Some(s)
    }
}

#[cfg(test)]
mod lexical_tests {
    use super::*;

    /// A `Trigram` from a literal (panics if not ≤ MAX_LEN — test-only).
    fn tg(s: &str) -> Trigram {
        Trigram::try_from_str(s).unwrap()
    }
    /// `&[&str]` literals → `Vec<Trigram>`.
    fn tgs(ss: &[&str]) -> Vec<Trigram> {
        ss.iter().map(|s| tg(s)).collect()
    }

    fn store() -> DerivedEngine {
        let dir = std::env::temp_dir().join(format!(
            "shrike-derived-lex-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
        build_snapshot_live(
            &e,
            &[
                (
                    1,
                    "field".into(),
                    "Front".into(),
                    "the mitochondria is the powerhouse".into(),
                ),
                (
                    2,
                    "field".into(),
                    "Front".into(),
                    "momentum is mass times velocity".into(),
                ),
            ],
            1,
        )
        .unwrap();
        e
    }

    /// An EMPTY store plus its directory, for the bitmap-tier tests that build their
    /// own rows and inspect the backing tables.
    fn fresh_store() -> (DerivedEngine, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "shrike-derived-delta-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
        (e, dir)
    }

    #[test]
    fn substring_finds_literal_hits_and_signals_fallback() {
        let e = store();
        let hits = e
            .search_substring("mitochondria", 10, None, &[])
            .unwrap()
            .unwrap();
        assert_eq!(hits[0].0, 1);
        assert!(e.search_substring("mi", 10, None, &[]).unwrap().is_none()); // sub-trigram → fallback
        assert!(e
            .search_substring("q\"uo", 10, None, &[])
            .unwrap()
            .unwrap()
            .is_empty()); // quotes safe
    }

    #[test]
    fn fuzzy_ranks_typos_and_floors_noise() {
        let e = store();
        let hits = e.search_fuzzy("mitochondira", 10, None, &[]).unwrap(); // transposition
        assert!(hits.iter().any(|(nid, ..)| *nid == 1));
        assert!(e.search_fuzzy("xy", 10, None, &[]).unwrap().is_empty()); // too short to rank
    }

    #[test]
    fn fuzzy_rank_query_matches_accumulate_and_rank_overlap() {
        // The bit-sliced threshold ranker must produce byte-identical results to the
        // reference (accumulate_overlap + rank_overlap) on identical inputs — across
        // multi-segment notes (Front+Back, so a note's best field wins the collapse),
        // ties (ranked by note id), and every top_k cut (which exercises the
        // early-stop). Both paths consume the same pruned terms + loaded bitmaps, so
        // any divergence is the ranker's, not the prune/load.
        let dir = std::env::temp_dir().join(format!(
            "shrike-derived-xcheck-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
        let words = ["alphabravo", "charlie", "deltaecho", "foxtrot", "golfhotel"];
        let mut rows: Vec<(i64, String, String, String)> = Vec::new();
        for n in 1..=12usize {
            let front = format!(
                "{} {}",
                words[n % words.len()],
                words[(n + 1) % words.len()]
            );
            let back = words[(n + 2) % words.len()].to_string();
            rows.push((n as i64, "field".into(), "Front".into(), front));
            rows.push((n as i64, "field".into(), "Back".into(), back));
        }
        build_snapshot_live(&e, &rows, 1).unwrap();

        let conn = e.read_pool.checkout().unwrap();
        let policy = e.fuzzy_cap_policy();
        let queries = [
            "alphabravo charlie",
            "deltaecho foxtrot golfhotel",
            "alphabravo charlie deltaecho",
            "charlie golfhotel",
        ];
        for q in queries {
            let Some(grams) = DerivedEngine::fuzzy_grams(q) else {
                continue;
            };
            let gram_vec: Vec<Trigram> = grams.iter().copied().collect();
            let df = DerivedEngine::trigram_dfs(&conn, &gram_vec).unwrap();
            let pruned = DerivedEngine::prune_to_rare_terms(&grams, &df, policy);
            let bitmaps = DerivedEngine::load_trigram_bitmaps(&conn, &pruned).unwrap();
            for &top_k in &[1usize, 2, 3, 5, 10, 100] {
                // Reference: accumulate_overlap → candidates(≥2) → seg_meta → rank.
                let overlap = DerivedEngine::accumulate_overlap(&pruned, &bitmaps);
                let candidates: Vec<i64> = overlap
                    .iter()
                    .filter(|(_, &c)| c >= FUZZY_MIN_SHARED)
                    .map(|(&r, _)| r)
                    .collect();
                let seg_meta =
                    DerivedEngine::seg_meta_for_rowids(&conn, &candidates, &[], None).unwrap();
                let reference = DerivedEngine::rank_overlap(&overlap, &seg_meta, top_k);
                let bit_sliced =
                    DerivedEngine::fuzzy_rank_query(&conn, &pruned, &bitmaps, top_k, &[], None)
                        .unwrap();
                assert_eq!(reference, bit_sliced, "query {q:?} top_k {top_k}");
            }
        }
    }

    #[test]
    fn fuzzy_ranks_by_overlap_not_rowid_recall_safe() {
        // The overlap ranker's load-bearing property: the cut is by trigram
        // overlap, NEVER by rowid. The strongest match here is the LAST-ingested
        // note (highest rowid) — a rowid `LIMIT` (the old bm25 over-fetch) would
        // drop it; overlap ranks it first. Distractors share two query trigrams
        // (qwx, zvk); the target "qwxzvk" shares all four (qwx, wxz, xzv, zvk).
        let dir = std::env::temp_dir().join(format!(
            "shrike-derived-overlap-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
        let mut rows: Vec<(i64, String, String, String)> = (1..=5)
            .map(|n| {
                (
                    n,
                    "field".into(),
                    "Front".into(),
                    format!("qwx zvk distractor {n}"),
                )
            })
            .collect();
        // Highest note id → last ingested → highest rowid.
        rows.push((6, "field".into(), "Front".into(), "qwxzvk".into()));
        build_snapshot_live(&e, &rows, 1).unwrap();

        // top_k=1: the highest-rowid, highest-overlap note wins — not a low-rowid
        // distractor. Proves the cut is overlap-ordered, not rowid-truncated.
        let top1 = e.search_fuzzy("qwxzvk", 1, None, &[]).unwrap();
        assert_eq!(top1.iter().map(|(nid, ..)| *nid).collect::<Vec<_>>(), [6]);
        // Widen: the target still leads, distractors (overlap 2) follow by note id.
        let top_all = e.search_fuzzy("qwxzvk", 10, None, &[]).unwrap();
        assert_eq!(top_all[0].0, 6, "highest overlap ranks first");
        assert_eq!(
            top_all.len(),
            6,
            "all overlap>=2 notes surface, none rowid-dropped"
        );
    }

    #[test]
    fn fuzzy_scope_is_a_rust_filter_equivalent_to_filtering_the_unscoped_results() {
        // Scope is applied in Rust at provenance hydration (seg_meta drops out-of-scope
        // note_ids), not as a `note_id IN` MATCH predicate. Two invariants: scoping to
        // EVERY id is a no-op (equals the unscoped fast path exactly — survivors, order,
        // source/ref, AND snippet), and scoping to a SUBSET returns exactly the unscoped
        // results whose note is in scope, same ranking. The first is the recall-
        // neutrality guarantee (deferred provenance + the Rust scope filter change
        // nothing observed); the second proves the filter actually bites. top_k (10) is
        // never hit here (<= 10 notes), so the subset equals the filtered unscoped slice.
        let dir = std::env::temp_dir().join(format!(
            "shrike-derived-fuzzyparity-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
        let mut rows: Vec<(i64, String, String, String)> = (1..=8)
            .map(|n| {
                (
                    n,
                    "field".into(),
                    "Front".into(),
                    format!("mitochondria powerhouse cell {n}"),
                )
            })
            .collect();
        rows.push((
            9,
            "field".into(),
            "Front".into(),
            "wholly unrelated text".into(),
        ));
        // A hidden-source segment: the fast path reads its posting (rowid-only) but
        // must drop it at hydration under `exclude_sources` — the same in-Rust filter
        // as scope. Results must agree with AND without the exclude.
        rows.push((
            10,
            "vlm".into(),
            "Image".into(),
            "mitochondria powerhouse described".into(),
        ));
        build_snapshot_live(&e, &rows, 1).unwrap();
        let all_ids: Vec<i64> = (1..=10).collect();
        let subset: Vec<i64> = vec![2, 4, 6];

        for (q, exclude) in [
            ("mitochondira", &[][..]),      // typo, no exclude → vlm note present
            ("mitochondira", &["vlm"][..]), // typo, exclude vlm → vlm note dropped
            ("mitochondria powerhouse", &["vlm"][..]),
            ("powerhuse cell", &[][..]),
        ] {
            let unfiltered = e.search_fuzzy(q, 10, None, exclude).unwrap();
            let scoped_all = e.search_fuzzy(q, 10, Some(&all_ids), exclude).unwrap();
            assert!(!unfiltered.is_empty(), "expected fuzzy hits for {q:?}");
            assert_eq!(
                unfiltered, scoped_all,
                "scope=all diverged from the unscoped fast path for {q:?} exclude={exclude:?}"
            );
            // A SUBSET scope returns exactly the unscoped results whose note is in
            // scope, in the same order — the in-Rust filter, nothing else moved.
            let scoped_subset = e.search_fuzzy(q, 10, Some(&subset), exclude).unwrap();
            let expected: Vec<_> = unfiltered
                .iter()
                .filter(|m| subset.contains(&m.0))
                .cloned()
                .collect();
            assert_eq!(
                scoped_subset, expected,
                "scope=subset was not the in-scope slice of unscoped for {q:?} exclude={exclude:?}"
            );
            // top_k is applied AFTER the scope filter: a subset scope with top_k below
            // the in-scope match count returns the in-scope slice truncated — NOT the
            // unscoped top_k then filtered. Guards against truncating before filtering.
            assert!(
                expected.len() >= 2,
                "fixture must have >= 2 in-scope hits for {q:?}"
            );
            let scoped_top1 = e.search_fuzzy(q, 1, Some(&subset), exclude).unwrap();
            assert_eq!(
                scoped_top1,
                expected[..1].to_vec(),
                "scope+top_k was not the truncated in-scope slice for {q:?} exclude={exclude:?}"
            );
            // An empty scope (a deck/tag that matched no notes) drops everything — the
            // same "match nothing" the SQL substring path applies for an empty scope.
            assert!(
                e.search_fuzzy(q, 10, Some(&[]), exclude)
                    .unwrap()
                    .is_empty(),
                "empty scope must return no fuzzy hits for {q:?} exclude={exclude:?}"
            );
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn fuzzy_overlap_is_over_rare_trigrams_not_a_common_phrase() {
        // The deliberate floor shift this ranker makes: a candidate must share the
        // query's RARE (discriminative) trigrams, not merely a common phrase. With
        // "the theory of" inflated to a high document frequency, a query typo'd on
        // the discriminative word surfaces the genuine near-match and DROPS a note
        // that shares only the common phrase — a precision win for a fuzzy signal.
        let dir = std::env::temp_dir().join(format!(
            "shrike-derived-discrim-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
        // 30 notes carrying the common phrase → its trigrams get a high DF and are
        // pruned out of every query's rare set.
        let mut rows: Vec<(i64, String, String, String)> = (1..=30)
            .map(|n| {
                (
                    n,
                    "field".into(),
                    "Front".into(),
                    format!("the theory of subject number {n}"),
                )
            })
            .collect();
        rows.push((
            100,
            "field".into(),
            "Front".into(),
            "the theory of relativity".into(),
        ));
        rows.push((
            101,
            "field".into(),
            "Front".into(),
            "the theory of evolution".into(),
        ));
        build_snapshot_live(&e, &rows, 1).unwrap();

        let hits = e
            .search_fuzzy("the theory of relatvity", 10, None, &[])
            .unwrap();
        let ids: Vec<i64> = hits.iter().map(|(nid, ..)| *nid).collect();
        assert!(
            ids.contains(&100),
            "the genuine typo near-match (relativity) surfaces via its rare trigrams"
        );
        assert!(
            !ids.contains(&101),
            "a common-phrase-only coincidence (evolution) is dropped — rare-set floor"
        );
    }

    #[test]
    fn exclude_sources_hides_vector_only_rows_from_lexical_search() {
        // A VectorOnly recognition source (VLM describe) is STORED for
        // provenance + reconcile, but excluded from substring/fuzzy BEFORE
        // ranking — so its prose can NEVER surface on a lexical query.
        let dir = std::env::temp_dir().join(format!(
            "shrike-derived-excl-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
        build_snapshot_live(
            &e,
            &[
                // A normal field row + a VLM-describe row, same distinctive term.
                (
                    1,
                    "field".into(),
                    "Front".into(),
                    "ordinary field text".into(),
                ),
                (
                    2,
                    "vlm".into(),
                    "photo.png".into(),
                    "a sunlit mountain valley with grazing cattle".into(),
                ),
            ],
            1,
        )
        .unwrap();

        // Unscoped, no exclusion: the vlm row IS findable (the row exists).
        let visible = e
            .search_substring("mountain", 10, None, &[])
            .unwrap()
            .unwrap();
        assert!(visible.iter().any(|(nid, ..)| *nid == 2));
        // The describe prose is hidden once "vlm" is excluded — substring AND
        // fuzzy both drop it, and the row is gone before ranking/limit.
        let hidden = e
            .search_substring("mountain", 10, None, &["vlm"])
            .unwrap()
            .unwrap();
        assert!(hidden.iter().all(|(nid, ..)| *nid != 2));
        let fz = e
            .search_fuzzy("montain valley", 10, None, &["vlm"])
            .unwrap();
        assert!(fz.iter().all(|(nid, ..)| *nid != 2));
        // The ordinary field row is unaffected by the exclusion.
        let field = e
            .search_substring("field", 10, None, &["vlm"])
            .unwrap()
            .unwrap();
        assert!(field.iter().any(|(nid, ..)| *nid == 1));
        // match_rows honors the exclusion directly too.
        let raw = e
            .match_rows("\"valley\"", 10, false, None, &["vlm"])
            .unwrap();
        assert!(raw.is_empty());

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn batch_lexical_matches_loop_of_singular() {
        // The batched reads must return, per query and in order, EXACTLY what
        // looping the singular reads returns — across servable / sub-trigram /
        // no-match queries, scoped / unscoped / empty-scope, and with / without a
        // hidden source. Batching changes only HOW the reads are issued (one lock,
        // one scope staging, one compiled statement), never WHAT they return.
        let dir = std::env::temp_dir().join(format!(
            "shrike-derived-batchparity-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
        build_snapshot_live(
            &e,
            &[
                (
                    1,
                    "field".into(),
                    "Front".into(),
                    "the mitochondria is the powerhouse".into(),
                ),
                (
                    2,
                    "field".into(),
                    "Front".into(),
                    "momentum is mass times velocity".into(),
                ),
                (
                    3,
                    "field".into(),
                    "Front".into(),
                    "mitochondrial dna replication".into(),
                ),
                // A hidden (VectorOnly) source sharing a term, to exercise exclude.
                (
                    3,
                    "vlm".into(),
                    "img.png".into(),
                    "a labelled diagram of the mitochondria".into(),
                ),
            ],
            1,
        )
        .unwrap();

        // A growth policy (cap > 6 on a long query) so the parity check exercises a
        // per-query cap that is NOT the fixed floor — proving batch==serial holds
        // when the cap varies by query length, not just at fixed-6.
        e.set_fuzzy_cap_policy(FuzzyCapPolicy {
            floor: 6,
            k: 2.7,
            ceiling: 12,
        });

        // literal hit, transposition typo, sub-trigram (None/empty), no-match,
        // second literal, and a LONG multi-word query (many trigrams → cap > 6 under
        // the growth policy, so the pruned set differs from the short queries') — one
        // of each kind the two paths must agree on.
        let queries = [
            "mitochondria",
            "mitochondira",
            "mi",
            "zzznomatch",
            "momentum",
            "the mitochondrial dna replication powerhouse momentum velocity",
        ];
        let q_refs: Vec<&str> = queries.to_vec();
        let scope_some: [i64; 2] = [1, 3];
        let scope_empty: [i64; 0] = [];
        let scopes: [Option<&[i64]>; 3] = [None, Some(&scope_some), Some(&scope_empty)];
        let excl_vlm: [&str; 1] = ["vlm"];
        let excludes: [&[&str]; 2] = [&[], &excl_vlm];
        let limit = 10i64;

        for scope in scopes {
            for exclude in excludes {
                let want_sub: Vec<Option<Vec<LexicalRow>>> = queries
                    .iter()
                    .map(|q| e.search_substring(q, limit, scope, exclude).unwrap())
                    .collect();
                let got_sub = e
                    .search_substring_batch(&q_refs, limit, scope, exclude)
                    .unwrap();
                assert_eq!(
                    got_sub, want_sub,
                    "substring parity (scope={scope:?}, exclude={exclude:?})"
                );

                let want_fz: Vec<Vec<LexicalRow>> = queries
                    .iter()
                    .map(|q| e.search_fuzzy(q, limit, scope, exclude).unwrap())
                    .collect();
                let got_fz = e
                    .search_fuzzy_batch(&q_refs, limit, scope, exclude)
                    .unwrap();
                assert_eq!(
                    got_fz, want_fz,
                    "fuzzy parity (scope={scope:?}, exclude={exclude:?})"
                );
            }
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn nfc_nfd_equivalent_forms_match() {
        // Canonically-equivalent strings in different normalization forms must
        // match — both indexed text and queries are NFC-normalized. Tests BOTH
        // directions: an NFD query against an NFC index (query normalization) and
        // an NFC query against an NFD-indexed note (index normalization).
        let composed = "café"; // NFC: c a f é(U+00E9)
        let decomposed = "cafe\u{0301}"; // NFD: c a f e + combining acute
        assert_ne!(
            composed, decomposed,
            "the two forms are distinct byte sequences"
        );

        let mk = |label: &str, stored: &str| {
            let dir = std::env::temp_dir().join(format!(
                "shrike-derived-nfc-{}-{}-{}",
                label,
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
            build_snapshot_live(&e, &[(1, "field".into(), "Front".into(), stored.into())], 1)
                .unwrap();
            (e, dir)
        };

        // (1) NFC-indexed, queried in NFD — the query is normalized to match.
        let (e, dir) = mk("idxnfc", composed);
        assert!(
            e.search_substring(decomposed, 10, None, &[])
                .unwrap()
                .unwrap()
                .iter()
                .any(|(nid, ..)| *nid == 1),
            "NFD query finds NFC-indexed substring"
        );
        assert!(
            e.search_fuzzy(decomposed, 10, None, &[])
                .unwrap()
                .iter()
                .any(|(nid, ..)| *nid == 1),
            "NFD query finds NFC-indexed fuzzy"
        );
        std::fs::remove_dir_all(dir).ok();

        // (2) NFD-indexed, queried in NFC — the indexed text is normalized.
        let (e, dir) = mk("idxnfd", decomposed);
        assert!(
            e.search_substring(composed, 10, None, &[])
                .unwrap()
                .unwrap()
                .iter()
                .any(|(nid, ..)| *nid == 1),
            "NFC query finds NFD-indexed substring (index normalized on insert)"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn trigram_dfs_reports_document_frequency() {
        let dir = std::env::temp_dir().join(format!(
            "shrike-derived-df-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
        build_snapshot_live(
            &e,
            &[
                (1, "field".into(), "Front".into(), "abc xyz".into()),
                (2, "field".into(), "Front".into(), "abc def".into()),
            ],
            1,
        )
        .unwrap();
        let conn = e.read_pool.checkout().unwrap();
        let df = DerivedEngine::trigram_dfs(&conn, &tgs(&["abc", "xyz", "qqq"])).unwrap();
        assert_eq!(df.get("abc"), Some(&2), "'abc' is in both docs");
        assert_eq!(df.get("xyz"), Some(&1), "'xyz' is in one doc");
        assert_eq!(df.get("qqq"), None, "absent trigram has no row (DF 0)");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn trigram_df_snapshot_equals_live_vocab_after_build() {
        // The materialized DF the fuzzy prune reads must equal what fts5vocab would
        // compute live — so on a freshly-built index the prune picks the SAME rare
        // trigrams and fuzzy results are identical to the always-fresh-vocab path.
        // (Staleness between rebuilds is by design; this pins the fresh case.)
        let dir = std::env::temp_dir().join(format!(
            "shrike-derived-dfsnap-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
        build_snapshot_live(
            &e,
            &[
                (
                    1,
                    "field".into(),
                    "Front".into(),
                    "the quick brown fox".into(),
                ),
                (
                    2,
                    "field".into(),
                    "Front".into(),
                    "the lazy brown dog".into(),
                ),
                (
                    3,
                    "field".into(),
                    "Front".into(),
                    "quick quick quick".into(),
                ),
            ],
            1,
        )
        .unwrap();
        let conn = e.read_pool.checkout().unwrap();
        let collect = |sql: &str| -> std::collections::BTreeMap<String, i64> {
            let mut stmt = conn.prepare(sql).unwrap();
            stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
                .unwrap()
                .map(Result::unwrap)
                .collect()
        };
        let live = collect("SELECT term, doc FROM idx_vocab");
        let snapshot = collect("SELECT term, df FROM trigram_df");
        assert!(
            !snapshot.is_empty(),
            "the build must materialize trigram_df"
        );
        assert_eq!(
            snapshot, live,
            "trigram_df snapshot diverged from idx_vocab"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn refresh_derived_snapshots_picks_up_incrementally_added_trigrams() {
        // The #955 fix: a rebuild materializes trigram_df, but an incremental
        // ingest_many adds a note whose NOVEL trigram the snapshot doesn't know (it
        // is refreshed only at rebuild). refresh_derived_snapshots re-materializes
        // from the live index, so the fuzzy prune sees the new trigram with no full
        // rebuild — the debounced ingest-tail refresh's job, here driven directly.
        let dir = std::env::temp_dir().join(format!(
            "shrike-derived-dfincr-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
        build_snapshot_live(
            &e,
            &[(1, "field".into(), "Front".into(), "alpha beta".into())],
            1,
        )
        .unwrap();
        // Incremental write (NOT a rebuild) of a note carrying a novel trigram "zqw".
        e.ingest_many(&[(2, vec![("Front".into(), "zqwx".into())])], "field")
            .unwrap();
        {
            // Stale: the rebuild's snapshot predates the incremental write, so the
            // new trigram is absent (DF 0) — the fuzzy prune would mis-rank it.
            let conn = e.read_pool.checkout().unwrap();
            assert_eq!(
                DerivedEngine::trigram_dfs(&conn, &[tg("zqw")])
                    .unwrap()
                    .get("zqw"),
                None,
                "the snapshot is stale before the refresh"
            );
        }
        shrike_store::DerivedStore::refresh_derived_snapshots(&e).unwrap();
        {
            let conn = e.read_pool.checkout().unwrap();
            assert_eq!(
                DerivedEngine::trigram_dfs(&conn, &[tg("zqw")])
                    .unwrap()
                    .get("zqw"),
                Some(&1),
                "refresh picks up the incrementally-added trigram"
            );
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn prune_to_rare_keeps_rarest_present_plus_all_absent_drops_common() {
        use std::collections::{BTreeSet, HashMap};
        let grams: BTreeSet<Trigram> =
            tgs(&["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "zzz"])
                .into_iter()
                .collect();
        let df: HashMap<Trigram, i64> = [
            ("aaa", 1000),
            ("bbb", 500),
            ("ccc", 3),
            ("ddd", 1),
            ("eee", 2),
            ("fff", 800),
            ("ggg", 50),
            ("zzz", 0), // absent from the snapshot (a typo, OR written since #958)
        ]
        .iter()
        .map(|(k, v)| (tg(k), *v))
        .collect();
        // Under the default (fixed-6) policy: the 6 rarest PRESENT (rarest-first),
        // THEN every absent trigram appended.
        let terms = DerivedEngine::prune_to_rare_terms(&grams, &df, FuzzyCapPolicy::default());
        assert_eq!(
            terms,
            tgs(&["ddd", "eee", "ccc", "ggg", "bbb", "fff", "zzz"])
        );
        assert!(
            !terms.iter().any(|t| t.as_str() == "aaa"),
            "dropped the commonest PRESENT (DF 1000, beyond the rarest 6)"
        );
        assert!(
            terms.iter().any(|t| t.as_str() == "zzz"),
            "KEEPS the absent (DF 0) trigram — it may be written-since-snapshot"
        );
    }

    #[test]
    fn fuzzy_cap_default_is_fixed_six() {
        // The default policy is byte-identical to the historical FUZZY_MAX_TRIGRAMS
        // truncate: every gram count clamps to 6, so adopting the policy is a no-op
        // until an eval sets a different one.
        let p = FuzzyCapPolicy::default();
        assert_eq!(p.floor, FUZZY_MAX_TRIGRAMS);
        assert_eq!(p.ceiling, FUZZY_MAX_TRIGRAMS);
        for n in [0usize, 1, 5, 6, 7, 12, 30, 60, 200] {
            assert_eq!(p.cap(n), 6, "default cap must stay fixed-6 at n={n}");
        }
    }

    #[test]
    fn fuzzy_cap_curve_table() {
        // Pin the log-growth curve at the representative trigram counts the eval
        // sweeps, so a change to the constants or the rounding is caught here. The
        // curve is `clamp(6 + round(2.7·ln(n/6)), 6, 12)`.
        let p = FuzzyCapPolicy {
            floor: 6,
            k: 2.7,
            ceiling: 12,
        };
        // (n, expected cap): the floor pins n<=6 at 6; growth is slow (log); the
        // ceiling pins the long tail at 12.
        let table = [
            (2usize, 6usize), // below floor → floor
            (6, 6),           // at floor → floor (ln term is 0)
            (9, 7),           // 6 + round(2.7·0.405)=6+1
            (12, 8),          // 6 + round(2.7·0.693)=6+2
            (18, 9),          // 6 + round(2.7·1.099)=6+3
            (24, 10),         // 6 + round(2.7·1.386)=6+4
            (30, 10),         // 6 + round(2.7·1.609)=6+4
            (45, 11),         // 6 + round(2.7·2.015)=6+5
            (60, 12),         // 6 + round(2.7·2.303)=6+6 → ceiling
            (120, 12),        // well past the ceiling → clamped to 12
        ];
        for (n, want) in table {
            assert_eq!(p.cap(n), want, "cap({n}) under k=2.7 ceiling=12");
        }
    }

    #[test]
    fn fuzzy_cap_floor_sweep() {
        // The floor sweep arms (fixed-6/5/4): floor==ceiling pins the cap flat at
        // the floor for every n — the simplest lever the eval measures.
        for floor in [4usize, 5, 6] {
            let p = FuzzyCapPolicy {
                floor,
                k: 2.7,
                ceiling: floor,
            };
            for n in [0usize, 3, floor, floor + 10, 60] {
                assert_eq!(p.cap(n), floor, "fixed-{floor} cap at n={n}");
            }
        }
    }

    #[test]
    fn fuzzy_cap_misconfigured_ceiling_below_floor_degrades_to_floor() {
        // A ceiling below the floor must not panic the clamp (an inverted range) —
        // the upper bound is taken as max(floor, ceiling), so a mis-config degrades
        // to "fixed at the floor" rather than aborting.
        let p = FuzzyCapPolicy {
            floor: 6,
            k: 2.7,
            ceiling: 3,
        };
        for n in [2usize, 6, 30, 100] {
            assert_eq!(p.cap(n), 6, "pins at the floor at n={n}");
        }
    }

    #[test]
    fn prune_grows_cap_under_a_log_policy() {
        // Under a growth policy a longer gram set keeps MORE of its rarest present
        // trigrams than fixed-6 — the recall lever the eval measures. With 8 present
        // grams and cap(8)=7 (6 + round(2.7·ln(8/6))=6+round(0.78)=7), the prune
        // keeps 7, one more than the fixed-6 default.
        use std::collections::{BTreeSet, HashMap};
        let grams: BTreeSet<Trigram> =
            tgs(&["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh"])
                .into_iter()
                .collect();
        let df: HashMap<Trigram, i64> = [
            ("aaa", 800),
            ("bbb", 700),
            ("ccc", 600),
            ("ddd", 500),
            ("eee", 400),
            ("fff", 300),
            ("ggg", 200),
            ("hhh", 100),
        ]
        .iter()
        .map(|(k, v)| (tg(k), *v))
        .collect();
        let growth = FuzzyCapPolicy {
            floor: 6,
            k: 2.7,
            ceiling: 12,
        };
        let kept = DerivedEngine::prune_to_rare_terms(&grams, &df, growth);
        assert_eq!(kept.len(), 7, "cap(8)=7 keeps the 7 rarest present");
        // The 7 rarest (lowest DF first); the commonest one ("aaa", DF 800) is the
        // only one dropped.
        assert_eq!(
            kept,
            tgs(&["hhh", "ggg", "fff", "eee", "ddd", "ccc", "bbb"])
        );
        // Default fixed-6 over the same input keeps only 6.
        let default_kept =
            DerivedEngine::prune_to_rare_terms(&grams, &df, FuzzyCapPolicy::default());
        assert_eq!(default_kept.len(), 6, "fixed-6 keeps 6");
    }

    #[test]
    fn sample_posting_passes_through_at_or_below_the_ceiling() {
        // A posting that fits is returned BYTE-IDENTICALLY: no sampling fires at any
        // standard scale, so the common case keeps every rowid.
        let exact: Vec<i64> = (0..FUZZY_POSTING_CEILING as i64).collect();
        assert_eq!(
            DerivedEngine::sample_posting(exact.clone()),
            exact,
            "a posting AT the ceiling is untouched"
        );
        let small: Vec<i64> = (0..1000).collect();
        assert_eq!(
            DerivedEngine::sample_posting(small.clone()),
            small,
            "a posting below the ceiling is untouched"
        );
    }

    #[test]
    fn sample_posting_caps_and_spans_past_the_ceiling() {
        // A posting past the ceiling is capped to EXACTLY the ceiling, and the sample
        // SPANS the whole range — not a prefix `LIMIT` (which would drop the
        // highest-rowid notes). The posting is the rowids 0..2N for a ceiling N, so
        // every kept value reveals where in the range it was drawn from.
        let n = FUZZY_POSTING_CEILING;
        let posting: Vec<i64> = (0..2 * n as i64).collect();
        let sampled = DerivedEngine::sample_posting(posting);
        assert_eq!(sampled.len(), n, "capped to exactly the ceiling");
        assert_eq!(sampled[0], 0, "keeps the lowest rowid");
        // A prefix LIMIT over 0..2n would top out at n-1; an evenly-spaced sample
        // reaches the high half, proving the high-rowid tail is sampled, not cut. The
        // last kept index is (n-1)*2n/n = 2n-2, so the max is at the very top.
        let max = *sampled.last().unwrap();
        assert!(
            max >= n as i64,
            "the sample spans past the prefix a LIMIT would cut at (max {max} >= {n})"
        );
        // Ascending in, ascending out (the stride walks a sorted posting in order).
        assert!(
            sampled.windows(2).all(|w| w[0] < w[1]),
            "the sample preserves ascending rowid order with no duplicates"
        );
    }

    #[test]
    fn sample_posting_just_over_the_ceiling_caps_spans_and_dedups() {
        // The FRAGILE boundary the 2x cap/span test misses: just past the ceiling the
        // integer-division stride `i * len / CEILING` is barely above 1.0, the regime
        // where consecutive floors could collide and yield a DUPLICATE rowid (which
        // would over-count a note's overlap, corrupting the fuzzy ranking). They
        // don't — when `len > CEILING` the real stride exceeds 1, so consecutive
        // floors strictly increase — but only a near-ceiling length exercises it.
        // Sweeps every length in `[CEILING+1, CEILING+64]` plus a few irregular
        // multiples: capped to exactly the ceiling, lowest rowid kept, the high tail
        // reached (not a prefix cut), and strictly ascending (== no duplicate).
        let n = FUZZY_POSTING_CEILING;
        for len in (n + 1..=n + 64).chain([n + n / 2, 2 * n + 1, 3 * n - 1, 10 * n + 7]) {
            let posting: Vec<i64> = (0..len as i64).collect();
            let s = DerivedEngine::sample_posting(posting);
            assert_eq!(s.len(), n, "len {len}: capped to exactly the ceiling");
            assert_eq!(s[0], 0, "len {len}: keeps the lowest rowid");
            // The fragile property: strictly ascending == NO duplicate rowid. A
            // colliding integer-division stride (two `i` mapping to the same floor)
            // would repeat a rowid and over-count its note's overlap; because `len >
            // CEILING` the real stride exceeds 1, floors strictly increase and no
            // rowid repeats. This is what the 2x test cannot isolate.
            assert!(
                s.windows(2).all(|w| w[0] < w[1]),
                "len {len}: strictly ascending — no duplicate rowid from a colliding stride"
            );
            // Anti-prefix-cut: the sample reaches at least as far as a prefix LIMIT
            // would (max == (n-1)*len/n >= n-1). At len == n+1 the two coincide (only
            // one element is dropped, so max == n-1), so the boundary-correct
            // assertion is `>=`, strengthening to `>` once len is comfortably over the
            // ceiling — the high-rowid tail is sampled, never truncated.
            let max = *s.last().unwrap();
            let prefix_max = n as i64 - 1;
            assert!(
                max >= prefix_max,
                "len {len}: never worse than a prefix cut"
            );
            if len >= 2 * n {
                assert!(
                    max > prefix_max,
                    "len {len}: spans strictly past a prefix cut"
                );
            }
        }
    }

    #[test]
    fn sample_posting_is_deterministic() {
        // Same posting in → same sample out: the bound is a pure function of the
        // term's own rowids (no batch-wide or positional-stateful input), so the
        // batched and singular fuzzy paths sample identically.
        let posting: Vec<i64> = (0..3 * FUZZY_POSTING_CEILING as i64)
            .map(|x| x * 7)
            .collect();
        let a = DerivedEngine::sample_posting(posting.clone());
        let b = DerivedEngine::sample_posting(posting);
        assert_eq!(
            a, b,
            "deterministic across repeated calls on the same posting"
        );
    }

    #[test]
    fn fuzzy_finds_a_match_via_just_written_trigrams_under_a_stale_snapshot() {
        // #958: a note matching the query ONLY through trigrams written since the DF
        // snapshot must still surface — the prune keeps absent (DF-0) trigrams, so
        // the overlap scan sees the just-written ones even though the snapshot lags.
        // The old behavior (truncate the absent trigrams) dropped this note.
        let dir = std::env::temp_dir().join(format!(
            "shrike-derived-recall958-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
        // Eight notes sharing a common prefix; the rebuild materializes trigram_df
        // over THEM (so "commonprefix" trigrams are present, the suffix's are not).
        let rows: Vec<(i64, String, String, String)> = (1..=8)
            .map(|n| {
                (
                    n,
                    "field".into(),
                    "Front".into(),
                    format!("commonprefix alpha {n}"),
                )
            })
            .collect();
        build_snapshot_live(&e, &rows, 1).unwrap();
        // Incremental write (NOT a rebuild) of a note that overlaps the query ONLY
        // via novel trigrams absent from the snapshot.
        e.ingest_many(&[(100, vec![("Front".into(), "zzqxwv".into())])], "field")
            .unwrap();
        // Stale snapshot — deliberately do NOT refresh trigram_df.
        let hits = e.search_fuzzy("commonprefixzzqxwv", 50, None, &[]).unwrap();
        assert!(
            hits.iter().any(|(nid, ..)| *nid == 100),
            "note 100, matched only via just-written trigrams, must surface under a stale snapshot"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn delete_then_insert_reusing_freed_rowids_surfaces_new_notes() {
        // A delete-heavy-then-insert batch reuses the freed high rowids. The new notes'
        // trigrams are NOVEL (absent at build), so they are unmaterialized and served
        // by the live posting read — surfacing regardless of the reused rowids. (The
        // matching materialized-trigram reuse case — where a base bitmap holds a
        // reused rowid — is covered by `materialized_trigram_delta_handles_rowid_reuse`.)
        let dir = std::env::temp_dir().join(format!(
            "shrike-derived-genreuse-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
        let rows: Vec<(i64, String, String, String)> = (1..=10i64)
            .map(|n| (n, "field".into(), "Front".into(), format!("commonword{n}")))
            .collect();
        build_snapshot_live(&e, &rows, 1).unwrap();
        e.remove(&[6, 7, 8, 9, 10], None).unwrap();
        e.ingest_many(
            &[
                (11, vec![("Front".into(), "xraywhiskey".into())]),
                (12, vec![("Front".into(), "uniformtango".into())]),
                (13, vec![("Front".into(), "sierraromeo".into())]),
            ],
            "field",
        )
        .unwrap();
        for (nid, q) in [
            (11, "xraywhiskey"),
            (12, "uniformtango"),
            (13, "sierraromeo"),
        ] {
            let hits = e.search_fuzzy(q, 10, None, &[]).unwrap();
            assert!(
                hits.iter().any(|h| h.0 == nid),
                "new note {nid} ({q}) must surface after a reused-rowid write"
            );
        }
        std::fs::remove_dir_all(dir).ok();
    }

    /// A raw read connection to a derived store's backing file (for asserting on the
    /// bitmap-tier tables directly).
    fn raw(dir: &std::path::Path) -> Connection {
        let c = Connection::open(dir.join("shrike.db")).unwrap();
        rusqlite::vtab::array::load_module(&c).unwrap();
        c
    }

    fn term_count(conn: &Connection, table: &str, term: &str) -> i64 {
        conn.query_row(
            &format!("SELECT count(*) FROM {table} WHERE term = ?1"),
            [term],
            |r| r.get(0),
        )
        .unwrap()
    }

    #[test]
    fn incremental_add_to_a_materialized_trigram_surfaces_via_the_delta() {
        // A note added incrementally to a MATERIALIZED trigram's posting surfaces
        // immediately — no fold, no rebuild — because the query reads
        // (base ∪ added) \ removed and the write put its rowid in `added`.
        let (e, dir) = fresh_store();
        let rows: Vec<(i64, String, String, String)> = (1..=5i64)
            .map(|n| {
                (
                    n,
                    "field".into(),
                    "Front".into(),
                    format!("alphabet entry {n}"),
                )
            })
            .collect();
        build_snapshot_live(&e, &rows, 1).unwrap();
        // Sanity: "alp" is materialized (small collection, DF ≪ C).
        let c = raw(&dir);
        assert_eq!(term_count(&c, "trigram_bitmap", "alp"), 1);

        e.ingest_many(
            &[(6, vec![("Front".into(), "alphabet latecomer".into())])],
            "field",
        )
        .unwrap();
        // The write recorded a delta for the materialized trigram, and NO fold ran.
        assert!(
            term_count(&c, "trigram_delta", "alp") >= 1,
            "delta tracks the add"
        );
        let hits = e.search_fuzzy("alphabet", 10, None, &[]).unwrap();
        assert!(
            hits.iter().any(|(nid, ..)| *nid == 6),
            "note 6 must surface via base ∪ added, before any fold"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn materialized_trigram_delta_handles_rowid_reuse() {
        // The delete-then-reuse case THROUGH a materialized trigram: a note holding a
        // materialized trigram is deleted (freeing its rowid), then a new note with no
        // such trigram reuses that rowid. The base bitmap still holds the freed rowid,
        // so without the delta's `removed` the query would hydrate the reused rowid and
        // surface the WRONG note. (#998 acceptance: rowid reuse handled via the delta.)
        let (e, dir) = fresh_store();
        // Three notes share "alphakeyword"; note 3 is built last → highest rowid.
        let rows: Vec<(i64, String, String, String)> = (1..=3i64)
            .map(|n| {
                (
                    n,
                    "field".into(),
                    "Front".into(),
                    "alphakeyword here".into(),
                )
            })
            .collect();
        build_snapshot_live(&e, &rows, 1).unwrap();
        // Remove note 3 (frees the max idx rowid), then add a note with NO "alpha…"
        // trigrams — FTS5 reuses the freed rowid for it.
        e.remove(&[3], None).unwrap();
        e.ingest_many(
            &[(4, vec![("Front".into(), "zulu mike november".into())])],
            "field",
        )
        .unwrap();
        // Prove the reuse actually happened (else the test wouldn't exercise the hazard).
        let c = raw(&dir);
        let reused: Option<i64> = c
            .query_row(
                "SELECT note_id FROM rowmap WHERE rowid = 3 AND source = 'field'",
                [],
                |r| r.get(0),
            )
            .optional()
            .unwrap();
        assert_eq!(reused, Some(4), "note 4 reused note 3's freed rowid 3");

        let hits = e.search_fuzzy("alphakeyword", 10, None, &[]).unwrap();
        assert!(
            !hits.iter().any(|(nid, ..)| *nid == 4),
            "note 4 (reused rowid, no alpha trigrams) must NOT surface for 'alphakeyword'"
        );
        assert!(
            !hits.iter().any(|(nid, ..)| *nid == 3),
            "deleted note 3 must not surface"
        );
        assert!(
            hits.iter().any(|(nid, ..)| *nid == 1) && hits.iter().any(|(nid, ..)| *nid == 2),
            "the surviving notes 1 and 2 still surface"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn fold_absorbs_the_delta_into_the_base() {
        // The debounced fold absorbs a materialized trigram's pending delta into its
        // base and clears the delta — the result is unchanged (the note still
        // surfaces, now via base), and the base cardinality equals the live DF.
        let (e, dir) = fresh_store();
        let rows: Vec<(i64, String, String, String)> = (1..=3i64)
            .map(|n| {
                (
                    n,
                    "field".into(),
                    "Front".into(),
                    format!("alphabet entry {n}"),
                )
            })
            .collect();
        build_snapshot_live(&e, &rows, 1).unwrap();
        e.ingest_many(
            &[(4, vec![("Front".into(), "alphabet four".into())])],
            "field",
        )
        .unwrap();
        let c = raw(&dir);
        assert!(
            term_count(&c, "trigram_delta", "alp") >= 1,
            "delta present pre-fold"
        );

        e.fold_trigram_bitmaps().unwrap();

        assert_eq!(
            term_count(&c, "trigram_delta", "alp"),
            0,
            "the fold cleared the delta"
        );
        assert_eq!(
            term_count(&c, "trigram_dirty", "alp"),
            0,
            "the fold cleared the dirty set"
        );
        // The folded base now equals the live DF (4 notes share "alp").
        let df: i64 = c
            .query_row("SELECT df FROM trigram_df WHERE term = 'alp'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(df, 4, "all four notes share the trigram");
        let hits = e.search_fuzzy("alphabet", 10, None, &[]).unwrap();
        assert!(
            hits.iter().any(|(nid, ..)| *nid == 4),
            "note 4 still surfaces, now via the folded base"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn fold_demotes_a_trigram_that_grew_common_promote_waits_for_rebuild() {
        // With C = 3: "xqz" starts materialized (DF 2 < 3), "wkj" unmaterialized (DF
        // 4 ≥ 3). Writes flip both — "xqz" rises to DF 4, "wkj" falls to DF 2. The fold
        // DEMOTES the now-common "xqz" (drops its base, served live) but does NOT
        // promote "wkj" (promote is deferred to the rebuild — rebuilding its
        // trigrams()-fold posting off the FTS5-folded live index could desync). A full
        // rebuild then materializes "wkj".
        let (e, dir) = fresh_store();
        e.set_materialize_ceiling(3);
        let mut rows: Vec<(i64, String, String, String)> = Vec::new();
        for n in 1..=2 {
            rows.push((n, "field".into(), "Front".into(), "xqz seed".into()));
        }
        for n in 3..=6 {
            rows.push((n, "field".into(), "Front".into(), "wkj seed".into()));
        }
        build_snapshot_live(&e, &rows, 1).unwrap();
        let c = raw(&dir);
        assert_eq!(
            term_count(&c, "trigram_bitmap", "xqz"),
            1,
            "xqz materialized (DF 2 < 3)"
        );
        assert_eq!(
            term_count(&c, "trigram_bitmap", "wkj"),
            0,
            "wkj unmaterialized (DF 4 ≥ 3)"
        );

        // Raise xqz to DF 4 (demote candidate); drop wkj to DF 2 (would-promote).
        e.ingest_many(
            &[
                (7, vec![("Front".into(), "xqz more".into())]),
                (8, vec![("Front".into(), "xqz more".into())]),
            ],
            "field",
        )
        .unwrap();
        e.remove(&[5, 6], None).unwrap();

        e.fold_trigram_bitmaps().unwrap();

        assert_eq!(
            term_count(&c, "trigram_bitmap", "xqz"),
            0,
            "xqz demoted by the fold (grew to DF 4 ≥ 3)"
        );
        assert_eq!(
            term_count(&c, "trigram_bitmap", "wkj"),
            0,
            "wkj NOT promoted by the fold — promote is deferred to the rebuild"
        );
        assert_eq!(term_count(&c, "trigram_dirty", "xqz"), 0, "dirty cleared");

        // A full rebuild materializes the now-rare wkj.
        e.build(
            &[
                (1, "field".into(), "Front".into(), "xqz seed".into()),
                (2, "field".into(), "Front".into(), "xqz seed".into()),
                (3, "field".into(), "Front".into(), "wkj seed".into()),
                (4, "field".into(), "Front".into(), "wkj seed".into()),
                (7, "field".into(), "Front".into(), "xqz more".into()),
                (8, "field".into(), "Front".into(), "xqz more".into()),
            ],
            &[1, 2, 3, 4, 7, 8],
            2,
        )
        .unwrap();
        assert_eq!(
            term_count(&c, "trigram_bitmap", "wkj"),
            1,
            "the rebuild materialized wkj (DF 2 < 3)"
        );
        assert_eq!(
            term_count(&c, "trigram_bitmap", "xqz"),
            0,
            "xqz stays unmaterialized after rebuild (DF 4 ≥ 3)"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn divergent_script_surfaces_via_the_materialized_tier() {
        // Greek content (whose str::to_lowercase fold diverges from FTS5's on final
        // sigma) is findable through the materialized base+delta tier: build + an
        // incremental add of the same word both surface. Sanity that the
        // single-tokenizer base serves divergent scripts at all.
        let (e, dir) = fresh_store();
        let rows: Vec<(i64, String, String, String)> = (1..=3i64)
            .map(|n| {
                (
                    n,
                    "field".into(),
                    "Front".into(),
                    format!("ΛΟΓΟΣ ΘΕΟΣ note {n}"),
                )
            })
            .collect();
        build_snapshot_live(&e, &rows, 1).unwrap();
        e.ingest_many(
            &[(4, vec![("Front".into(), "ΛΟΓΟΣ ΘΕΟΣ late".into())])],
            "field",
        )
        .unwrap();
        let hits = e.search_fuzzy("ΛΟΓΟΣ ΘΕΟΣ", 10, None, &[]).unwrap();
        assert!(
            hits.iter().any(|(nid, ..)| *nid == 1),
            "built Greek note surfaces"
        );
        assert!(
            hits.iter().any(|(nid, ..)| *nid == 4),
            "added Greek note surfaces"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn divergent_fold_rowid_reuse_no_false_positive() {
        // THE fold-divergence regression. The base MUST be keyed by the same tokenizer
        // (`trigrams()`) as the delta and the query — NOT FTS5's own fold. A note with
        // two final-sigma words ("ΛΟΓΟΣ ΘΕΟΣ") yields TWO trigrams that FTS5 folds to σ
        // (γοσ, εοσ) but `trigrams()` folds to ς (γος, εος). With an FTS5-keyed base
        // (the bug), a delete keyed by `trigrams()` can't shed the rowid from the σ-keyed
        // base entries, so after a rowid reuse a medial-σ query (which `trigrams()` folds
        // to γοσ/εοσ) reads the stale base and surfaces the unrelated reusing note with
        // overlap 2 (≥ FUZZY_MIN_SHARED) — a false positive. A `trigrams()`-keyed base
        // sheds the rowid under the same keys the query reads, so it cannot.
        let (e, dir) = fresh_store();
        let rows: Vec<(i64, String, String, String)> = (1..=3i64)
            .map(|n| (n, "field".into(), "Front".into(), "ΛΟΓΟΣ ΘΕΟΣ".into()))
            .collect();
        build_snapshot_live(&e, &rows, 1).unwrap();
        e.remove(&[3], None).unwrap(); // frees the max idx rowid
        e.ingest_many(
            &[(99, vec![("Front".into(), "hotel india juliet kilo".into())])],
            "field",
        )
        .unwrap();
        let c = raw(&dir);
        let reused: Option<i64> = c
            .query_row(
                "SELECT note_id FROM rowmap WHERE rowid = 3 AND source = 'field'",
                [],
                |r| r.get(0),
            )
            .optional()
            .unwrap();
        assert_eq!(reused, Some(99), "note 99 reused note 3's freed rowid");
        // Medial sigmas (followed by a letter) → trigrams() keeps σ → γοσ, εοσ.
        let hits = e.search_fuzzy("αγοσα βεοσγ", 10, None, &[]).unwrap();
        assert!(
            !hits.iter().any(|(nid, ..)| *nid == 99),
            "the reused-rowid note must NOT surface (no false positive from a stale base key)"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn schema_reset_drops_the_bitmap_tier() {
        // A schema bump (or idx/rowmap desync) resets the derived data; with no global
        // freshness gate, the bitmap tier MUST be dropped too, or a stale base bitmap
        // over a wiped idx would serve rowids that no longer mean what they did.
        let (e, dir) = fresh_store();
        build_snapshot_live(
            &e,
            &[(
                1,
                "field".into(),
                "Front".into(),
                "alphabet entry one".into(),
            )],
            9,
        )
        .unwrap();
        {
            let c = raw(&dir);
            assert!(
                term_count(&c, "trigram_bitmap", "alp") >= 1,
                "materialized pre-reset"
            );
        }
        drop(e);
        // Reopen at a different schema version → reset.
        let e2 = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 99).unwrap();
        drop(e2);
        let c = raw(&dir);
        assert_eq!(
            c.query_row("SELECT count(*) FROM trigram_bitmap", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0,
            "the schema reset dropped the base bitmaps"
        );
        std::fs::remove_dir_all(dir).ok();
    }
}

#[cfg(test)]
mod hardening_tests {
    //! Open-time integrity, fallible count, journal-mode policy, and
    //! the staged-id-set path for collection-scale scopes/deletes.

    use super::*;

    fn temp_db() -> (std::path::PathBuf, std::path::PathBuf) {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "shrike-derived-hardening-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shrike.db");
        (dir, path)
    }

    #[test]
    fn open_resets_on_idx_rowmap_desync() {
        let (dir, path) = temp_db();
        {
            let e = DerivedEngine::open(path.to_str().unwrap(), 1).unwrap();
            build_snapshot_live(
                &e,
                &[(1, "field".into(), "Front".into(), "consistent text".into())],
                100,
            )
            .unwrap();
            assert_eq!(e.count().unwrap(), 1);
        }
        // Desync the pairing out-of-band: an idx row with no rowmap partner.
        {
            let raw = Connection::open(&path).unwrap();
            raw.execute("INSERT INTO idx(txt) VALUES('orphan text')", [])
                .unwrap();
        }
        // Reopen: the integrity check treats the mismatch as corruption —
        // empty store, col_mod watermark cleared so the next drift rebuilds.
        let e = DerivedEngine::open(path.to_str().unwrap(), 1).unwrap();
        assert_eq!(e.count().unwrap(), 0);
        assert_eq!(e.get_col_mod(), None);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn count_surfaces_a_broken_store_instead_of_zero() {
        let (dir, path) = temp_db();
        let e = DerivedEngine::open(path.to_str().unwrap(), 1).unwrap();
        assert_eq!(e.count().unwrap(), 0);
        // Break the store out-of-band; count must error, not read as empty.
        {
            let raw = Connection::open(&path).unwrap();
            raw.execute("DROP TABLE rowmap", []).unwrap();
        }
        let err = e.count().unwrap_err();
        assert_eq!(err.kind(), shrike_error::ErrorKind::Unavailable);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn open_puts_the_store_in_wal_mode() {
        let (dir, path) = temp_db();
        // A pre-existing rollback-journal store (e.g. one written by an older build).
        {
            let raw = Connection::open(&path).unwrap();
            raw.pragma_update(None, "journal_mode", "DELETE").unwrap();
        }
        let e = DerivedEngine::open(path.to_str().unwrap(), 1).unwrap();
        // open() switches the file to WAL — persistent in the file header, so a
        // fresh connection to the same file (a pool read connection) inherits it
        // and reads concurrently with the single writer.
        let raw = Connection::open(&path).unwrap();
        let mode: String = raw
            .query_row("PRAGMA journal_mode", [], |r| r.get(0))
            .unwrap();
        assert_eq!(mode.to_lowercase(), "wal");
        drop(e);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn read_pool_serves_concurrent_reads_and_reuses_connections() {
        let (dir, path) = temp_db();
        let e = std::sync::Arc::new(DerivedEngine::open(path.to_str().unwrap(), 1).unwrap());
        build_snapshot_live(
            &e,
            &[
                (
                    1,
                    "field".into(),
                    "Front".into(),
                    "the mitochondria is the powerhouse".into(),
                ),
                (
                    2,
                    "field".into(),
                    "Front".into(),
                    "momentum is mass times velocity".into(),
                ),
                (
                    3,
                    "field".into(),
                    "Front".into(),
                    "mitochondrial dna replication".into(),
                ),
            ],
            1,
        )
        .unwrap();

        // Many threads hammer both lexical reads at once. Under the old single
        // mutexed connection these serialized; the pool hands each thread its own
        // connection so they run concurrently (WAL) and must all agree on the
        // committed data — a serialization bug would corrupt or deadlock here.
        let threads = 8;
        let mut handles = Vec::new();
        for _ in 0..threads {
            let e = std::sync::Arc::clone(&e);
            handles.push(std::thread::spawn(move || {
                for _ in 0..50 {
                    let sub = e
                        .search_substring_batch(&["mitochondria"], 10, None, &[])
                        .unwrap();
                    assert_eq!(sub.len(), 1);
                    let ids: std::collections::BTreeSet<i64> = sub[0]
                        .as_ref()
                        .expect("served")
                        .iter()
                        .map(|r| r.0)
                        .collect();
                    assert_eq!(ids, std::collections::BTreeSet::from([1, 3]));

                    let fz = e
                        .search_fuzzy_batch(&["mitochondira"], 10, None, &[])
                        .unwrap();
                    assert_eq!(fz.len(), 1);
                    assert!(!fz[0].is_empty(), "the transposition typo fuzzy-matches");
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // Every checkout was returned (reuse, no leak) and the pool grew no wider
        // than the live concurrency: idle count is in 1..=threads.
        let idle = e.read_pool.idle.lock().unwrap().len();
        assert!(
            (1..=threads).contains(&idle),
            "pool idle conns = {idle}, expected 1..={threads}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn read_pool_reads_concurrently_with_writes() {
        let (dir, path) = temp_db();
        let e = std::sync::Arc::new(DerivedEngine::open(path.to_str().unwrap(), 1).unwrap());
        build_snapshot_live(
            &e,
            &[(
                1,
                "field".into(),
                "Front".into(),
                "mitochondria seed".into(),
            )],
            1,
        )
        .unwrap();

        // Start the writer and readers together (the barrier) so the reads
        // genuinely overlap the writes; both run BOUNDED loops so the index can
        // never blow up. The window stays small — a few hundred tiny notes.
        let writes = 200i64;
        let reads = 200;
        let readers = 4;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(readers + 1));

        // Writer: incremental ingests on the SINGLE write connection while reads run.
        let writer = {
            let e = std::sync::Arc::clone(&e);
            let barrier = std::sync::Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for i in 0..writes {
                    e.ingest(
                        100 + i,
                        "field",
                        &[("Front".to_string(), format!("mitochondria gen {i}"))],
                    )
                    .unwrap();
                }
            })
        };

        // Readers: pooled reads keep returning a consistent snapshot (always at
        // least the seed note) and must never error or deadlock against the
        // in-flight writer — the WAL reader/writer concurrency the pool exists for.
        let mut handles = Vec::new();
        for _ in 0..readers {
            let e = std::sync::Arc::clone(&e);
            let barrier = std::sync::Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..reads {
                    let sub = e
                        .search_substring_batch(&["mitochondria"], 50, None, &[])
                        .unwrap();
                    assert!(
                        !sub[0].as_ref().expect("served").is_empty(),
                        "the seed note is in every committed snapshot"
                    );
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        writer.join().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rebuild_swap_window_must_not_surface_invalid_input_for_a_valid_read() {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(8100);
        let dir = std::env::temp_dir().join(format!(
            "shrike-swaprace-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shrike.db");
        let e = std::sync::Arc::new(DerivedEngine::open(path.to_str().unwrap(), 1).unwrap());
        let seed: Vec<(i64, String, String, String)> = (1..=20)
            .map(|n| {
                (
                    n,
                    "field".into(),
                    "F".into(),
                    format!("mitochondria note {n}"),
                )
            })
            .collect();
        build_snapshot_live(&e, &seed, 1).unwrap();

        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        // CPU burners starve the scheduler so a reader is pre-empted INSIDE the
        // FTS5 vtable constructor during the swap — the contention the parallel
        // suite creates. Without this the window is too narrow to observe.
        let ncpu = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let mut burners = Vec::new();
        for _ in 0..(ncpu * 2) {
            let stop = std::sync::Arc::clone(&stop);
            burners.push(std::thread::spawn(move || {
                let mut x: u64 = 1;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    for _ in 0..50_000 {
                        x = x.wrapping_mul(2654435761).wrapping_add(1);
                    }
                    std::hint::black_box(x);
                }
            }));
        }

        let rebuilder = {
            let e = std::sync::Arc::clone(&e);
            let stop = std::sync::Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut gen = 2i64;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let rows: Vec<(i64, String, String, String)> = (1..=20)
                        .map(|n| {
                            (
                                n,
                                "field".into(),
                                "F".into(),
                                format!("mitochondria note {n} gen {gen}"),
                            )
                        })
                        .collect();
                    build_snapshot_live(&e, &rows, gen).unwrap();
                    gen += 1;
                }
            })
        };

        let hit = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let e = std::sync::Arc::clone(&e);
            let hit = std::sync::Arc::clone(&hit);
            let stop = std::sync::Arc::clone(&stop);
            handles.push(std::thread::spawn(move || {
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    if let Err(err) = e.search_fuzzy_batch(&["mitochondria"], 50, None, &[]) {
                        hit.lock()
                            .unwrap()
                            .push(format!("fuzzy kind={:?}: {err}", err.kind()));
                    }
                    if let Err(err) = e.search_substring_batch(&["mitochondria"], 50, None, &[]) {
                        hit.lock()
                            .unwrap()
                            .push(format!("substring kind={:?}: {err}", err.kind()));
                    }
                }
            }));
        }

        std::thread::sleep(std::time::Duration::from_secs(3));
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in handles {
            h.join().unwrap();
        }
        rebuilder.join().unwrap();
        for b in burners {
            b.join().unwrap();
        }

        let errs = hit.lock().unwrap();
        assert!(
            errs.is_empty(),
            "a valid lexical read concurrent with a rebuild surfaced {} error(s):\n{}",
            errs.len(),
            errs.iter().take(5).cloned().collect::<Vec<_>>().join("\n")
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn scope_and_delete_work_beyond_the_inline_cap() {
        let (dir, path) = temp_db();
        let e = DerivedEngine::open(path.to_str().unwrap(), 1).unwrap();
        let n = (DerivedEngine::INLINE_ID_MAX + 100) as i64;
        let rows: Vec<(i64, String, String, String)> = (1..=n)
            .map(|i| {
                (
                    i,
                    "field".into(),
                    "Front".into(),
                    format!("note body {i} shared"),
                )
            })
            .collect();
        build_snapshot_live(&e, &rows, 100).unwrap();
        assert_eq!(e.count().unwrap(), n);

        // A scope wider than the inline cap rides the staged TEMP table and
        // still restricts correctly.
        let scope: Vec<i64> = (1..=n).collect();
        let hits = e
            .match_rows("\"shared\"", 10, false, Some(&scope), &[])
            .unwrap();
        assert!(!hits.is_empty());
        let narrow = e
            .match_rows("\"shared\"", 10, false, Some(&[2]), &[])
            .unwrap();
        assert_eq!(narrow.iter().map(|r| r.0).collect::<Vec<_>>(), vec![2]);

        // A delete wider than the inline cap clears everything in one call.
        e.remove(&scope, None).unwrap();
        assert_eq!(e.count().unwrap(), 0);
        // And the idx side went with the rowmap side (the pairing held).
        let conn = e.lock();
        let idx_left: i64 = conn
            .query_row("SELECT count(*) FROM idx", [], |r| r.get(0))
            .unwrap();
        assert_eq!(idx_left, 0);
        std::fs::remove_dir_all(dir).ok();
    }
}
