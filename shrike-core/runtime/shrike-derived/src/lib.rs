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

use rusqlite::Connection;
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

impl DerivedEngine {
    /// Open (or create) the store and ensure the schema, resetting the derived
    /// data on a schema-version mismatch (no migrations — it's a rebuildable
    /// cache). Errors are `unavailable` — the facade recovers by discarding.
    /// The current sidecar schema. v2: the `segments` table for recognition
    /// structure (boxes/spans) + recognition meta keys. A bump drops
    /// everything — recognition rows re-derive via the pending sweep.
    pub const SCHEMA_VERSION: i64 = 2;

    /// Open (or create) the sidecar database at `path`, migrating to
    /// `schema_version`.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or its schema migrated.
    pub fn open(path: &str, schema_version: i64) -> NativeResult<Self> {
        configure_sqlite_perf();
        let conn = Connection::open(path).map_err(db_err)?;
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

    fn create_tables(conn: &Connection) -> NativeResult<()> {
        conn.execute(Self::IDX_DDL, []).map_err(db_err)?;
        conn.execute(Self::IDX_VOCAB_DDL, []).map_err(db_err)?;
        conn.execute(Self::TRIGRAM_DF_DDL, []).map_err(db_err)?;
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

    fn delete_rows(conn: &Connection, note_ids: &[i64], source: Option<&str>) -> NativeResult<()> {
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
        Ok(())
    }

    fn insert_rows(
        conn: &Connection,
        note_id: i64,
        source: &str,
        refs_text: &[(String, String)],
    ) -> NativeResult<()> {
        // prepare_cached: the two insert statements parse once per connection,
        // not once per row (a rebuild would otherwise pay ~2 prepares per field
        // row; the cache also serves every later ingest).
        let mut ins_idx = conn
            .prepare_cached("INSERT INTO idx(txt) VALUES(?1)")
            .map_err(db_err)?;
        let mut ins_map = conn
            .prepare_cached("INSERT INTO rowmap(rowid, note_id, source, ref) VALUES(?1,?2,?3,?4)")
            .map_err(db_err)?;
        for (reference, text) in refs_text {
            if text.trim().is_empty() {
                continue;
            }
            // The idx→rowmap pairing rides last_insert_rowid() on THIS
            // connection — sound only under the engine's single mutexed
            // connection (the module-docs invariant; verified at open). The text
            // is NFC-normalized so the index agrees with NFC-normalized queries.
            ins_idx.execute([nfc(text).as_ref()]).map_err(db_err)?;
            let rowid = conn.last_insert_rowid();
            ins_map
                .execute(rusqlite::params![rowid, note_id, source, reference])
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
        let tx = conn.transaction().map_err(db_err)?;
        Self::delete_rows(&tx, &[note_id], Some(source))?;
        Self::insert_rows(&tx, note_id, source, refs_text)?;
        tx.commit().map_err(db_err)
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
        let tx = conn.transaction().map_err(db_err)?;
        let ids: Vec<i64> = notes.iter().map(|(id, _)| *id).collect();
        Self::delete_rows(&tx, &ids, Some(source))?;
        for (i, (note_id, refs_text)) in notes.iter().enumerate() {
            if last.get(note_id) == Some(&i) {
                Self::insert_rows(&tx, *note_id, source, refs_text)?;
            }
        }
        tx.commit().map_err(db_err)
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
        let tx = conn.transaction().map_err(db_err)?;
        Self::delete_rows(&tx, note_ids, source)?;
        Self::delete_gated(&tx, note_ids, source)?;
        tx.commit().map_err(db_err)
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
        // finished rebuild leaves it fresh; best-effort, because a stale snapshot
        // only degrades the prune (which tolerates drift), never correctness — it
        // must not fail an otherwise-successful rebuild.
        if let Err(e) = self.refresh_trigram_df() {
            tracing::warn!(
                error = %e,
                "trigram_df refresh failed; fuzzy prune will use a stale DF snapshot"
            );
        }
        Ok(total)
    }

    /// Re-materialize [`Self::TRIGRAM_DF_DDL`]'s `trigram_df` from the live index's
    /// `idx_vocab`. fts5vocab computes `doc` by walking each term's doclist, so doing
    /// it once here lets the fuzzy prune read DF as a cheap primary-key lookup
    /// instead of re-counting doclists per query. Its own short transaction, so a
    /// concurrent reader sees the prior snapshot until it commits — never an empty
    /// table. DF only orders trigrams for the prune, which tolerates drift (a stale
    /// snapshot shifts WHICH trigrams a query scans, not whether a match counts), so
    /// between rebuilds the snapshot may lag the live index.
    ///
    /// # Errors
    ///
    /// Returns an error if the vocabulary read or the table rewrite fails.
    fn refresh_trigram_df(&self) -> NativeResult<()> {
        let mut conn = self.lock();
        let tx = conn.transaction().map_err(db_err)?;
        tx.execute("DELETE FROM trigram_df", []).map_err(db_err)?;
        tx.execute(
            "INSERT INTO trigram_df(term, df) SELECT term, doc FROM idx_vocab",
            [],
        )
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
        // Small scopes inline as literals (i64s — no injection surface);
        // large ones are staged in a TEMP table, since SQLite's parser caps
        // long expression lists.
        let scope_clause = match scope {
            Some(ids) if !ids.is_empty() => {
                if ids.len() <= Self::INLINE_ID_MAX {
                    let csv = ids
                        .iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    format!("AND m.note_id IN ({csv}) ")
                } else {
                    Self::stage_id_set(&conn, "shrike_scope_ids", ids)?;
                    "AND m.note_id IN (SELECT id FROM temp.shrike_scope_ids) ".to_string()
                }
            }
            Some(_) => "AND 0 ".to_string(), // an empty scope matches nothing
            None => String::new(),
        };
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
    /// scope staging and ONE compiled statement (through the connection's
    /// statement cache) across the whole batch. The fused-search lexical reads
    /// call this once for all query strings rather than re-staging the scope and
    /// recompiling the statement per query. Returns one row vector per
    /// expression, in `exprs` order.
    ///
    /// `conn` is a checked-out [`ReadPool`] connection — read-only by discipline
    /// (SELECTs plus the TEMP scope staging below, never a store write).
    ///
    /// The scope (when present) is staged ONCE in a per-connection TEMP table and
    /// referenced as an invariant `IN (SELECT id FROM temp.…)` subquery, so the
    /// SQL text — hence the statement-cache key — is identical for every query in
    /// the batch and the statement compiles once for the whole set. (The singular
    /// [`Self::match_rows`] inlines a small scope; the result set is identical.)
    ///
    /// # Errors
    ///
    /// Returns an error if scope staging fails, or any expression's MATCH query
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
        // Stage the scope ONCE for the whole batch as an invariant subquery: it is
        // referenced by every query below, and the staged form keeps the SQL — the
        // statement-cache key — stable so the statement compiles once.
        let scope_clause = match scope {
            Some(ids) if !ids.is_empty() => {
                Self::stage_id_set(conn, "shrike_scope_ids", ids)?;
                "AND m.note_id IN (SELECT id FROM temp.shrike_scope_ids) ".to_string()
            }
            Some(_) => "AND 0 ".to_string(), // an empty scope matches nothing
            None => String::new(),
        };
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
    /// Per-term `term = ?` primary-key seek on the plain table (mmap-served, one
    /// cached statement), NOT a doclist count through fts5vocab. Reads on the passed
    /// [`ReadPool`] connection.
    ///
    /// # Errors
    ///
    /// Returns an error if the lookup fails.
    fn trigram_dfs(
        conn: &Connection,
        terms: &[&str],
    ) -> NativeResult<std::collections::HashMap<String, i64>> {
        if terms.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        let run = || -> rusqlite::Result<std::collections::HashMap<String, i64>> {
            let mut stmt = conn.prepare_cached("SELECT df FROM trigram_df WHERE term = ?1")?;
            let mut m = std::collections::HashMap::new();
            for &term in terms {
                let mut q = stmt.query([term])?;
                if let Some(r) = q.next()? {
                    m.insert(term.to_string(), r.get::<_, i64>(0)?);
                }
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

pub use shrike_store::LexicalRow;

/// Lowercased char-level trigrams (mirrors the Python `_trigrams`: code-point
/// windows over `text.lower()`).
pub fn trigrams(text: &str) -> Vec<String> {
    let lowered: Vec<char> = text.to_lowercase().chars().collect();
    if lowered.len() < MIN_TRIGRAM {
        return Vec::new();
    }
    (0..=lowered.len() - MIN_TRIGRAM)
        .map(|i| lowered[i..i + MIN_TRIGRAM].iter().collect())
        .collect()
}

/// Quote a term as an FTS5 string literal (wrap in double quotes, double
/// internal ones) — the only safe way to feed arbitrary user text into MATCH.
pub fn fts_quote(term: &str) -> String {
    format!("\"{}\"", term.replace('"', "\"\""))
}

impl DerivedEngine {
    /// One query's trigram set, `None` when it has fewer than [`FUZZY_MIN_SHARED`]
    /// trigram windows (too short to rank). NFC-normalized so the trigrams match
    /// the NFC-normalized index. [`Self::prune_to_rare_terms`] derives the (smaller)
    /// rare-trigram set the overlap ranker actually queries and counts.
    fn fuzzy_grams(query: &str) -> Option<std::collections::BTreeSet<String>> {
        let normalized = nfc(query);
        let grams = trigrams(normalized.trim());
        if grams.len() < FUZZY_MIN_SHARED {
            return None;
        }
        Some(grams.into_iter().collect())
    }

    /// The fuzzy candidate trigrams: only the **rarest** (lowest document-frequency)
    /// of `grams`, capped at [`FUZZY_MAX_TRIGRAMS`]. The common trigrams bloat the
    /// match set and discriminate least; the rare ones are what actually find a
    /// typo'd word. `df` is per-trigram document frequency from `idx_vocab` (0 =
    /// absent → sorts rarest, matches nothing, harmless). The "common"/"rare"
    /// judgement is the collection's own statistics — no language assumption. The
    /// overlap ranker counts how many of THESE a candidate segment shares, so the
    /// floor is over the rare set, not the full query gram set.
    fn prune_to_rare_terms(
        grams: &std::collections::BTreeSet<String>,
        df: &std::collections::HashMap<String, i64>,
    ) -> Vec<String> {
        let mut by_df: Vec<&String> = grams.iter().collect();
        // Rarest PRESENT trigram first; DF 0 (absent from the index — matches
        // nothing) sorts last so it never crowds out a rare-but-present trigram.
        // Term as a deterministic tie-break.
        by_df.sort_by(|a, b| {
            let rank = |g: &String| {
                let d = df.get(g).copied().unwrap_or(0);
                (d == 0, d)
            };
            rank(a).cmp(&rank(b)).then_with(|| a.cmp(b))
        });
        by_df.truncate(FUZZY_MAX_TRIGRAMS);
        by_df.into_iter().cloned().collect()
    }

    /// Accumulate one query's per-segment trigram overlap: how many of its pruned
    /// (rare) trigrams each indexed rowid shares, from the per-term posting sets
    /// gathered by [`Self::fuzzy_term_rowids`] (rowid-only) or
    /// [`Self::term_segments_batch`] (with provenance). Provenance-free, so it runs
    /// before the survivors' `(note_id, source, ref)` is known.
    fn accumulate_overlap(
        pruned_terms: &[String],
        term_rowids: &std::collections::HashMap<String, Vec<i64>>,
    ) -> std::collections::HashMap<i64, usize> {
        let mut overlap: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        for term in pruned_terms {
            if let Some(rowids) = term_rowids.get(term) {
                for &rid in rowids {
                    *overlap.entry(rid).or_insert(0) += 1;
                }
            }
        }
        overlap
    }

    /// Rank one query's accumulated overlap into its survivors. A segment's overlap
    /// is how many pruned trigrams matched it; a note's overlap is its best
    /// segment's. Keep notes sharing at least [`FUZZY_MIN_SHARED`] pruned trigrams,
    /// one (best-overlap, lowest-rowid) segment per note, ordered overlap-desc then
    /// note-id-asc, capped at `top_k`. Returns `(note_id, source, ref, rowid)`; the
    /// rowid drives the deferred snippet read. Recall-safe: the cut is by overlap,
    /// never by rowid — every matched segment is ranked, unlike a bm25 `LIMIT` over
    /// the OR. A rowid present in `overlap` but ABSENT from `seg_meta` is dropped:
    /// the JOIN read pre-filters scope/exclude so its `seg_meta` carries only kept
    /// segments, and the unfiltered path's [`Self::seg_meta_for_rowids`] omits the
    /// `exclude_sources` segments — so "absent" means "filtered out," and skipping
    /// it makes a note rank by its best NON-excluded segment, exactly as the JOIN
    /// path does by never counting the excluded ones.
    fn rank_overlap(
        overlap: &std::collections::HashMap<i64, usize>,
        seg_meta: &std::collections::HashMap<i64, (i64, String, String)>,
        top_k: usize,
    ) -> Vec<(i64, String, String, i64)> {
        // Best segment per note: highest overlap, lowest rowid as the deterministic
        // tie-break (so the chosen snippet segment is stable run to run).
        let mut best: std::collections::HashMap<i64, (usize, i64)> =
            std::collections::HashMap::new();
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

    /// [`Self::accumulate_overlap`] then [`Self::rank_overlap`] against the inline
    /// `seg_meta` of the filtered JOIN read — the scope/exclude path, where
    /// provenance is read alongside the postings.
    fn merge_overlap(
        pruned_terms: &[String],
        term_rowids: &std::collections::HashMap<String, Vec<i64>>,
        seg_meta: &std::collections::HashMap<i64, (i64, String, String)>,
        top_k: usize,
    ) -> Vec<(i64, String, String, i64)> {
        let overlap = Self::accumulate_overlap(pruned_terms, term_rowids);
        Self::rank_overlap(&overlap, seg_meta, top_k)
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
    /// posting is identical for every query in the batch, since scope/exclude are
    /// batch-wide), then merged per query into per-note overlap. The UNSCOPED case
    /// (the common one) reads postings rowid-only ([`Self::fuzzy_term_rowids`]) and
    /// hydrates `(note_id, source, ref)` for only the overlap candidates
    /// ([`Self::seg_meta_for_rowids`]); a scope filter takes the JOIN path
    /// ([`Self::term_segments_batch`] + [`Self::merge_overlap`]) instead. bm25's
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
        let gram_sets: Vec<Option<std::collections::BTreeSet<String>>> =
            queries.iter().map(|q| Self::fuzzy_grams(q)).collect();
        // One batched DF lookup over every distinct trigram, then prune each served
        // query to its rarest (most discriminative) trigrams.
        let distinct: std::collections::BTreeSet<&str> = gram_sets
            .iter()
            .flatten()
            .flat_map(|g| g.iter().map(String::as_str))
            .collect();
        // One pool connection for the whole fuzzy read: the DF lookup, the posting
        // reads, and the snippet reads all run on it. Each sub-read is its own
        // statement (its own committed snapshot) — the same per-sub-read freshness
        // as before; the freshness bracket flags a write that lands mid-read.
        let conn = self.read_pool.checkout()?;
        let df = Self::trigram_dfs(&conn, &distinct.into_iter().collect::<Vec<_>>())?;
        let pruned: Vec<Option<Vec<String>>> = gram_sets
            .iter()
            .map(|g| g.as_ref().map(|gs| Self::prune_to_rare_terms(gs, &df)))
            .collect();
        // Read each DISTINCT pruned trigram's posting once (shared across queries).
        let distinct_terms: std::collections::BTreeSet<&str> = pruned
            .iter()
            .flatten()
            .flatten()
            .map(String::as_str)
            .collect();
        if distinct_terms.is_empty() {
            return Ok(queries.iter().map(|_| Vec::new()).collect());
        }
        let distinct_terms_vec: Vec<&str> = distinct_terms.into_iter().collect();
        // Merge per query into ranked survivors (note_id, source, ref, rowid). A
        // SCOPE filter is pushed into the MATCH — the `note_id IN scope` predicate
        // needs the rowmap JOIN, and for a tight scope SQL-side filtering beats
        // scanning-then-dropping — so that path reads provenance inline. The
        // unscoped case (the common one, including the always-present
        // `exclude_sources` hidden-source list) reads postings rowid-ONLY — no
        // per-posting JOIN, no string allocs over the full lists — then hydrates
        // `(note_id, source, ref)` once for the candidate set (overlap >= the
        // floor), applying `exclude_sources` in that batched read.
        let survivors: Vec<Vec<(i64, String, String, i64)>> = if scope.is_some() {
            let (term_rowids, seg_meta) =
                Self::term_segments_batch(&conn, &distinct_terms_vec, scope, exclude_sources)?;
            pruned
                .iter()
                .map(|p| {
                    p.as_ref().map_or_else(Vec::new, |terms| {
                        Self::merge_overlap(terms, &term_rowids, &seg_meta, top_k as usize)
                    })
                })
                .collect()
        } else {
            let term_rowids = Self::fuzzy_term_rowids(&conn, &distinct_terms_vec)?;
            let overlaps: Vec<std::collections::HashMap<i64, usize>> = pruned
                .iter()
                .map(|p| {
                    p.as_ref()
                        .map_or_else(std::collections::HashMap::new, |terms| {
                            Self::accumulate_overlap(terms, &term_rowids)
                        })
                })
                .collect();
            let mut candidates: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
            for ov in &overlaps {
                for (&rid, &count) in ov {
                    if count >= FUZZY_MIN_SHARED {
                        candidates.insert(rid);
                    }
                }
            }
            let seg_meta = Self::seg_meta_for_rowids(
                &conn,
                &candidates.into_iter().collect::<Vec<_>>(),
                exclude_sources,
            )?;
            overlaps
                .iter()
                .map(|ov| Self::rank_overlap(ov, &seg_meta, top_k as usize))
                .collect()
        };
        // Build snippets for the surviving rowids only, from text read by a plain
        // rowid lookup (no MATCH re-scan) and windowed in Rust around the query's
        // own rare trigrams.
        let snippet_jobs: Vec<(&[String], Vec<i64>)> = pruned
            .iter()
            .zip(&survivors)
            .map(|(p, surv)| {
                let terms: &[String] = match p {
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

    /// For each of `terms` (already FTS5-safe trigrams), the indexed segments whose
    /// text contains it: `term_rowids[term]` is the matching idx rowids, and
    /// `seg_meta[rowid]` their `(note_id, source, ref)` provenance (owned once,
    /// shared across queries). One MATCH per term — no rank, no text, no limit, just
    /// the posting — sharing ONE connection and ONE staged scope across the set, like
    /// [`Self::match_rows_batch`]. No `LIMIT`: the overlap ranker needs every match
    /// to stay recall-safe; a rowid `LIMIT` here would silently drop high-rowid
    /// (recently-added) high-overlap notes. Reads on the passed [`ReadPool`]
    /// connection.
    ///
    /// # Errors
    ///
    /// Returns an error if scope staging or any term's MATCH query fails.
    #[allow(clippy::type_complexity)]
    fn term_segments_batch(
        conn: &Connection,
        terms: &[&str],
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<(
        std::collections::HashMap<String, Vec<i64>>,
        std::collections::HashMap<i64, (i64, String, String)>,
    )> {
        let mut term_rowids: std::collections::HashMap<String, Vec<i64>> =
            std::collections::HashMap::new();
        let mut seg_meta: std::collections::HashMap<i64, (i64, String, String)> =
            std::collections::HashMap::new();
        if terms.is_empty() {
            return Ok((term_rowids, seg_meta));
        }
        let span = tracing::debug_span!("derived.fuzzy_terms", n = terms.len());
        let _enter = span.enter();
        let scope_clause = match scope {
            Some(ids) if !ids.is_empty() => {
                Self::stage_id_set(conn, "shrike_scope_ids", ids)?;
                "AND m.note_id IN (SELECT id FROM temp.shrike_scope_ids) ".to_string()
            }
            Some(_) => "AND 0 ".to_string(), // an empty scope matches nothing
            None => String::new(),
        };
        let exclude_clause = if exclude_sources.is_empty() {
            String::new()
        } else {
            let placeholders = (0..exclude_sources.len())
                .map(|i| format!("?{}", i + 2))
                .collect::<Vec<_>>()
                .join(",");
            format!("AND m.source NOT IN ({placeholders}) ")
        };
        let sql = format!(
            "SELECT idx.rowid, m.note_id, m.source, m.ref \
             FROM idx JOIN rowmap m ON m.rowid = idx.rowid \
             WHERE idx MATCH ?1 {scope_clause}{exclude_clause}"
        );
        for term in terms {
            let quoted = fts_quote(term);
            let run =
                || -> rusqlite::Result<Result<Vec<(i64, i64, String, String)>, NativeError>> {
                    let mut params: Vec<&dyn rusqlite::ToSql> = vec![&quoted];
                    params.extend(exclude_sources.iter().map(|s| s as &dyn rusqlite::ToSql));
                    let mut stmt = conn.prepare_cached(&sql)?;
                    let mut q = stmt.query(rusqlite::params_from_iter(params))?;
                    let mut rows: Vec<(i64, i64, String, String)> = Vec::new();
                    loop {
                        let row = match q.next() {
                            Ok(Some(r)) => r,
                            Ok(None) => break,
                            Err(e) if is_retryable(&e) => return Err(e),
                            Err(e) => {
                                return Ok(Err(NativeError::invalid_input(format!(
                                    "fts5 match: {e}"
                                ))))
                            }
                        };
                        match (|| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)))() {
                            Ok(tuple) => rows.push(tuple),
                            Err(e) if is_retryable(&e) => return Err(e),
                            Err(e) => {
                                return Ok(Err(NativeError::invalid_input(format!(
                                    "fts5 match: {e}"
                                ))))
                            }
                        }
                    }
                    Ok(Ok(rows))
                };
            let rows = with_busy_retry(run)??;
            let mut rids = Vec::with_capacity(rows.len());
            for (rowid, note_id, source, r) in rows {
                rids.push(rowid);
                seg_meta.entry(rowid).or_insert((note_id, source, r));
            }
            term_rowids.insert((*term).to_string(), rids);
        }
        Ok((term_rowids, seg_meta))
    }

    /// Each term's matching idx rowids, rowid-ONLY — no `rowmap` JOIN, no
    /// provenance. FTS5 yields rowids straight off the posting list with no table
    /// access, the cheapest posting read; provenance is deferred to the overlap
    /// candidates ([`Self::seg_meta_for_rowids`]). For the UNFILTERED fuzzy read
    /// only: a scope/exclude filter needs `note_id`/`source` in the MATCH, so it
    /// takes the JOIN path ([`Self::term_segments_batch`]). No `LIMIT`, for the same
    /// recall reason as that path. Reads on the passed [`ReadPool`] connection.
    ///
    /// # Errors
    ///
    /// Returns an error if any term's MATCH query fails.
    fn fuzzy_term_rowids(
        conn: &Connection,
        terms: &[&str],
    ) -> NativeResult<std::collections::HashMap<String, Vec<i64>>> {
        let mut term_rowids: std::collections::HashMap<String, Vec<i64>> =
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
            let rids = with_busy_retry(run)??;
            term_rowids.insert((*term).to_string(), rids);
        }
        Ok(term_rowids)
    }

    /// Provenance `(note_id, source, ref)` per idx rowid, for a set of rowids,
    /// dropping any whose `source` is in `exclude_sources`. Hydrates the overlap
    /// candidates the unfiltered fuzzy path deferred (it read postings rowid-only,
    /// hidden sources included), applying the `exclude_sources` filter HERE rather
    /// than in the per-posting scan. A dropped rowid is then absent from the
    /// returned map, so [`Self::rank_overlap`] skips it — the same effect as the
    /// JOIN path's `source NOT IN`, off the hot scan.
    ///
    /// The rowid set goes in as an INLINE integer `IN` list (the rowids are our own
    /// `i64`s — no injection), NOT a staged temp table. That is load-bearing: a temp
    /// table's btree pages bypass mmap (it maps only the main DB file), so they
    /// fault through `pcache1`'s `STATIC_LRU` mutex and serialize the parallel fuzzy
    /// chunks. `rowmap` lives in the mmap'd main file, so an inline `IN` seeks it
    /// (rowid is its primary key) with no shared-cache mutex. Chunked so the SQL
    /// text stays bounded. Reads on the passed [`ReadPool`] connection.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn seg_meta_for_rowids(
        conn: &Connection,
        rowids: &[i64],
        exclude_sources: &[&str],
    ) -> NativeResult<std::collections::HashMap<i64, (i64, String, String)>> {
        let mut seg_meta: std::collections::HashMap<i64, (i64, String, String)> =
            std::collections::HashMap::new();
        for chunk in rowids.chunks(Self::INLINE_ID_MAX) {
            let id_list = chunk
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                "SELECT m.rowid, m.note_id, m.source, m.ref FROM rowmap m \
                 WHERE m.rowid IN ({id_list})"
            );
            let run = || -> rusqlite::Result<Vec<(i64, i64, String, String)>> {
                let mut stmt = conn.prepare(&sql)?;
                let rows = stmt
                    .query_map([], |row| {
                        Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(rows)
            };
            for (rowid, note_id, source, r) in with_busy_retry(run)? {
                if exclude_sources.contains(&source.as_str()) {
                    continue;
                }
                seg_meta.insert(rowid, (note_id, source, r));
            }
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
        jobs: &[(&[String], Vec<i64>)],
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
    fn window_snippet(txt: &str, terms: &[String]) -> Option<String> {
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
            terms.contains(&tri)
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
    fn fuzzy_unfiltered_fast_path_equals_the_scoped_join_path() {
        // The deferred-provenance fast path (scope=None: postings read rowid-only,
        // then note_id/source/ref hydrated for the candidate set) must return
        // EXACTLY what the scope/exclude JOIN path returns. Scoping to every note id
        // forces the JOIN path over identical data, so the two results — survivors,
        // order, source/ref, AND snippet — must be equal. That equality is the
        // recall-neutrality guarantee: deferring provenance changes nothing observed.
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
        // must drop it at hydration under `exclude_sources`, exactly as the JOIN
        // path drops it in the MATCH. Both must agree with AND without the exclude.
        rows.push((
            10,
            "vlm".into(),
            "Image".into(),
            "mitochondria powerhouse described".into(),
        ));
        build_snapshot_live(&e, &rows, 1).unwrap();
        let all_ids: Vec<i64> = (1..=10).collect();

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
                "unfiltered fast path diverged from the scoped JOIN path for {q:?} exclude={exclude:?}"
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

        // literal hit, transposition typo, sub-trigram (None/empty), no-match,
        // second literal — one of each kind the two paths must agree on.
        let queries = [
            "mitochondria",
            "mitochondira",
            "mi",
            "zzznomatch",
            "momentum",
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
        let df = DerivedEngine::trigram_dfs(&conn, &["abc", "xyz", "qqq"]).unwrap();
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
    fn prune_to_rare_keeps_rarest_present_and_drops_common_and_absent() {
        use std::collections::{BTreeSet, HashMap};
        let grams: BTreeSet<String> = ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "zzz"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let df: HashMap<String, i64> = [
            ("aaa", 1000),
            ("bbb", 500),
            ("ccc", 3),
            ("ddd", 1),
            ("eee", 2),
            ("fff", 800),
            ("ggg", 50),
            ("zzz", 0), // absent from the index
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
        let terms = DerivedEngine::prune_to_rare_terms(&grams, &df);
        // Keeps the FUZZY_MAX_TRIGRAMS (6) rarest PRESENT, rarest-first.
        assert_eq!(terms, ["ddd", "eee", "ccc", "ggg", "bbb", "fff"]);
        assert!(
            !terms.iter().any(|t| t == "aaa"),
            "dropped the commonest (DF 1000)"
        );
        assert!(
            !terms.iter().any(|t| t == "zzz"),
            "dropped the absent (DF 0) trigram"
        );
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
