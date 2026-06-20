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
//! **The single mutexed connection is a correctness invariant, not just
//! thread-safety**: `idx` is an FTS5 virtual table, so nothing at the schema
//! level (no FK, no trigger) ties its rowids to `rowmap` — the pairing holds
//! because every write rides this one connection's `last_insert_rowid()` under
//! [`DerivedEngine::lock`]. A move to a connection pool must make the coupling
//! structural first.
//!
//! MATCH-expression building, trigram filtering, and the state machine stay
//! facade-side; this crate is storage + queries only. Pure Rust — no pyo3.

#![deny(missing_docs)]
#![deny(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc
)]

use std::sync::Mutex;

use rusqlite::Connection;
use shrike_error::{ErrorKind, NativeError, NativeResult};

/// Mirrors `shrike.derived.SNIPPET_TOKENS` (the facade doesn't pass it — it's
/// part of the pinned engine behaviour).
const SNIPPET_TOKENS: i64 = 12;

pub use shrike_store::MatchRow;

/// Whether this build statically links rusqlite's bundled SQLite.
/// Bundled guarantees FTS5 + trigram; a platform-linked build must rely on
/// [`fts5_trigram_available`] instead.
pub const fn sqlite_bundled() -> bool {
    cfg!(feature = "bundled")
}

/// Whether the linked SQLite has FTS5 with the trigram tokenizer.
///
/// Probed on a throwaway in-memory connection — the same check the stdlib
/// engine's probe performs. Trivially true under the bundled default; genuinely
/// load-bearing when linked against a platform SQLite.
pub fn fts5_trigram_available() -> bool {
    let Ok(conn) = Connection::open_in_memory() else {
        return false;
    };
    conn.execute_batch("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='trigram')")
        .is_ok()
}

/// The derived-text store: the FTS5 trigram sidecar over note/recognized
/// text, backing the lexical search signals plus the recognition bookkeeping.
pub struct DerivedEngine {
    conn: Mutex<Connection>,
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
            Err(e) if is_busy(&e) && attempt < BUSY_RETRIES => {
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
        let conn = Connection::open(path).map_err(db_err)?;
        // Plain rollback journaling + synchronous=NORMAL. This store has exactly
        // one connection (the engine mutex serializes everything), so WAL's
        // concurrent-reader payoff never materializes — while its -wal/-shm
        // sidecars complicate the one-file story the relay/sync design wants.
        // DELETE keeps the store a single file between transactions; NORMAL may
        // lose the last transaction on power loss (never integrity), which a
        // rebuildable cache absorbs: the col_mod watermark lags, reads as drift,
        // rebuilds. (Opening a previously-WAL file converts it back; WAL is the
        // one persistent journal mode.)
        conn.pragma_update(None, "journal_mode", "DELETE")
            .map_err(db_err)?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(db_err)?;
        // "One connection" holds per ENGINE, but two engines can share the file
        // (the kernel's + the Python facade's read surface). With the default
        // busy_timeout of 0, a read overlapping a write transaction gets an
        // instant SQLITE_BUSY instead of waiting out a brief lock.
        conn.busy_timeout(std::time::Duration::from_secs(5))
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
        })
    }

    /// Drop the derived tables + the col_mod watermark (schema bump or
    /// integrity failure) — the next drift detection rebuilds from scratch.
    fn reset_tables(conn: &Connection) -> NativeResult<()> {
        for sql in [
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

    fn create_tables(conn: &Connection) -> NativeResult<()> {
        conn.execute(Self::IDX_DDL, []).map_err(db_err)?;
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
            // connection (the module-docs invariant; verified at open).
            ins_idx.execute([text]).map_err(db_err)?;
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
        // Atomic swap: prune dead-note rows, rename the shadow over the live
        // tables, stamp col_mod — all in ONE short transaction.
        self.swap_shadow_and_stamp(live_notes, col_mod)?;
        Ok(total)
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
            ins_idx.execute([text]).map_err(db_err)?;
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
    /// a partial. A crash before this leaves the live tables + old col_mod intact.
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
        tx.commit().map_err(db_err)
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
                    Err(e) if is_busy(&e) => return Err(e), // retried
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
                    Err(e) if is_busy(&e) => return Err(e),
                    Err(e) => {
                        return Ok(Err(NativeError::invalid_input(format!("fts5 match: {e}"))))
                    }
                }
            }
            Ok(Ok(rows))
        };
        with_busy_retry(run)?
    }

    /// [`Self::match_rows`] over many MATCH expressions, sharing ONE connection
    /// lock, ONE scope staging, and ONE compiled statement (through the
    /// connection's statement cache) across the whole batch. The fused-search
    /// lexical reads call this once for all query strings rather than locking,
    /// re-staging the scope, and recompiling the statement per query. Returns one
    /// row vector per expression, in `exprs` order.
    ///
    /// The scope (when present) is staged ONCE in a per-connection TEMP table and
    /// referenced as an invariant `IN (SELECT id FROM temp.…)` subquery, so the
    /// SQL text — hence the statement-cache key — is stable across the queries in
    /// the batch and across searches, keeping the cache warm. (The singular
    /// [`Self::match_rows`] inlines a small scope; the result set is identical.)
    ///
    /// # Errors
    ///
    /// Returns an error if scope staging fails, or any expression's MATCH query
    /// fails (the batch stops at the first failure, like the singular read).
    pub fn match_rows_batch(
        &self,
        exprs: &[String],
        limit: i64,
        with_text: bool,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Vec<Vec<MatchRow>>> {
        let span = tracing::debug_span!("derived.match_batch", n = exprs.len(), limit, with_text);
        let _enter = span.enter();
        let conn = self.lock();
        let txt_col = if with_text { "idx.txt" } else { "NULL" };
        // Stage the scope ONCE for the whole batch as an invariant subquery (a
        // large scope used to re-stage inside every per-query match_rows; the
        // staged form also keeps the SQL — the statement-cache key — stable).
        let scope_clause = match scope {
            Some(ids) if !ids.is_empty() => {
                Self::stage_id_set(&conn, "shrike_scope_ids", ids)?;
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
                        Err(e) if is_busy(&e) => return Err(e), // retried
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
                        Err(e) if is_busy(&e) => return Err(e),
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
    /// The fuzzy MATCH inputs for one query: its trigram set and the FTS5 OR
    /// expression over those trigrams. `None` when the query has fewer than
    /// [`FUZZY_MIN_SHARED`] trigram windows (too short to rank). Shared by the
    /// singular and batched fuzzy paths so they build the same expression.
    fn fuzzy_expr(query: &str) -> Option<(std::collections::BTreeSet<String>, String)> {
        let grams = trigrams(query.trim());
        if grams.len() < FUZZY_MIN_SHARED {
            return None;
        }
        let gram_set: std::collections::BTreeSet<String> = grams.into_iter().collect();
        let expr = gram_set
            .iter()
            .map(|g| fts_quote(g))
            .collect::<Vec<_>>()
            .join(" OR ");
        Some((gram_set, expr))
    }

    /// The fuzzy post-filter shared by the singular and batched paths: from FTS5's
    /// OR-matched rows (best-first, with text), keep one (best) row per note that
    /// shares at least [`FUZZY_MIN_SHARED`] trigrams with the query, capped at
    /// `top_k`. The min-overlap floor drops single-common-trigram noise that an
    /// FTS5 OR (matching ≥1 trigram) would otherwise surface.
    fn fuzzy_filter(
        rows: Vec<MatchRow>,
        gram_set: &std::collections::BTreeSet<String>,
        top_k: i64,
    ) -> Vec<LexicalRow> {
        let mut seen = std::collections::HashSet::new();
        let mut out: Vec<LexicalRow> = Vec::new();
        for (note_id, source, r, txt, snippet) in rows {
            let txt_grams: std::collections::BTreeSet<String> =
                trigrams(txt.as_deref().unwrap_or("")).into_iter().collect();
            if gram_set.intersection(&txt_grams).count() < FUZZY_MIN_SHARED {
                continue;
            }
            if !seen.insert(note_id) {
                continue; // dedup to one (best) row per note — rows arrive best-first
            }
            out.push((note_id, source, r, snippet));
            if out.len() as i64 >= top_k {
                break;
            }
        }
        out
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
        let q = query.trim();
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

    /// Notes sharing trigrams with `query` (typo/partial tolerant), best-first
    /// by FTS5 bm25, deduped to one (best) row per note, requiring at least
    /// [`FUZZY_MIN_SHARED`] shared trigrams (drops one-trigram noise). Empty
    /// when the query is too short to rank.
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
        let Some((gram_set, expr)) = Self::fuzzy_expr(query) else {
            return Ok(Vec::new());
        };
        let rows = self.match_rows(&expr, top_k * 4, true, scope, exclude_sources)?;
        Ok(Self::fuzzy_filter(rows, &gram_set, top_k))
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
            let trimmed = q.trim();
            if trimmed.chars().count() < MIN_TRIGRAM {
                served.push(false);
            } else {
                served.push(true);
                exprs.push(fts_quote(trimmed));
            }
        }
        let mut batched = self
            .match_rows_batch(&exprs, limit, false, scope, exclude_sources)?
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
        // Keep each servable query's trigram set for the post-filter; the
        // unservable ones reattach as empty results.
        let mut exprs: Vec<String> = Vec::new();
        let mut gram_sets: Vec<Option<std::collections::BTreeSet<String>>> =
            Vec::with_capacity(queries.len());
        for q in queries {
            match Self::fuzzy_expr(q) {
                Some((gram_set, expr)) => {
                    exprs.push(expr);
                    gram_sets.push(Some(gram_set));
                }
                None => gram_sets.push(None),
            }
        }
        let mut batched = self
            .match_rows_batch(&exprs, top_k * 4, true, scope, exclude_sources)?
            .into_iter();
        let out = gram_sets
            .into_iter()
            .map(|maybe| match maybe {
                // A served query draws its batched rows (in order) for the
                // post-filter; `map_or_else` keeps this panic-free.
                Some(gram_set) => batched
                    .next()
                    .map_or_else(Vec::new, |rows| Self::fuzzy_filter(rows, &gram_set, top_k)),
                None => Vec::new(),
            })
            .collect();
        Ok(out)
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
    fn open_converts_a_wal_store_back_to_single_file() {
        let (dir, path) = temp_db();
        {
            let raw = Connection::open(&path).unwrap();
            raw.pragma_update(None, "journal_mode", "WAL").unwrap();
        }
        {
            let _e = DerivedEngine::open(path.to_str().unwrap(), 1).unwrap();
        }
        // WAL is the one persistent journal mode — after open() the file is
        // back on rollback journaling and a fresh connection sees it.
        let raw = Connection::open(&path).unwrap();
        let mode: String = raw
            .query_row("PRAGMA journal_mode", [], |r| r.get(0))
            .unwrap();
        assert_eq!(mode.to_lowercase(), "delete");
        assert!(!path.with_extension("db-wal").exists());
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
