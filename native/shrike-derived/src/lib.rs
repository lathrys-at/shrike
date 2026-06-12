//! Native derived-text engine (#281): the FTS5-trigram store under the
//! `DerivedTextStore` facade, on rusqlite's **bundled** SQLite.
//!
//! The ONE implementation of the sidecar (`idx` FTS5 trigram + `rowmap`
//! provenance + `segments` + `gated` below-gate markers + `meta` in
//! `shrike.db`; the Python
//! `SqliteDerivedEngine` it originally mirrored retired with the engine
//! cutover). On a schema-version mismatch — or an `idx`↔`rowmap` pairing
//! inconsistency found at open — the derived data is dropped and the next
//! drift rebuilds: the derived-cache answer, no migrations. The bundled
//! SQLite always has FTS5 + the trigram tokenizer, which is this engine's
//! user-facing win: the facade's availability probe stops being load-bearing.
//!
//! **The single mutexed connection is a correctness invariant, not just
//! thread-safety** (#396): `idx` is an FTS5 virtual table, so nothing at the
//! schema level (no FK, no trigger) ties its rowids to `rowmap` — the pairing
//! holds because every write rides this one connection's
//! `last_insert_rowid()` under [`DerivedEngine::lock`]. A future move to a
//! connection pool must make the coupling structural first.
//!
//! MATCH-expression building, trigram filtering, and the state machine stay
//! facade-side; this crate is storage + queries only. Pure Rust — no pyo3.

use std::sync::Mutex;

use rusqlite::Connection;
use shrike_ffi::{NativeError, NativeResult};

/// Mirrors `shrike.derived.SNIPPET_TOKENS` (the facade doesn't pass it — it's
/// part of the pinned engine behaviour).
const SNIPPET_TOKENS: i64 = 12;

/// (note_id, source, ref, txt, snippet) — one MATCH result row.
pub type MatchRow = (i64, String, String, Option<String>, Option<String>);

/// Whether this build statically links rusqlite's bundled SQLite (#300).
/// Bundled guarantees FTS5 + trigram; a platform-linked build must rely on
/// [`fts5_trigram_available`] instead.
pub const fn sqlite_bundled() -> bool {
    cfg!(feature = "bundled")
}

/// Whether the linked SQLite has FTS5 with the trigram tokenizer (#300).
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

pub struct DerivedEngine {
    conn: Mutex<Connection>,
}

fn db_err(e: rusqlite::Error) -> NativeError {
    NativeError::unavailable(format!("sqlite: {e}"))
}

impl DerivedEngine {
    /// Open (or create) the store and ensure the schema, resetting the derived
    /// data on a schema-version mismatch (no migrations — it's a rebuildable
    /// cache). Errors are `unavailable` — the facade recovers by discarding.
    /// The current sidecar schema. v2 (#228): the `segments` table for
    /// recognition structure (boxes/spans) + recognition meta keys. A bump
    /// drops everything — recognition rows re-derive via the pending sweep.
    pub const SCHEMA_VERSION: i64 = 2;

    pub fn open(path: &str, schema_version: i64) -> NativeResult<Self> {
        let conn = Connection::open(path).map_err(db_err)?;
        // Plain rollback journaling + synchronous=NORMAL (#396). WAL was the
        // original mode, but this store has exactly one connection (the
        // engine mutex serializes everything), so WAL's concurrent-reader
        // payoff never materializes — while its -wal/-shm sidecars complicate
        // the one-file story the relay/sync design wants. DELETE keeps the
        // store a single file between transactions; NORMAL may lose the last
        // transaction on power loss (never integrity), which a rebuildable
        // cache absorbs: the col_mod watermark lags, reads as drift, rebuilds.
        // (Opening a previously-WAL file converts it back; WAL is the one
        // persistent journal mode.)
        conn.pragma_update(None, "journal_mode", "DELETE")
            .map_err(db_err)?;
        conn.pragma_update(None, "synchronous", "NORMAL")
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
        // corruption: drop the derived data so the next drift rebuild
        // restores a consistent store (recognition rows re-derive via the
        // pending sweep). A silent mismatch would serve provenance for the
        // wrong notes.
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

    fn create_tables(conn: &Connection) -> NativeResult<()> {
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS idx USING fts5(txt, tokenize='trigram')",
            [],
        )
        .map_err(db_err)?;
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
        // Below-gate recognition markers (#416): a (note, source, ref) the
        // recognizer judged and the gate dropped — no text row exists, but
        // the pending sweep must count it DONE (or it re-recognizes forever).
        // Invalidated with the recognized rows on a recognizer-fingerprint
        // change ([`Self::clear_gated`]). IF NOT EXISTS + create_tables-on-open
        // retrofits existing stores, like the rowmap_source index — no schema
        // bump (an absent/empty table just means nothing is marked yet).
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
        // rowmap, and texts_for_source(OCR) sits on every upsert's tail
        // (#445). IF NOT EXISTS + create_tables-on-open retrofits existing
        // stores.
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

    pub fn get_col_mod(&self) -> Option<i64> {
        let conn = self.lock();
        conn.query_row("SELECT value FROM meta WHERE key='col_mod'", [], |r| {
            r.get(0)
        })
        .ok()
    }

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
    /// to 0 — a locked/corrupt store must not read as an empty one (#396).
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
        for (reference, text) in refs_text {
            if text.trim().is_empty() {
                continue;
            }
            // The idx→rowmap pairing rides last_insert_rowid() on THIS
            // connection — sound only under the engine's single mutexed
            // connection (the module-docs invariant; verified at open).
            conn.execute("INSERT INTO idx(txt) VALUES(?1)", [text])
                .map_err(db_err)?;
            let rowid = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO rowmap(rowid, note_id, source, ref) VALUES(?1,?2,?3,?4)",
                rusqlite::params![rowid, note_id, source, reference],
            )
            .map_err(db_err)?;
        }
        Ok(())
    }

    /// Replace a note's text rows for one source (incremental upsert), in one
    /// transaction.
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

    /// Replace MANY notes' text rows for one source in ONE transaction
    /// (#445): callers hold whole upsert batches, and one-commit-per-note
    /// under DELETE journaling is journal create+fsync+delete per note. The
    /// delete half batches across all ids; inserts pair idx↔rowmap per row
    /// exactly like `ingest`.
    pub fn ingest_many(
        &self,
        notes: &[(i64, Vec<(String, String)>)],
        source: &str,
    ) -> NativeResult<()> {
        if notes.is_empty() {
            return Ok(());
        }
        // Duplicate ids: LAST entry wins — matching the per-note loop this
        // replaced (each occurrence delete+inserted, so the final one stood).
        // The batch deletes each id's rows ONCE up front, so without this
        // guard a duplicate would double-insert (#447 review).
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
    /// The rebuild is **collection-derived-sources-scoped** (#228): it
    /// replaces `field` rows (cheap — re-read from the collection) but
    /// PRESERVES recognition-derived rows (`ocr`/`asr` — expensive, with
    /// their own fingerprint-keyed invalidation), so a boot-drift rebuild
    /// never forces re-recognition. Recognition rows whose note vanished
    /// from the new row set are pruned (the note was deleted).
    pub fn build(&self, rows: &[(i64, String, String, String)], col_mod: i64) -> NativeResult<()> {
        let mut conn = self.lock();
        let tx = conn.transaction().map_err(db_err)?;
        tx.execute(
            "DELETE FROM idx WHERE rowid IN (SELECT rowid FROM rowmap WHERE source = 'field')",
            [],
        )
        .map_err(db_err)?;
        tx.execute("DELETE FROM rowmap WHERE source = 'field'", [])
            .map_err(db_err)?;
        for (note_id, source, reference, text) in rows {
            if text.trim().is_empty() {
                continue;
            }
            tx.execute("INSERT INTO idx(txt) VALUES(?1)", [text])
                .map_err(db_err)?;
            let rowid = tx.last_insert_rowid();
            tx.execute(
                "INSERT INTO rowmap(rowid, note_id, source, ref) VALUES(?1,?2,?3,?4)",
                rusqlite::params![rowid, note_id, source, reference],
            )
            .map_err(db_err)?;
        }
        // Prune recognition rows (and their segments, and below-gate markers)
        // for notes no longer in the collection: the build rows are the
        // authoritative note set.
        let live: std::collections::HashSet<i64> = rows.iter().map(|r| r.0).collect();
        let stale: Vec<i64> = {
            let mut stmt = tx
                .prepare(
                    "SELECT DISTINCT note_id FROM rowmap WHERE source != 'field' \
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
            Self::delete_rows(&tx, &stale, None)?;
            Self::delete_gated(&tx, &stale, None)?;
        }
        tx.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('col_mod', ?1)",
            [col_mod],
        )
        .map_err(db_err)?;
        tx.commit().map_err(db_err)
    }

    /// A free-form meta value (e.g. the recognizer fingerprint, #228).
    pub fn meta_get(&self, key: &str) -> NativeResult<Option<String>> {
        let conn = self.lock();
        Ok(conn
            .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| {
                r.get::<_, String>(0)
            })
            .ok())
    }

    pub fn meta_set(&self, key: &str, value: &str) -> NativeResult<()> {
        let conn = self.lock();
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?1, ?2)",
            rusqlite::params![key, value],
        )
        .map_err(db_err)?;
        Ok(())
    }

    /// Store one item's recognition structure (#228: segments JSON — boxes
    /// for OCR, time spans for ASR) alongside its text row, keyed like the
    /// row. One pass, many consumers: #230 (occlusion) reads these back.
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
    /// has already been recognized" set (#228). Deliberately a full-set read:
    /// the sweep's pending diff needs the complete set, and the pairs are
    /// small. Bounding belongs to the sweep's batching, not this query.
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

    /// Record below-gate outcomes (#416): each (note_id, ref) was recognized
    /// and the gate dropped it — no text row, but the pending sweep counts it
    /// done. One transaction per batch.
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
    /// [`Self::refs_for_source`] by the pending sweep's done-set diff (#416).
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
    /// invalidation path (#416): a new engine re-judges everything, gated
    /// items included, exactly like stored rows re-derive.
    pub fn clear_gated(&self, source: &str) -> NativeResult<()> {
        let conn = self.lock();
        conn.execute("DELETE FROM gated WHERE source = ?1", [source])
            .map_err(db_err)?;
        Ok(())
    }

    /// All (note_id, ref, text) rows for one source — the embed-input
    /// composition reads recognized text back for vector minting (#199).
    /// Deliberately a full-set read (the composition consumes the whole set);
    /// volume is bounded by recognized-text size, not media size.
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

    /// `texts_for_source` scoped to a note set (#445): the per-upsert embed
    /// composition needs only the WRITTEN notes' recognized texts — the
    /// full-set read belongs to rebuild/reconcile, not the op tail.
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
    /// match to those note ids INSIDE the query (the #177 scoped-search path:
    /// the id set comes from anki's indexed deck:/tag: search, so scoped
    /// literal search needs no over-fetch and no post-hoc recall gamble).
    pub fn match_rows(
        &self,
        expr: &str,
        limit: i64,
        with_text: bool,
        scope: Option<&[i64]>,
    ) -> NativeResult<Vec<MatchRow>> {
        let span = tracing::debug_span!("derived.match", limit, with_text);
        let _enter = span.enter();
        let conn = self.lock();
        let txt_col = if with_text { "idx.txt" } else { "NULL" };
        // Small scopes inline as literals (i64s — no injection surface);
        // large ones are staged in a TEMP table, since SQLite's parser caps
        // long expression lists (#396).
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
        let sql = format!(
            "SELECT m.note_id, m.source, m.ref, {txt_col}, \
             snippet(idx, 0, '', '', '…', ?1) \
             FROM idx JOIN rowmap m ON m.rowid = idx.rowid \
             WHERE idx MATCH ?2 {scope_clause}ORDER BY rank LIMIT ?3"
        );
        let mut stmt = conn.prepare(&sql).map_err(db_err)?;
        let rows = stmt
            .query_map(rusqlite::params![SNIPPET_TOKENS, expr, limit], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
            })
            .map_err(|e| NativeError::invalid_input(format!("fts5 match: {e}")))?
            .collect::<Result<Vec<MatchRow>, _>>()
            .map_err(|e| NativeError::invalid_input(format!("fts5 match: {e}")))?;
        Ok(rows)
    }
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

    #[test]
    fn scoped_match_restricts_to_the_id_set() {
        // The #177 scoped-search path: the id set rides INSIDE the FTS5
        // query, so a scoped literal/fuzzy search has exact recall within
        // scope and zero hits outside it.
        let (e, _dir) = store();
        e.build(
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
        let all = e.search_substring("krebs", 10, None).unwrap().unwrap();
        let ids: Vec<i64> = all.iter().map(|r| r.0).collect();
        assert!(ids.contains(&1) && ids.contains(&2));

        // Scoped to note 2 only.
        let scoped = e
            .search_substring("krebs", 10, Some(&[2]))
            .unwrap()
            .unwrap();
        assert_eq!(scoped.iter().map(|r| r.0).collect::<Vec<_>>(), vec![2]);

        // An empty scope matches nothing (never falls open).
        let none = e.search_substring("krebs", 10, Some(&[])).unwrap().unwrap();
        assert!(none.is_empty());

        // Fuzzy honors the same scope.
        let fz = e.search_fuzzy("kreps cycle", 10, Some(&[1])).unwrap();
        assert_eq!(fz.iter().map(|r| r.0).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn probe_reports_linkage_capability() {
        // Under the bundled default the probe MUST pass (the #281 guarantee);
        // under platform linkage it reports whatever the host library has —
        // on this dev host the test only runs if the store above worked, so
        // the probe must agree.
        assert!(fts5_trigram_available());
        if sqlite_bundled() {
            assert!(fts5_trigram_available());
        }
    }

    #[test]
    fn ingest_many_matches_per_note_ingest() {
        // One-transaction batch ingest (#445) is behavior-identical to the
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
            .search_substring("new text one", 10, None)
            .unwrap()
            .unwrap();
        assert_eq!(hits.iter().map(|r| r.0).collect::<Vec<_>>(), vec![1]);
        let old = e
            .search_substring("old text one", 10, None)
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
        // both new APIs (#447 review).
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
            .search_substring("second version", 10, None)
            .unwrap()
            .unwrap();
        assert_eq!(hits.iter().map(|r| r.0).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn build_ingest_remove_count_round_trip() {
        let (e, dir) = store();
        e.build(
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
        let hits = e.match_rows("\"chloroplast\"", 10, false, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1);
        assert_eq!(hits[0].1, "field");
        assert_eq!(hits[0].2, "Front");

        e.remove(&[1], None).unwrap();
        assert_eq!(e.count().unwrap(), 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn match_returns_snippet_and_text_when_asked() {
        let (e, dir) = store();
        e.build(
            &[(7, "field".into(), "F".into(), "alpha beta gamma".into())],
            1,
        )
        .unwrap();
        let rows = e.match_rows("\"beta\"", 10, true, None).unwrap();
        assert_eq!(rows[0].3.as_deref(), Some("alpha beta gamma"));
        assert!(rows[0].4.as_deref().unwrap().contains("beta"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn bad_match_expression_is_invalid_input() {
        let (e, dir) = store();
        e.build(&[(1, "field".into(), "F".into(), "abc".into())], 1)
            .unwrap();
        let err = e.match_rows("AND AND (", 10, false, None).unwrap_err();
        assert_eq!(err.kind, shrike_ffi::ErrorKind::InvalidInput);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rebuild_preserves_recognition_rows_and_prunes_orphans() {
        // #228: a drift rebuild replaces `field` rows but never discards
        // recognition-derived rows (re-recognition is expensive) — except for
        // notes that vanished from the collection.
        let (e, _dir) = store();
        e.build(
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
        e.build(
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
            .search_substring("electron transport", 10, None)
            .unwrap()
            .unwrap();
        assert_eq!(hits[0].0, 1);
        assert_eq!(hits[0].1, "ocr");
        let field_hits = e.search_substring("EDITED", 10, None).unwrap().unwrap();
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

    #[test]
    fn gated_markers_persist_survive_ingest_and_invalidate() {
        // #416: below-gate markers round-trip, survive a sibling image's
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
        e.build(
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
        e.build(&[(1, "field".into(), "F".into(), "abc".into())], 9)
            .unwrap();
        drop(e);
        let path = dir.join("shrike.db");
        let e2 = DerivedEngine::open(path.to_str().unwrap(), 2).unwrap();
        assert_eq!(e2.count().unwrap(), 0);
        assert_eq!(e2.get_col_mod(), None);
        std::fs::remove_dir_all(dir).ok();
    }
}

// ── lexical search policy (#331: re-homed from the Python facade) ────────────
// The MATCH-expression building + result filtering that used to live in
// `shrike/derived.py`. One implementation: the Python facade delegates here
// through the binding, and the kernel's search assembly calls it directly.

/// FTS5's trigram tokenizer can't match a term shorter than 3 chars.
pub const MIN_TRIGRAM: usize = 3;
/// A fuzzy candidate must share at least this many query trigrams (noise floor).
pub const FUZZY_MIN_SHARED: usize = 2;

/// One lexical hit with its provenance: `(note_id, source, ref, snippet)`.
pub type LexicalRow = (i64, String, String, Option<String>);

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
    /// Notes whose derived text literally contains `query` (case-insensitive),
    /// with `(source, ref, snippet)` provenance. `None` tells the caller to use
    /// the `find_notes` fallback (query shorter than a trigram); a MATCH error
    /// is a real error (the caller decides whether to degrade).
    pub fn search_substring(
        &self,
        query: &str,
        limit: i64,
        scope: Option<&[i64]>,
    ) -> NativeResult<Option<Vec<LexicalRow>>> {
        let q = query.trim();
        if q.chars().count() < MIN_TRIGRAM {
            return Ok(None);
        }
        // A quoted phrase → contiguous (literal substring) match.
        let rows = self.match_rows(&fts_quote(q), limit, false, scope)?;
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
    pub fn search_fuzzy(
        &self,
        query: &str,
        top_k: i64,
        scope: Option<&[i64]>,
    ) -> NativeResult<Vec<LexicalRow>> {
        let grams = trigrams(query.trim());
        if grams.len() < FUZZY_MIN_SHARED {
            return Ok(Vec::new());
        }
        let gram_set: std::collections::BTreeSet<String> = grams.into_iter().collect();
        let expr: Vec<String> = gram_set.iter().map(|g| fts_quote(g)).collect();
        let rows = self.match_rows(&expr.join(" OR "), top_k * 4, true, scope)?;
        let mut seen = std::collections::HashSet::new();
        let mut out: Vec<LexicalRow> = Vec::new();
        for (note_id, source, r, txt, snippet) in rows {
            // Min-overlap floor: FTS5 OR matches >= 1 trigram; require a few
            // shared so a single common gram doesn't surface noise.
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
        e.build(
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
            .search_substring("mitochondria", 10, None)
            .unwrap()
            .unwrap();
        assert_eq!(hits[0].0, 1);
        assert!(e.search_substring("mi", 10, None).unwrap().is_none()); // sub-trigram → fallback
        assert!(e
            .search_substring("q\"uo", 10, None)
            .unwrap()
            .unwrap()
            .is_empty()); // quotes safe
    }

    #[test]
    fn fuzzy_ranks_typos_and_floors_noise() {
        let e = store();
        let hits = e.search_fuzzy("mitochondira", 10, None).unwrap(); // transposition
        assert!(hits.iter().any(|(nid, ..)| *nid == 1));
        assert!(e.search_fuzzy("xy", 10, None).unwrap().is_empty()); // too short to rank
    }
}

#[cfg(test)]
mod hardening_tests {
    //! #396: open-time integrity, fallible count, journal-mode policy, and
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
            e.build(
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
        assert_eq!(err.kind, shrike_ffi::ErrorKind::Unavailable);
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
        e.build(&rows, 100).unwrap();
        assert_eq!(e.count().unwrap(), n);

        // A scope wider than the inline cap rides the staged TEMP table and
        // still restricts correctly.
        let scope: Vec<i64> = (1..=n).collect();
        let hits = e.match_rows("\"shared\"", 10, false, Some(&scope)).unwrap();
        assert!(!hits.is_empty());
        let narrow = e.match_rows("\"shared\"", 10, false, Some(&[2])).unwrap();
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
