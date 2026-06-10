//! Native derived-text engine (#281): the FTS5-trigram store under the
//! `DerivedTextStore` facade, on rusqlite's **bundled** SQLite.
//!
//! Implements exactly the surface of `shrike.derived.SqliteDerivedEngine` —
//! identical schema (`idx` FTS5 trigram + `rowmap` provenance + `meta`),
//! identical SQL, the same `shrike.db` file (either engine opens a store the
//! other wrote; on a schema-version mismatch the data is dropped and the next
//! drift rebuilds, the derived-cache answer). The bundled SQLite always has
//! FTS5 + the trigram tokenizer, which is this engine's user-facing win: the
//! facade's availability probe stops being load-bearing.
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
    pub fn open(path: &str, schema_version: i64) -> NativeResult<Self> {
        let conn = Connection::open(path).map_err(db_err)?;
        conn.pragma_update(None, "journal_mode", "WAL").map_err(db_err)?;

        conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value)", [])
            .map_err(db_err)?;
        let version: Option<i64> = conn
            .query_row("SELECT value FROM meta WHERE key='schema_version'", [], |r| r.get(0))
            .ok();
        if let Some(v) = version {
            if v != schema_version {
                conn.execute("DROP TABLE IF EXISTS idx", []).map_err(db_err)?;
                conn.execute("DROP TABLE IF EXISTS rowmap", []).map_err(db_err)?;
                conn.execute("DELETE FROM meta WHERE key='col_mod'", []).map_err(db_err)?;
            }
        }
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
        conn.execute("CREATE INDEX IF NOT EXISTS rowmap_note ON rowmap(note_id, source)", [])
            .map_err(db_err)?;
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?1)",
            [schema_version],
        )
        .map_err(db_err)?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().expect("derived conn lock poisoned")
    }

    pub fn get_col_mod(&self) -> Option<i64> {
        let conn = self.lock();
        conn.query_row("SELECT value FROM meta WHERE key='col_mod'", [], |r| r.get(0)).ok()
    }

    pub fn set_col_mod(&self, value: i64) -> NativeResult<()> {
        let conn = self.lock();
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('col_mod', ?1)", [value])
            .map_err(db_err)?;
        Ok(())
    }

    pub fn count(&self) -> i64 {
        let conn = self.lock();
        conn.query_row("SELECT count(*) FROM rowmap", [], |r| r.get(0)).unwrap_or(0)
    }

    fn delete_rows(
        conn: &Connection,
        note_ids: &[i64],
        source: Option<&str>,
    ) -> NativeResult<()> {
        if note_ids.is_empty() {
            return Ok(());
        }
        let marks = vec!["?"; note_ids.len()].join(",");
        let mut clause = format!("note_id IN ({marks})");
        if source.is_some() {
            clause.push_str(" AND source=?");
        }
        let mut stmt = conn
            .prepare(&format!("SELECT rowid FROM rowmap WHERE {clause}"))
            .map_err(db_err)?;
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = note_ids
            .iter()
            .map(|n| Box::new(*n) as Box<dyn rusqlite::ToSql>)
            .collect();
        if let Some(s) = source {
            params.push(Box::new(s.to_string()));
        }
        let rowids: Vec<i64> = stmt
            .query_map(rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())), |r| {
                r.get(0)
            })
            .map_err(db_err)?
            .collect::<Result<_, _>>()
            .map_err(db_err)?;
        if rowids.is_empty() {
            return Ok(());
        }
        let rmarks = vec!["?"; rowids.len()].join(",");
        conn.execute(
            &format!("DELETE FROM idx WHERE rowid IN ({rmarks})"),
            rusqlite::params_from_iter(rowids.iter()),
        )
        .map_err(db_err)?;
        conn.execute(
            &format!("DELETE FROM rowmap WHERE rowid IN ({rmarks})"),
            rusqlite::params_from_iter(rowids.iter()),
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
            conn.execute("INSERT INTO idx(txt) VALUES(?1)", [text]).map_err(db_err)?;
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

    /// Drop notes' rows (all sources, or just one), in one transaction.
    pub fn remove(&self, note_ids: &[i64], source: Option<&str>) -> NativeResult<()> {
        let mut conn = self.lock();
        let tx = conn.transaction().map_err(db_err)?;
        Self::delete_rows(&tx, note_ids, source)?;
        tx.commit().map_err(db_err)
    }

    /// Full (re)build from (note_id, source, ref, text) rows; stamps col_mod.
    /// One transaction — a failure rolls everything back.
    pub fn build(
        &self,
        rows: &[(i64, String, String, String)],
        col_mod: i64,
    ) -> NativeResult<()> {
        let mut conn = self.lock();
        let tx = conn.transaction().map_err(db_err)?;
        tx.execute("DELETE FROM idx", []).map_err(db_err)?;
        tx.execute("DELETE FROM rowmap", []).map_err(db_err)?;
        for (note_id, source, reference, text) in rows {
            if text.trim().is_empty() {
                continue;
            }
            tx.execute("INSERT INTO idx(txt) VALUES(?1)", [text]).map_err(db_err)?;
            let rowid = tx.last_insert_rowid();
            tx.execute(
                "INSERT INTO rowmap(rowid, note_id, source, ref) VALUES(?1,?2,?3,?4)",
                rusqlite::params![rowid, note_id, source, reference],
            )
            .map_err(db_err)?;
        }
        tx.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('col_mod', ?1)", [col_mod])
            .map_err(db_err)?;
        tx.commit().map_err(db_err)
    }

    /// One FTS5 MATCH (rank-ordered), returning provenance + snippet rows.
    /// A bad expression is `invalid_input` — the facade maps it to its
    /// OperationalError fallback path.
    pub fn match_rows(
        &self,
        expr: &str,
        limit: i64,
        with_text: bool,
    ) -> NativeResult<Vec<MatchRow>> {
        let conn = self.lock();
        let txt_col = if with_text { "idx.txt" } else { "NULL" };
        let sql = format!(
            "SELECT m.note_id, m.source, m.ref, {txt_col}, \
             snippet(idx, 0, '', '', '…', ?1) \
             FROM idx JOIN rowmap m ON m.rowid = idx.rowid \
             WHERE idx MATCH ?2 ORDER BY rank LIMIT ?3"
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
    fn build_ingest_remove_count_round_trip() {
        let (e, dir) = store();
        e.build(
            &[
                (1, "field".into(), "Front".into(), "the mitochondria".into()),
                (2, "field".into(), "Front".into(), "powerhouse of the cell".into()),
                (3, "field".into(), "Back".into(), "   ".into()), // blank → skipped
            ],
            100,
        )
        .unwrap();
        assert_eq!(e.count(), 2);
        assert_eq!(e.get_col_mod(), Some(100));

        e.ingest(1, "field", &[("Front".into(), "the chloroplast".into())]).unwrap();
        assert_eq!(e.count(), 2);
        let hits = e.match_rows("\"chloroplast\"", 10, false).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1);
        assert_eq!(hits[0].1, "field");
        assert_eq!(hits[0].2, "Front");

        e.remove(&[1], None).unwrap();
        assert_eq!(e.count(), 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn match_returns_snippet_and_text_when_asked() {
        let (e, dir) = store();
        e.build(&[(7, "field".into(), "F".into(), "alpha beta gamma".into())], 1).unwrap();
        let rows = e.match_rows("\"beta\"", 10, true).unwrap();
        assert_eq!(rows[0].3.as_deref(), Some("alpha beta gamma"));
        assert!(rows[0].4.as_deref().unwrap().contains("beta"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn bad_match_expression_is_invalid_input() {
        let (e, dir) = store();
        e.build(&[(1, "field".into(), "F".into(), "abc".into())], 1).unwrap();
        let err = e.match_rows("AND AND (", 10, false).unwrap_err();
        assert_eq!(err.kind, shrike_ffi::ErrorKind::InvalidInput);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn schema_version_bump_resets_data() {
        let (e, dir) = store();
        e.build(&[(1, "field".into(), "F".into(), "abc".into())], 9).unwrap();
        drop(e);
        let path = dir.join("shrike.db");
        let e2 = DerivedEngine::open(path.to_str().unwrap(), 2).unwrap();
        assert_eq!(e2.count(), 0);
        assert_eq!(e2.get_col_mod(), None);
        std::fs::remove_dir_all(dir).ok();
    }
}
