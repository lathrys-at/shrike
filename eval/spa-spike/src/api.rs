//! The actions-over-HTTP edge — STUBBED.
//!
//! #505 builds the real `/actions/{name}` edge (the UI surface, distinct from
//! MCP); the wire contract is "shrike-schemas verbatim". This spike hand-rolls
//! a tiny shim so the two screens are real without blocking on #505:
//!
//!   * [`list_notes`] POSTs to `/actions/list_notes` and deserializes the
//!     response straight into the canonical `ListNotesResponse` — ZERO codegen.
//!     That import is the evidence. (The real `/actions/search_notes` returns
//!     `SearchResponse`, an RRF-fused per-signal shape; the spike's search box
//!     filters client-side and reuses `ListNotesResponse` since the screen reads
//!     only the note list.)
//!   * [`status`] GETs `/status` and deserializes `ServerStatus`.
//!
//! The public fns return in-memory fixtures so the spike runs with no daemon;
//! the `live_*` helpers are the real gloo-net fetches (compiled, so the round
//! trip is proven to build to wasm32). Flipping the public fns to call them is
//! the whole swap — the deserialize target (a shrike-schemas type) does not move.

use serde::Serialize;
use shrike_schemas::{ListNotesResponse, Note, ServerStatus};

/// The actions edge takes/returns shrike-schemas types verbatim. We restate a
/// couple of request envelopes here only because the spike doesn't depend on
/// the Python side; in the real client these are imported too.
#[derive(Serialize)]
pub struct ListNotesRequest {
    pub limit: u32,
}

#[derive(Serialize)]
pub struct SearchNotesRequest {
    pub query: String,
    pub top_k: u32,
}

/// What the browser screen renders per row. In the real edge this is the
/// canonical `Note` (id/note_type/deck/tags/modified/content); we keep the full
/// type so the import is exercised end to end.
pub type NoteRow = Note;

// ----- transport -------------------------------------------------------------
//
// The public `list_notes`/`search_notes`/`status` below return in-memory
// fixtures so the spike runs headless with no daemon. The `live_*` helpers are
// the REAL actions-edge fetches — compiled (so the gloo-net + serde round trip
// is proven to build to wasm32) but not called by the fixtures. Flipping the
// public fns to call them is the whole swap; the deserialize target (a
// shrike-schemas type) does not move.

/// Real `POST /actions/list_notes`. Same-origin (the daemon serves the SPA
/// behind the transport guard), so no cross-origin/credential surface. The edge
/// returns the canonical `ListNotesResponse` ({notes, total, limit}) verbatim —
/// the actions body == the MCP tool's structuredContent (#505 strict parity).
#[allow(dead_code)]
pub async fn live_list_notes(req: ListNotesRequest) -> Result<ListNotesResponse, String> {
    use gloo_net::http::Request;
    Request::post("/actions/list_notes")
        .json(&req)
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<ListNotesResponse>()
        .await
        .map_err(|e| e.to_string())
}

/// Real `GET /status` — deserializes straight into the canonical `ServerStatus`.
#[allow(dead_code)]
pub async fn live_status() -> Result<ServerStatus, String> {
    use gloo_net::http::Request;
    Request::get("/status")
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<ServerStatus>()
        .await
        .map_err(|e| e.to_string())
}

/// Stubbed `/actions/list_notes`. Returns the canonical `ListNotesResponse`,
/// the same wrapper shape the real edge serializes (notes + total + limit).
pub async fn list_notes(req: ListNotesRequest) -> Result<ListNotesResponse, String> {
    let notes = fixture_notes();
    let total = notes.len() as i64;
    Ok(ListNotesResponse {
        notes,
        total,
        limit: req.limit as i64,
    })
}

/// Stubbed `/actions/search_notes`. The spike filters the fixtures client-side so
/// the search box is live and returns a `ListNotesResponse`; the REAL
/// `/actions/search_notes` returns `SearchResponse` (RRF-fused per-signal groups,
/// run server-side). The screen reads only the note list, so the simpler wrapper
/// is faithful for this evaluation.
pub async fn search_notes(req: SearchNotesRequest) -> Result<ListNotesResponse, String> {
    let q = req.query.to_lowercase();
    let notes: Vec<NoteRow> = fixture_notes()
        .into_iter()
        .filter(|n| {
            n.content
                .as_ref()
                .map(|c| c.values().any(|v| v.to_lowercase().contains(&q)))
                .unwrap_or(false)
        })
        .take(req.top_k as usize)
        .collect();
    let total = notes.len() as i64;
    Ok(ListNotesResponse {
        notes,
        total,
        limit: req.top_k as i64,
    })
}

/// Stubbed `GET /status`. Deserializes a fixture JSON into the canonical
/// `ServerStatus` — including the per-space modality `coverage` matrix
/// (#498/#235) the status pane renders.
pub async fn status() -> Result<ServerStatus, String> {
    serde_json::from_str::<ServerStatus>(STATUS_FIXTURE).map_err(|e| e.to_string())
}

/// The raw card HTML for a note (front+back rendered by the daemon). The real
/// edge returns the template-rendered card; the spike returns a small Anki-ish
/// snippet with CSS + a script so the sandbox behaviour is visible.
pub fn card_html(note: &NoteRow) -> String {
    let front = note
        .content
        .as_ref()
        .and_then(|c| c.values().next().cloned())
        .unwrap_or_default();
    format!(
        r#"<html><head><style>
          body {{ font-family: serif; padding: 1rem; }}
          .q {{ font-size: 1.4rem; }}
        </style></head><body>
          <div class="q">{front}</div>
          <hr id="answer">
          <div class="a">[back side]</div>
          <script>
            /* Card script runs (allow-scripts) but the frame has an opaque
               origin (NO allow-same-origin) so it cannot reach the host. */
            document.body.dataset.cardScriptRan = "true";
          </script>
        </body></html>"#
    )
}

// ----- fixtures --------------------------------------------------------------

fn fixture_notes() -> Vec<NoteRow> {
    fn note(id: i64, front: &str, back: &str, deck: &str) -> NoteRow {
        let mut content = std::collections::BTreeMap::new();
        content.insert("Front".to_string(), front.to_string());
        content.insert("Back".to_string(), back.to_string());
        Note {
            id,
            note_type: "Basic".to_string(),
            deck: deck.to_string(),
            tags: vec!["spike".to_string()],
            modified: "2026-06-12T00:00:00".to_string(),
            content: Some(content),
        }
    }
    vec![
        note(1, "What is the capital of France?", "Paris", "Geography"),
        note(
            2,
            "Define photosynthesis",
            "Conversion of light to chemical energy",
            "Biology",
        ),
        note(3, "<b>Newton's second law</b>", "F = ma", "Physics"),
    ]
}

const STATUS_FIXTURE: &str = r#"{
  "running": true,
  "wire_protocol_version": 1,
  "pid": 4242,
  "url": "http://127.0.0.1:8372/mcp",
  "collection": "/home/user/.local/share/Anki2/User 1/collection.anki2",
  "log_level": "INFO",
  "log_dir": "/home/user/.local/state/shrike/log",
  "uptime": "1:23:45",
  "embedding": {
    "state": "running",
    "available": true,
    "pid": 4243,
    "url": "http://127.0.0.1:8373",
    "model": "all-MiniLM-L6-v2",
    "modalities": ["text"]
  },
  "index": {
    "state": "ready",
    "available": true,
    "size": 1284,
    "ndim": 384
  },
  "derived": { "state": "ready", "available": true, "fts5": true, "size": 1284 },
  "recognition": { "state": "unavailable", "backend": null },
  "coverage": { "text": true, "image": false, "audio": false }
}"#;
