//! Screen 1 — collection browser.
//!
//! Demonstrates: list/search over the (stubbed) actions edge with canonical
//! `shrike_schemas::Note` rows, and the card preview rendered in a SANDBOXED
//! iframe via `srcdoc` (NOT same-origin) so Anki template CSS/JS/MathJax runs
//! isolated from the host. The frame is `sandbox="allow-scripts"` only.

use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api::{self, NoteRow, SearchNotesRequest};

#[component]
pub fn CollectionBrowser() -> impl IntoView {
    let (query, set_query) = signal(String::new());
    let (rows, set_rows) = signal(Vec::<NoteRow>::new());
    let (selected, set_selected) = signal(None::<NoteRow>);

    // Initial load + on every query change, hit the (stubbed) actions edge.
    let run = move || {
        let q = query.get();
        spawn_local(async move {
            let result = if q.trim().is_empty() {
                api::list_notes(api::ListNotesRequest { limit: 50 }).await
            } else {
                api::search_notes(SearchNotesRequest {
                    query: q,
                    top_k: 50,
                })
                .await
            };
            // The actions edge returns the canonical ListNotesResponse; the
            // screen reads its `notes` list.
            if let Ok(response) = result {
                set_rows.set(response.notes);
            }
        });
    };
    // Kick the first load.
    Effect::new(move |_| run());

    view! {
        <h2>"Collection browser"</h2>
        <input
            placeholder="Search notes…"
            on:input=move |ev| set_query.set(event_target_value(&ev))
            prop:value=query
        />
        <button on:click=move |_| run()>"Search"</button>

        <div style="display:flex; gap:1rem;">
            <div style="flex:1;">
                <For
                    each=move || rows.get()
                    key=|n| n.id
                    children=move |note: NoteRow| {
                        let n = note.clone();
                        let label = note
                            .content
                            .as_ref()
                            .and_then(|c| c.values().next().cloned())
                            .unwrap_or_else(|| format!("#{}", note.id));
                        view! {
                            <div class="row" on:click=move |_| set_selected.set(Some(n.clone()))>
                                <strong>{label}</strong>
                                " · " {note.deck.clone()}
                            </div>
                        }
                    }
                />
            </div>
            <div style="flex:1;">
                <h3>"Card preview (sandboxed)"</h3>
                {move || match selected.get() {
                    Some(note) => view! { <CardFrame note=note /> }.into_any(),
                    None => view! { <p>"Select a note."</p> }.into_any(),
                }}
            </div>
        </div>
    }
}

/// The card render frame — the one security-critical surface in the client.
///
/// SECURITY-CRITICAL LINE: `sandbox="allow-scripts"` and DELIBERATELY NOT
/// `allow-same-origin`. Card HTML is untrusted (Anki templates carry arbitrary
/// CSS/JS/MathJax from shared decks). With both flags the sandbox is defeated —
/// the frame would share the host origin and card JS could read the daemon's
/// DOM/storage. `allow-scripts` alone gives the frame a unique opaque origin:
/// scripts run, but the frame cannot script the host. `srcdoc` (not `src`)
/// keeps the bytes off any URL/navigation.
///
/// The spike does not wire a `postMessage` listener (so there is no inbound
/// message surface). When the real client adds one — for height-fit / link
/// interception — it MUST validate `event.source` and treat `event.origin` as
/// untrusted: a sandboxed opaque-origin frame posts with `origin: "null"`, so
/// the host listens only for messages from THIS frame's `contentWindow` and
/// never trusts the payload as a capability. The host-page CSP (see
/// `tauri.conf.json` `security.csp`) is the second layer under Tauri v2.
#[component]
fn CardFrame(note: NoteRow) -> impl IntoView {
    let html = api::card_html(&note);
    view! {
        // SECURITY-CRITICAL: allow-scripts WITHOUT allow-same-origin (see above).
        <iframe
            class="card"
            sandbox="allow-scripts"
            srcdoc=html
        ></iframe>
    }
}
