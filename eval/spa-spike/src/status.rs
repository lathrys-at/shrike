//! Screen 2 — server status pane.
//!
//! Demonstrates: deserializing the canonical `shrike_schemas::ServerStatus`
//! from `/status` (stubbed) and rendering the live state plus the per-space
//! modality coverage matrix (#498/#235). Every field read here is a canonical
//! type field — no hand-mirrored DTO, no codegen.

use leptos::prelude::*;
use leptos::task::spawn_local;
use shrike_schemas::{EmbeddingStatus, IndexStatus, ServerStatus};

#[component]
pub fn StatusPane() -> impl IntoView {
    let (status, set_status) = signal(None::<ServerStatus>);

    Effect::new(move |_| {
        spawn_local(async move {
            if let Ok(s) = crate::api::status().await {
                set_status.set(Some(s));
            }
        });
    });

    view! {
        <h2>"Server status"</h2>
        {move || match status.get() {
            None => view! { <p>"Loading…"</p> }.into_any(),
            Some(s) => view! { <StatusBody status=s /> }.into_any(),
        }}
    }
}

#[component]
fn StatusBody(status: ServerStatus) -> impl IntoView {
    let embedding = describe_embedding(&status.embedding);
    let index = describe_index(&status.index);
    let coverage = status.coverage.clone().unwrap_or_default();

    view! {
        <table>
            <tr><td>"PID"</td><td>{status.pid}</td></tr>
            <tr><td>"URL"</td><td>{status.url.clone()}</td></tr>
            <tr><td>"Collection"</td><td>{status.collection.clone()}</td></tr>
            <tr><td>"Wire protocol"</td><td>{status.wire_protocol_version}</td></tr>
            <tr><td>"Embedding"</td><td>{embedding}</td></tr>
            <tr><td>"Index"</td><td>{index}</td></tr>
            <tr>
                <td>"Derived (FTS5)"</td>
                <td>{format!("{:?} · {} rows", status.derived.state, status.derived.size)}</td>
            </tr>
        </table>

        <h3>"Modality coverage (#498/#235)"</h3>
        <table class="matrix">
            <For
                each=move || {
                    let mut entries: Vec<(String, bool)> =
                        coverage.iter().map(|(k, v)| (k.clone(), *v)).collect();
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                    entries
                }
                key=|(k, _)| k.clone()
                children=|(modality, served)| {
                    let (cls, mark) = if served { ("ok", "✓") } else { ("no", "✗") };
                    view! {
                        <tr>
                            <td>{modality}</td>
                            <td class=cls>{mark}</td>
                        </tr>
                    }
                }
            />
        </table>
    }
}

// The canonical enums are internally-tagged STRUCT variants (the import the
// spike is proving) — matched by field, not by a wrapped payload. This is the
// shape a TS codegen path mangles into a non-discriminated bag; here it is the
// type, verbatim.
fn describe_embedding(e: &EmbeddingStatus) -> String {
    match e {
        EmbeddingStatus::Running {
            model, modalities, ..
        } => format!(
            "running · {} · {:?}",
            model.clone().unwrap_or_else(|| "?".into()),
            modalities.clone().unwrap_or_default()
        ),
        EmbeddingStatus::Stopped { .. } => "stopped".into(),
        EmbeddingStatus::Failed { .. } => "failed".into(),
        EmbeddingStatus::NotConfigured { .. } => "not configured".into(),
    }
}

fn describe_index(i: &IndexStatus) -> String {
    match i {
        IndexStatus::Ready { base } => format!("ready · {} vectors", base.size),
        IndexStatus::Building { progress, .. } => {
            format!("building · {}/{}", progress.indexed, progress.total)
        }
        IndexStatus::Unavailable { .. } => "unavailable".into(),
        IndexStatus::Error { error, .. } => format!("error · {}", error),
    }
}
