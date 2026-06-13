//! Trunk/CSR entry point: mount the Leptos app to <body>.
//!
//! Evidence-only spike for #506. The two evaluation screens (collection
//! browser + server status pane) live in `browser` / `status`; `api` is the
//! stubbed actions-over-HTTP edge (#505 builds the real one).

mod api;
mod app;
mod browser;
mod status;

use leptos::prelude::*;

fn main() {
    // The real client wires `console_error_panic_hook` for readable wasm
    // panics; the spike keeps the dep surface minimal.
    mount_to_body(app::App);
}
