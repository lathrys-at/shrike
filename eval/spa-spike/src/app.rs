//! App shell: nav + client-side router over the two evaluation screens.

use leptos::prelude::*;
use leptos_router::components::{Route, Router, Routes, A};
use leptos_router::path;

use crate::browser::CollectionBrowser;
use crate::status::StatusPane;

#[component]
pub fn App() -> impl IntoView {
    view! {
        <Router>
            <nav>
                <A href="/">"Collection browser"</A>
                <A href="/status">"Server status"</A>
            </nav>
            <main>
                <Routes fallback=|| "Not found.".into_view()>
                    <Route path=path!("/") view=CollectionBrowser />
                    <Route path=path!("/status") view=StatusPane />
                </Routes>
            </main>
        </Router>
    }
}
