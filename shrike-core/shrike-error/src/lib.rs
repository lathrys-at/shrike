//! The shared error taxonomy every Shrike native crate returns across the FFI
//! seam (#269, epic #265; rewritten to thiserror in #705).
//!
//! This crate is **pure Rust** — no `pyo3`. It defines one thing: [`NativeError`],
//! the error every native compute crate returns, and its [`ErrorKind`] projection.
//! The Python-facing half (exception classes, the `kind` → exception mapping) lives
//! in `shrike-py`, the one crate allowed to depend on `pyo3` (enforced by
//! `//shrike-core:layering_check`); the marshaling/threading conventions are
//! documented on the binding crates that enforce them (`shrike-py`,
//! `shrike-mobile`), not here.
//!
//! # Error taxonomy
//!
//! Mirrors Shrike's Python-side expected-vs-bug split (`ToolInputError` vs a
//! genuine bug): [`ErrorKind::InvalidInput`] is expected bad input (surfaced to
//! Python without traceback noise), [`ErrorKind::Unavailable`] is a runtime
//! resource that isn't up (model not loaded, file missing), [`ErrorKind::Busy`]
//! is collection-lock contention (retryable), and [`ErrorKind::Internal`] is a
//! bug. `shrike-py` maps each kind to a distinct Python exception class, and the
//! Python facades translate those into the existing error surface.
//!
//! # Source chain
//!
//! `NativeError` carries an optional `#[source]` cause, so a failure that wraps a
//! leaf error (`serde_json`, `io`, and — feature-gated — `ort`/`rusqlite`/`ureq`)
//! keeps that cause recoverable Rust-side via [`std::error::Error::source`]: the
//! `?` operator converts the leaf in (the `From` impls below), and the context
//! label rides in [`NativeError::message`] while the leaf rides as the source.
//! The chain is for Rust-side diagnostics (logging, `{:?}`, `Display`); it does
//! **not** cross the FFI boundary — only `(kind, message, trace)` does.
//!
//! # Only lightweight leaves get a `From`
//!
//! This is a layer-floor crate every native crate depends on, including the
//! kernel. Only the lightweight, ubiquitous leaves (`serde_json`, `io`) get an
//! unconditional `From` here. The HEAVY/ENGINE leaves (`ort`, `rusqlite`,
//! `ureq`) deliberately get NONE — not even feature-gated: under Cargo workspace
//! feature unification a single `shrike-error` is built for the whole graph, so
//! a `from-ort` feature turned on by any crate would activate this crate's
//! optional `ort` dep for everyone and pull `ort` into the kernel
//! (`shrike-error -> shrike-collection -> shrike-kernel`), the exact bloat the
//! "kernel embeds never enter ort" firewall must prevent. A using crate instead
//! attaches its heavy leaf as the recoverable `#[source]` via
//! [`ResultExt::context`] / [`NativeError::with_source`], which boxes through
//! `Into<BoxError>` and needs no leaf dependency in this crate.

use std::error::Error;

use tracing_error::SpanTrace;

/// The boxed source carrier for any leaf error not given a dedicated `From`.
pub type BoxError = Box<dyn Error + Send + Sync + 'static>;

/// The expected-vs-bug split every native error declares.
///
/// Deliberately a CLOSED set: the four kinds map 1:1 onto the four Python
/// exception classes in `shrike-py`, so exhaustive matching there is the gate
/// that a new kind forces a new exception mapping. (`NativeError` itself is
/// closed by its private fields — the analog of `#[non_exhaustive]` for a
/// struct — so it stays extensible without breaking callers.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Expected bad input — the caller's request can't be honored as given.
    /// Python side: an input-error exception, logged without a traceback.
    InvalidInput,
    /// A runtime dependency isn't available (model not loaded, backend stopped,
    /// file missing). Python side: a runtime-error exception.
    Unavailable,
    /// The collection is held by another process (lock contention — usually
    /// Anki desktop). Expected and retryable, never a bug: the caller decides
    /// whether to wait. Python side: the busy exception the facades map onto
    /// the existing CollectionBusyError surface.
    Busy,
    /// A bug — anything that "can't happen". Python side: a runtime-error
    /// exception, logged with a traceback.
    Internal,
}

impl ErrorKind {
    /// The stable wire string for this kind (matched in `shrike-py`'s mapping).
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorKind::InvalidInput => "invalid_input",
            ErrorKind::Unavailable => "unavailable",
            ErrorKind::Busy => "busy",
            ErrorKind::Internal => "internal",
        }
    }
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The error type every Shrike native crate returns across the FFI seam.
///
/// Constructors capture the current `tracing` span trace (#308): with the
/// harness-installed subscriber active, [`NativeError::trace`] renders the span
/// context the error crossed (which op, which batch, which engine), and the
/// binding layer attaches it to the Python exception (PEP 678 notes) — so a
/// native failure is debuggable from the harness without native logging.
///
/// The `#[source]` cause keeps a wrapped leaf error recoverable Rust-side; it
/// does not cross the FFI boundary. `message` is the public, human-readable
/// context (the only string the boundary carries besides `kind`/`trace`).
#[derive(Debug, thiserror::Error)]
#[error("{kind}: {message}")]
pub struct NativeError {
    kind: ErrorKind,
    /// The human-readable context. Public: the binding layers move it onto the
    /// Python/JSON wire shape, and it's the field tests assert on.
    pub message: String,
    #[source]
    source: Option<BoxError>,
    span_trace: SpanTrace,
}

impl NativeError {
    /// The shared private constructor: every public path captures the span trace
    /// here, so a `NativeError` always carries one.
    fn build(kind: ErrorKind, message: String, source: Option<BoxError>) -> Self {
        Self {
            kind,
            message,
            source,
            span_trace: SpanTrace::capture(),
        }
    }

    /// Expected bad input ([`ErrorKind::InvalidInput`]), no wrapped cause.
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::build(ErrorKind::InvalidInput, message.into(), None)
    }

    /// A runtime dependency isn't up ([`ErrorKind::Unavailable`]), no cause.
    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::build(ErrorKind::Unavailable, message.into(), None)
    }

    /// A native-side bug ([`ErrorKind::Internal`]), no wrapped cause.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::build(ErrorKind::Internal, message.into(), None)
    }

    /// Collection-lock contention ([`ErrorKind::Busy`]), no wrapped cause.
    pub fn busy(message: impl Into<String>) -> Self {
        Self::build(ErrorKind::Busy, message.into(), None)
    }

    /// Build a `NativeError` of `kind` whose `message` is the context label and
    /// whose `#[source]` is `cause` — the explicit form behind [`ResultExt`] and
    /// the dedicated `From` impls. Use this (or `.context`) instead of
    /// `format!("label: {e}")` so the leaf cause stays recoverable.
    pub fn with_source(
        kind: ErrorKind,
        message: impl Into<String>,
        cause: impl Into<BoxError>,
    ) -> Self {
        Self::build(kind, message.into(), Some(cause.into()))
    }

    /// This error's taxonomy kind — the cheap projection the binding layer maps
    /// to a Python exception class.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The rendered span trace, or None when no spans were active (no subscriber
    /// installed, or the error arose outside any span).
    pub fn trace(&self) -> Option<String> {
        let rendered = self.span_trace.to_string();
        if rendered.is_empty() {
            None
        } else {
            Some(rendered)
        }
    }
}

/// The result type native compute functions return.
pub type NativeResult<T> = Result<T, NativeError>;

/// Add a `NativeError` context to a `Result`, keeping the original error as the
/// recoverable `#[source]` cause.
///
/// This is the idiomatic replacement for the old `.map_err(|e| NativeError::kind(
/// format!("label: {e}")))` closures: the context label becomes
/// [`NativeError::message`] and the leaf error becomes the source, so the cause
/// survives instead of being flattened into the string.
///
/// ```
/// use shrike_error::{ErrorKind, ResultExt};
///
/// fn parse(s: &str) -> shrike_error::NativeResult<u32> {
///     s.parse::<u32>()
///         .context(ErrorKind::InvalidInput, "not a count")
/// }
/// assert!(parse("x").is_err());
/// ```
pub trait ResultExt<T> {
    /// Map any error into a `NativeError` of `kind` with `message` as the label
    /// and the original error as the source.
    fn context(self, kind: ErrorKind, message: impl Into<String>) -> NativeResult<T>;

    /// Map any error into a `NativeError` of `kind`, building the label lazily
    /// (only on the error path).
    fn with_context<F, S>(self, kind: ErrorKind, message: F) -> NativeResult<T>
    where
        F: FnOnce() -> S,
        S: Into<String>;
}

impl<T, E> ResultExt<T> for Result<T, E>
where
    E: Into<BoxError>,
{
    fn context(self, kind: ErrorKind, message: impl Into<String>) -> NativeResult<T> {
        self.map_err(|e| NativeError::with_source(kind, message, e))
    }

    fn with_context<F, S>(self, kind: ErrorKind, message: F) -> NativeResult<T>
    where
        F: FnOnce() -> S,
        S: Into<String>,
    {
        self.map_err(|e| NativeError::with_source(kind, message().into(), e))
    }
}

// --- Leaf-error `From` impls -------------------------------------------------
//
// Unconditional: lightweight, ubiquitous leaves. A bare `?` lands here, taking
// the leaf's own message as the context (use `.context(..)` where a richer label
// is wanted). serde_json defaults to InvalidInput (malformed input is the common
// case; a serialize-side failure that is really a bug should use
// `NativeError::internal`/`.context(ErrorKind::Internal, ..)` explicitly).
// io defaults to Unavailable (the common case is a missing/unreadable file).

impl From<serde_json::Error> for NativeError {
    fn from(e: serde_json::Error) -> Self {
        Self::with_source(ErrorKind::InvalidInput, e.to_string(), e)
    }
}

impl From<std::io::Error> for NativeError {
    fn from(e: std::io::Error) -> Self {
        Self::with_source(ErrorKind::Unavailable, e.to_string(), e)
    }
}

// The heavy/engine leaves (ort, rusqlite, ureq) get NO `From` — see the
// crate-level docs: a using crate attaches them as `#[source]` via
// `ResultExt::context` / `with_source`, keeping them out of this floor crate's
// dependency closure (and thus out of the kernel).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_render_stably() {
        assert_eq!(ErrorKind::InvalidInput.as_str(), "invalid_input");
        assert_eq!(ErrorKind::Unavailable.as_str(), "unavailable");
        assert_eq!(ErrorKind::Internal.as_str(), "internal");
        assert_eq!(ErrorKind::Busy.as_str(), "busy");
        // Display agrees with as_str (the format string interpolates it).
        assert_eq!(ErrorKind::Busy.to_string(), "busy");
    }

    #[test]
    fn constructors_set_kind_and_message() {
        let e = NativeError::invalid_input("bad batch");
        assert_eq!(e.kind(), ErrorKind::InvalidInput);
        assert_eq!(e.to_string(), "invalid_input: bad batch");
        assert_eq!(
            NativeError::unavailable("no model").kind(),
            ErrorKind::Unavailable
        );
        assert_eq!(NativeError::internal("oops").kind(), ErrorKind::Internal);
        assert_eq!(NativeError::busy("locked").kind(), ErrorKind::Busy);
    }

    #[test]
    fn plain_constructors_have_no_source() {
        let e = NativeError::internal("oops");
        assert!(e.source().is_none());
    }

    #[test]
    fn with_source_keeps_the_cause_recoverable() {
        let leaf: BoxError = "leaf boom".into();
        let e = NativeError::with_source(ErrorKind::Internal, "wrapping context", leaf);
        assert_eq!(e.kind(), ErrorKind::Internal);
        // The context is the message; the leaf is the recoverable source.
        assert_eq!(e.to_string(), "internal: wrapping context");
        assert_eq!(e.source().unwrap().to_string(), "leaf boom");
    }

    #[test]
    fn context_extension_wraps_label_and_source() {
        let r: Result<(), std::io::Error> =
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "nope"));
        let e = r.context(ErrorKind::Unavailable, "open index").unwrap_err();
        assert_eq!(e.kind(), ErrorKind::Unavailable);
        assert_eq!(e.message, "open index");
        assert_eq!(e.source().unwrap().to_string(), "nope");
    }

    #[test]
    fn with_context_builds_label_lazily() {
        let ok: Result<u32, std::io::Error> = Ok(3);
        // The closure must not run on the Ok path.
        let v = ok
            .with_context(ErrorKind::Internal, || -> String {
                unreachable!("only on error")
            })
            .unwrap();
        assert_eq!(v, 3);
    }

    #[test]
    fn from_serde_json_flows_with_questionmark() {
        fn parse(s: &str) -> NativeResult<serde_json::Value> {
            Ok(serde_json::from_str(s)?)
        }
        let e = parse("{not json").unwrap_err();
        // serde_json defaults to InvalidInput; the leaf rides as the source.
        assert_eq!(e.kind(), ErrorKind::InvalidInput);
        assert!(e.source().is_some());
    }

    #[test]
    fn from_io_flows_with_questionmark() {
        fn read() -> NativeResult<String> {
            Ok(std::fs::read_to_string("/no/such/path/at/all")?)
        }
        let e = read().unwrap_err();
        assert_eq!(e.kind(), ErrorKind::Unavailable);
        assert!(e.source().is_some());
    }
}
