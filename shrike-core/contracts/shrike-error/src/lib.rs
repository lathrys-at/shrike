//! The shared error taxonomy every Shrike native crate returns across the FFI
//! seam.
//!
//! This crate is **pure Rust** — no `pyo3`. It defines one thing: [`NativeError`],
//! the error every native compute crate returns, and its [`ErrorKind`] projection.
//! The Python-facing half (exception classes, the `kind` → exception mapping) lives
//! in `shrike-pyo3`, the one crate allowed to depend on `pyo3` (enforced by
//! `//shrike-core:layering_check`); the marshaling/threading conventions are
//! documented on the binding crates that enforce them (`shrike-pyo3`,
//! `shrike-cabi`), not here.
//!
//! # Error taxonomy
//!
//! Mirrors Shrike's Python-side expected-vs-bug split (`ToolInputError` vs a
//! genuine bug): [`ErrorKind::InvalidInput`] is expected bad input (surfaced to
//! Python without traceback noise), [`ErrorKind::Unavailable`] is a runtime
//! resource that isn't up (model not loaded, file missing), [`ErrorKind::Busy`]
//! is collection-lock contention (retryable), and [`ErrorKind::Internal`] is a
//! bug. `shrike-pyo3` maps each kind to a distinct Python exception class, and the
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

#![deny(missing_docs)]
#![deny(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc
)]

use std::error::Error;

use tracing_error::SpanTrace;

/// The boxed source carrier for any leaf error not given a dedicated `From`.
pub type BoxError = Box<dyn Error + Send + Sync + 'static>;

/// The expected-vs-bug split every native error declares.
///
/// Deliberately a CLOSED set (not `#[non_exhaustive]`): the four kinds are an
/// INTERNAL taxonomy consumed only by in-workspace code, and they map 1:1 onto
/// the four Python exception classes in `shrike-pyo3`. The exhaustive match in
/// `to_py_err` is the gate — adding a kind here is a deliberate COMPILE ERROR
/// there, forcing a new exception mapping rather than letting a wildcard
/// fail-soft mis-map it. (`#[non_exhaustive]` is for a published crate's public
/// error enum; this is neither.) `NativeError` itself stays extensible the
/// struct way: construct it only via the constructors / `.context()`, never a
/// struct literal, so adding a field later is non-breaking.
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
    /// The stable wire string for this kind (matched in `shrike-pyo3`'s mapping).
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
/// Constructors capture the current `tracing` span trace: with the
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
    ///
    /// # Errors
    ///
    /// Returns the mapped `NativeError` when `self` is `Err`; an `Ok` value
    /// passes through unchanged.
    fn context(self, kind: ErrorKind, message: impl Into<String>) -> NativeResult<T>;

    /// Map any error into a `NativeError` of `kind`, building the label lazily
    /// (only on the error path).
    ///
    /// # Errors
    ///
    /// Returns the mapped `NativeError` when `self` is `Err`, calling `message`
    /// to build the label only on that path; an `Ok` value passes through
    /// unchanged.
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

    // --- Adversarial additions ---------------------------------------------
    //
    // Inline SplitMix64 (copied from shrike-store) for generative cases. No deps.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// Every variant; the count is pinned so adding/removing a kind forces this
    /// test (and the FFI `to_py_err` mapping it stands in for) to be revisited.
    const ALL_KINDS: [ErrorKind; 4] = [
        ErrorKind::InvalidInput,
        ErrorKind::Unavailable,
        ErrorKind::Busy,
        ErrorKind::Internal,
    ];

    /// The wire strings are a CONTRACT the Python side (`shrike-pyo3`'s kind →
    /// exception map) string-matches on. Pin each exact byte: a rename here that
    /// isn't mirrored there silently mis-maps every error of that kind to the
    /// fail-soft branch. Order-independent set check so a future reorder of the
    /// enum can't mask a changed string.
    #[test]
    fn as_str_pins_exact_wire_strings() {
        for kind in ALL_KINDS {
            let expected = match kind {
                ErrorKind::InvalidInput => "invalid_input",
                ErrorKind::Unavailable => "unavailable",
                ErrorKind::Busy => "busy",
                ErrorKind::Internal => "internal",
            };
            assert_eq!(kind.as_str(), expected);
            // Display must agree with as_str (the binding layer may use either).
            assert_eq!(kind.to_string(), expected);
            // as_str is 'static and stable across calls (same pointer-backed str).
            assert_eq!(kind.as_str(), kind.as_str());
        }
    }

    /// as_str must be INJECTIVE — two kinds sharing a wire string would collapse
    /// two distinct Python exception classes into one on the FFI seam.
    #[test]
    fn as_str_is_unique_per_variant() {
        let strs: Vec<&'static str> = ALL_KINDS.iter().map(|k| k.as_str()).collect();
        let mut deduped = strs.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(
            strs.len(),
            deduped.len(),
            "as_str collision across variants"
        );
        // Pin the variant count: a 5th kind needs a 5th wire string + Py mapping.
        assert_eq!(ALL_KINDS.len(), 4);
    }

    /// Each constructor must land on its matching kind — the kind drives the
    /// Python exception class, so a swapped constructor mis-classes the failure.
    #[test]
    fn every_constructor_maps_to_its_kind() {
        type Ctor = fn(&str) -> NativeError;
        let cases: [(Ctor, ErrorKind); 4] = [
            (
                |m| NativeError::invalid_input(m.to_owned()),
                ErrorKind::InvalidInput,
            ),
            (
                |m| NativeError::unavailable(m.to_owned()),
                ErrorKind::Unavailable,
            ),
            (|m| NativeError::busy(m.to_owned()), ErrorKind::Busy),
            (|m| NativeError::internal(m.to_owned()), ErrorKind::Internal),
        ];
        for (ctor, expected) in cases {
            let e = ctor("x");
            assert_eq!(e.kind(), expected);
            assert!(
                e.source().is_none(),
                "plain constructor must have no source"
            );
        }
    }

    /// The message rides the FFI seam verbatim — no trimming, escaping, or
    /// truncation. Hostile payloads (empty, unicode, embedded NUL/newline, very
    /// long) must round-trip byte-for-byte into `message`, and `Display` must be
    /// exactly `"{kind}: {message}"` with that same payload.
    #[test]
    fn message_is_preserved_verbatim_for_hostile_payloads() {
        let payloads = [
            "",
            "ascii simple",
            "unicode: café 日本語 🐦‍⬛ Ωμέγα",
            "embedded\nnewline\tand\ttabs",
            "embedded\0null\0bytes",
            "right-to-left \u{202E}reversed",
            &"x".repeat(100_000),
        ];
        for p in payloads {
            let e = NativeError::invalid_input(p);
            assert_eq!(e.message, p, "message mutated for payload {p:?}");
            assert_eq!(e.to_string(), format!("invalid_input: {p}"));
        }
    }

    /// Generative: a constructed error's `message` equals the input and `kind`
    /// is whatever was requested, across random kind/message combinations. Pins
    /// "constructor never reinterprets its inputs" beyond the curated cases.
    #[test]
    fn constructor_inputs_round_trip_generative() {
        let mut rng = Rng::new(0xDEAD_BEEF);
        for _ in 0..256 {
            let kind = ALL_KINDS[(rng.next_u64() % 4) as usize];
            // Build a pseudo-random message including some control/unicode bytes.
            let len = (rng.next_u64() % 40) as usize;
            // Includes control chars, NULs, and high-plane unicode — message must
            // survive all of it unchanged.
            let msg: String = (0..len)
                .map(|_| char::from_u32((rng.next_u64() % 0x110000) as u32).unwrap_or('\u{fffd}'))
                .collect();
            let e = match kind {
                ErrorKind::InvalidInput => NativeError::invalid_input(msg.clone()),
                ErrorKind::Unavailable => NativeError::unavailable(msg.clone()),
                ErrorKind::Busy => NativeError::busy(msg.clone()),
                ErrorKind::Internal => NativeError::internal(msg.clone()),
            };
            assert_eq!(e.kind(), kind);
            assert_eq!(e.message, msg);
            assert_eq!(e.to_string(), format!("{}: {}", kind.as_str(), msg));
        }
    }

    /// A multi-level chain (NativeError → mid leaf → root leaf) must walk fully
    /// via `Error::source()` iteration, and `.source()` on each level must yield
    /// the next — the Rust-side diagnostic chain the docs promise.
    #[test]
    fn source_chain_walks_multiple_levels() {
        // Build a 3-deep leaf chain using a custom error carrying a source.
        #[derive(Debug)]
        struct Layer {
            msg: &'static str,
            cause: Option<BoxError>,
        }
        impl std::fmt::Display for Layer {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.msg)
            }
        }
        impl Error for Layer {
            fn source(&self) -> Option<&(dyn Error + 'static)> {
                self.cause.as_deref().map(|c| c as &(dyn Error + 'static))
            }
        }
        let root = Layer {
            msg: "root cause",
            cause: None,
        };
        let mid = Layer {
            msg: "mid layer",
            cause: Some(Box::new(root)),
        };
        let top = NativeError::with_source(ErrorKind::Internal, "top context", mid);

        // Walk: top.source() == mid, mid.source() == root, root.source() == None.
        let mut chain: Vec<String> = Vec::new();
        let mut cur: Option<&(dyn Error + 'static)> = top.source();
        while let Some(e) = cur {
            chain.push(e.to_string());
            cur = e.source();
        }
        assert_eq!(
            chain,
            vec!["mid layer".to_string(), "root cause".to_string()]
        );
    }

    /// `Display` must NOT splice the source in — the top message is the context
    /// label only, so the boundary string isn't duplicated/leaked with the leaf.
    /// (The leaf is recoverable via `.source()`, never via the message string.)
    #[test]
    fn display_does_not_leak_or_duplicate_the_source() {
        let leaf: BoxError = "SECRET-LEAF-TEXT".into();
        let e = NativeError::with_source(ErrorKind::Unavailable, "public label", leaf);
        let shown = e.to_string();
        assert_eq!(shown, "unavailable: public label");
        assert!(
            !shown.contains("SECRET-LEAF-TEXT"),
            "source text leaked into Display"
        );
        // Debug, by contrast, may show internals, but must still carry the label
        // and must not panic.
        let dbg = format!("{e:?}");
        assert!(dbg.contains("public label"));
    }

    /// `Display` must be human-readable message text, not the derived Debug form
    /// (no struct-name / field noise).
    #[test]
    fn display_is_not_the_debug_form() {
        let e = NativeError::internal("boom");
        assert_eq!(e.to_string(), "internal: boom");
        assert!(!e.to_string().contains("NativeError"));
        assert!(!e.to_string().contains("span_trace"));
    }

    /// The serde_json `From` must attach the ORIGINAL error as the source and
    /// downcast back to `serde_json::Error` (not a re-stringified copy) — the
    /// kind must be InvalidInput as documented.
    #[test]
    fn from_serde_json_attaches_downcastable_source_with_kind() {
        let serde_err = serde_json::from_str::<serde_json::Value>("{bad").unwrap_err();
        let native: NativeError = serde_err.into();
        assert_eq!(native.kind(), ErrorKind::InvalidInput);
        let src = native.source().expect("source must be attached");
        assert!(
            src.downcast_ref::<serde_json::Error>().is_some(),
            "source did not downcast to serde_json::Error"
        );
    }

    /// The io `From` must attach the ORIGINAL io error as the source, downcast
    /// back (preserving `io::ErrorKind`), and map to Unavailable as documented.
    #[test]
    fn from_io_attaches_downcastable_source_with_kind() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let native: NativeError = io_err.into();
        assert_eq!(native.kind(), ErrorKind::Unavailable);
        let src = native.source().expect("source must be attached");
        let downcast = src
            .downcast_ref::<std::io::Error>()
            .expect("source did not downcast to io::Error");
        assert_eq!(downcast.kind(), std::io::ErrorKind::PermissionDenied);
    }

    /// A custom (non-leaf) error must box through `Into<BoxError>` via
    /// `with_source` and downcast back — the path the docs require for heavy
    /// leaves (ort/rusqlite/ureq) that get NO dedicated `From` in this crate.
    #[test]
    fn custom_error_boxes_and_downcasts_through_with_source() {
        #[derive(Debug)]
        struct EngineErr {
            code: u32,
        }
        impl std::fmt::Display for EngineErr {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "engine failed: {}", self.code)
            }
        }
        impl Error for EngineErr {}

        let e = NativeError::with_source(ErrorKind::Unavailable, "embed", EngineErr { code: 7 });
        let src = e.source().expect("source attached");
        let downcast = src
            .downcast_ref::<EngineErr>()
            .expect("custom error did not downcast back");
        assert_eq!(downcast.code, 7);
        // And via ResultExt::context the same custom error round-trips.
        let r: Result<(), EngineErr> = Err(EngineErr { code: 42 });
        let e2 = r.context(ErrorKind::Internal, "via context").unwrap_err();
        assert_eq!(
            e2.source()
                .unwrap()
                .downcast_ref::<EngineErr>()
                .unwrap()
                .code,
            42
        );
    }

    /// With NO tracing subscriber installed (the production default for these
    /// pure-Rust tests), `trace()` must be predictable and never panic. Pin the
    /// observed behavior so a regression in capture is caught.
    #[test]
    fn trace_is_none_without_a_subscriber_and_never_panics() {
        let e = NativeError::internal("no span here");
        // Must not panic when rendering an empty/absent trace.
        let t = e.trace();
        assert_eq!(t, None, "no subscriber installed should yield no trace");
        // Idempotent: calling twice yields the same result, no interior mutation.
        assert_eq!(e.trace(), e.trace());
    }

    /// `with_context` builds its label lazily AND, on the error path, the lazily
    /// built label becomes the message with the source still attached — the Ok
    /// path skips the closure (covered elsewhere); here we pin the error path.
    #[test]
    fn with_context_error_path_uses_built_label_and_keeps_source() {
        let mut called = false;
        let r: Result<(), std::io::Error> = Err(std::io::Error::other("raw"));
        let e = r
            .with_context(ErrorKind::Busy, || {
                called = true;
                format!("dynamic label {}", 1 + 1)
            })
            .unwrap_err();
        assert!(called, "closure must run on the error path");
        assert_eq!(e.kind(), ErrorKind::Busy);
        assert_eq!(e.message, "dynamic label 2");
        assert_eq!(e.source().unwrap().to_string(), "raw");
    }

    /// `ResultExt::context` on an `Ok` must pass the value through untouched and
    /// must NOT fabricate an error — the trait wraps only the Err arm.
    #[test]
    fn context_passes_ok_through_unchanged() {
        let ok: Result<u32, std::io::Error> = Ok(99);
        let v = ok.context(ErrorKind::Internal, "unused label").unwrap();
        assert_eq!(v, 99);
    }
}
