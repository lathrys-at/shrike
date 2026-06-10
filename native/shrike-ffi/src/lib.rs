//! FFI conventions shared by every Shrike native crate (#269, epic #265).
//!
//! This crate is **pure Rust** — no `pyo3` — and defines the two things every
//! native module shares: the error taxonomy and (in documentation) the
//! marshaling rules. The Python-facing half of these conventions (exception
//! classes, `allow_threads` wrapping) lives in `shrike-py`, the one crate
//! allowed to depend on `pyo3` (enforced by `//native:layering_check`).
//!
//! # Marshaling rules (epic #265 convention 6)
//!
//! Only these types cross the Python↔Rust boundary:
//!
//! - strings (`String`/`&str`) and byte buffers (`Vec<u8>`/`&[u8]`)
//! - f32 vectors / vector batches (zero-copy numpy interchange where arrays
//!   must cross, via the `numpy` crate in `shrike-py`)
//! - i64 key arrays
//! - small JSON-able maps (stats, health blocks)
//!
//! Never a live Python object, callback, or handle — calls are coarse and
//! batched so the boundary is crossed per *batch*, not per item.
//!
//! # Threading rules
//!
//! - All compute runs under `py.allow_threads` (GIL released) in `shrike-py`.
//! - No Python handle may cross into a worker thread; compute crates receive
//!   owned data only.
//!
//! # Error taxonomy
//!
//! Mirrors Shrike's Python-side expected-vs-bug split (`ToolInputError` vs a
//! genuine bug): [`ErrorKind::InvalidInput`] is expected bad input (surfaced to
//! Python without traceback noise), [`ErrorKind::Unavailable`] is a runtime
//! resource that isn't up (model not loaded, file missing), and
//! [`ErrorKind::Internal`] is a bug. `shrike-py` maps each kind to a distinct
//! Python exception class, and the Python facades translate those into the
//! existing error surface.

use std::error::Error;
use std::fmt;

/// The expected-vs-bug split every native error declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Expected bad input — the caller's request can't be honored as given.
    /// Python side: an input-error exception, logged without a traceback.
    InvalidInput,
    /// A runtime dependency isn't available (model not loaded, backend stopped,
    /// file missing). Python side: a runtime-error exception.
    Unavailable,
    /// A bug — anything that "can't happen". Python side: a runtime-error
    /// exception, logged with a traceback.
    Internal,
}

impl ErrorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorKind::InvalidInput => "invalid_input",
            ErrorKind::Unavailable => "unavailable",
            ErrorKind::Internal => "internal",
        }
    }
}

/// The error type every Shrike native crate returns across the FFI seam.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeError {
    pub kind: ErrorKind,
    pub message: String,
}

impl NativeError {
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidInput,
            message: message.into(),
        }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Unavailable,
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Internal,
            message: message.into(),
        }
    }
}

impl fmt::Display for NativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind.as_str(), self.message)
    }
}

impl Error for NativeError {}

/// The result type native compute functions return.
pub type NativeResult<T> = Result<T, NativeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_render_stably() {
        assert_eq!(ErrorKind::InvalidInput.as_str(), "invalid_input");
        assert_eq!(ErrorKind::Unavailable.as_str(), "unavailable");
        assert_eq!(ErrorKind::Internal.as_str(), "internal");
    }

    #[test]
    fn constructors_set_kind_and_message() {
        let e = NativeError::invalid_input("bad batch");
        assert_eq!(e.kind, ErrorKind::InvalidInput);
        assert_eq!(e.to_string(), "invalid_input: bad batch");
        assert_eq!(
            NativeError::unavailable("no model").kind,
            ErrorKind::Unavailable
        );
        assert_eq!(NativeError::internal("oops").kind, ErrorKind::Internal);
    }
}
