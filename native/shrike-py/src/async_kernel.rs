//! Async kernel bindings (#332, S3a): the kernel's runtime-agnostic futures
//! awaited natively on the harness's asyncio loop via the runtime-less bridge.
//!
//! This first slice binds the kernel's `SerializedCollection` — open / a
//! collection op / close, each an `asyncio.Future` — proving the full chain:
//! kernel future → injected executor → loop-driven polls → Python `await`.
//! Later S3 slices widen this to the kernel's orchestration (index drift,
//! saver debounce, runtime lifecycle) and swap the inline `MutexExecutor`
//! for a harness-injected worker executor.

use std::sync::Arc;

use pyo3::prelude::*;

use shrike_kernel::{MutexExecutor, SerializedCollection};

use crate::asyncio_bridge::future_into_py;

/// An open collection whose every op is an awaitable serialized through the
/// kernel's injected executor.
#[pyclass]
pub(crate) struct AsyncCollection {
    inner: Arc<SerializedCollection>,
}

/// Open a collection asynchronously; resolves to an [`AsyncCollection`].
///
/// S3a uses the kernel's `MutexExecutor` (inline, conforming): collection
/// jobs run inside the poll callback on the loop thread. The harness-injected
/// worker executor (jobs off the loop) is the next S3 slice.
#[pyfunction]
pub(crate) fn async_collection_open<'py>(
    py: Python<'py>,
    collection_path: String,
) -> PyResult<Bound<'py, PyAny>> {
    future_into_py(py, async move {
        let collection =
            SerializedCollection::open(collection_path, Arc::new(MutexExecutor::default())).await?;
        Ok(AsyncCollection {
            inner: Arc::new(collection),
        })
    })
}

#[pymethods]
impl AsyncCollection {
    /// The collection's modification stamp (an awaitable).
    fn col_mod<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        future_into_py(py, async move { inner.run(|core| core.col_mod()).await? })
    }

    /// Note ids matching a raw Anki search (an awaitable).
    fn find_notes<'py>(&self, py: Python<'py>, query: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        future_into_py(py, async move {
            inner.run(move |core| core.find_notes(&query)).await?
        })
    }

    /// Close the collection (an awaitable).
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        future_into_py(py, async move { inner.close().await })
    }
}
