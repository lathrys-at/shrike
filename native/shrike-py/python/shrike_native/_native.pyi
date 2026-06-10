"""Type stubs for the compiled shrike_native._native module (#269).

Hand-written against native/shrike-py/src/lib.rs; mypy.stubtest in the native
CI lane fails if a Rust signature drifts from these.
"""

# pyo3's #[pymodule] auto-generates __all__ from the registered names.
__all__ = [
    "version",
    "build_info",
    "parallel_sum",
    "checked_div",
    "NativeInputError",
    "NativeUnavailableError",
    "NativeInternalError",
]

class NativeInputError(ValueError): ...
class NativeUnavailableError(RuntimeError): ...
class NativeInternalError(RuntimeError): ...

def version() -> str: ...
def build_info() -> str: ...
def parallel_sum(values: list[float]) -> float: ...
def checked_div(a: float, b: float) -> float: ...
