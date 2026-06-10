"""Type stubs for the compiled shrike_native._native module (#269).

Hand-written against native/shrike-py/src/lib.rs; mypy.stubtest in the native
CI lane fails if a Rust signature drifts from these.
"""

from typing import final

# pyo3's #[pymodule] auto-generates __all__ from the registered names.
__all__ = [
    "version",
    "build_info",
    "parallel_sum",
    "checked_div",
    "init_onnx_runtime",
    "OnnxTextEmbedder",
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
def init_onnx_runtime(dylib_path: str) -> None: ...

@final
class OnnxTextEmbedder:
    def __new__(
        cls,
        model_path: str,
        tokenizer_path: str,
        *,
        providers: list[str],
        pooling: str,
        normalize: bool,
        max_length: int,
    ) -> OnnxTextEmbedder: ...
    def embed_chunk(self, texts: list[str]) -> list[list[float]]: ...
    def dim(self) -> int | None: ...
    def active_providers(self) -> list[str]: ...
    def unsupported_inputs(self) -> list[str]: ...
