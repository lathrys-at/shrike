"""Smoke test for the shrike_native binding module (#269).

Pins the FFI conventions' executable exemplars: marshaling (a Vec<f64> batch
in, an f64 out), the GIL-released compute path, and the error-taxonomy →
exception-class mapping. Runs under Bazel (//shrike-core/shrike-py:smoke_test) and
plain pytest alike.
"""

from __future__ import annotations

import shrike_native


def test_version_and_build_info() -> None:
    assert shrike_native.version()
    assert "shrike-py" in shrike_native.build_info()


def test_parallel_sum_marshals_f64_batch() -> None:
    assert shrike_native.parallel_sum([1.5, 2.5, 4.0]) == 8.0
    assert shrike_native.parallel_sum([]) == 0.0


def test_checked_div_ok() -> None:
    assert shrike_native.checked_div(9.0, 3.0) == 3.0


def test_invalid_input_maps_to_input_error() -> None:
    try:
        shrike_native.checked_div(1.0, 0.0)
    except shrike_native.NativeInputError as e:
        assert "division by zero" in str(e)
    else:
        raise AssertionError("expected NativeInputError")


def test_exception_taxonomy_bases() -> None:
    # InvalidInput is expected-bad-input → ValueError family (facades translate
    # it to the ToolInputError surface); the rest are runtime errors.
    assert issubclass(shrike_native.NativeInputError, ValueError)
    assert issubclass(shrike_native.NativeUnavailableError, RuntimeError)
    assert issubclass(shrike_native.NativeInternalError, RuntimeError)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
