"""shrike-pyo3 exception taxonomy: the ErrorKind -> PyException base classes.

The `ErrorKind -> Native*Error` mapping (the `to_py_err` match in
`shrike-pyo3/src/lib.rs`) is exhaustive over a closed 4-variant enum and is
covered *behaviourally* by the native suite (each variant is provoked and the
concrete class asserted). What that suite never pins is the load-bearing BASE
class each maps onto — the contract the harness facades catch on:

- `derived.py` translates `Native*` errors into `sqlite3.Error`,
- the actions edge keys `CollectionBusyError` off `NativeBusyError`,
- the CLI's input-error surface keys off the `ValueError` base.

So `NativeInputError` MUST stay a `ValueError` and the runtime trio MUST stay
`RuntimeError`s. A refactor that re-parents one would compile and pass the
behavioural tests (the concrete class is unchanged) while silently breaking the
`except ValueError`/`except RuntimeError` catches downstream. These assertions
fail that loudly. No kernel needed — the exception classes exist in every
`shrike_native` build, so this also guards the minimal-core profiles.
"""

from __future__ import annotations

import pytest

shrike_native = pytest.importorskip("shrike_native")


def test_input_error_is_a_value_error() -> None:
    assert issubclass(shrike_native.NativeInputError, ValueError)
    assert not issubclass(shrike_native.NativeInputError, RuntimeError)


def test_runtime_trio_are_runtime_errors_not_value_errors() -> None:
    for exc in (
        shrike_native.NativeUnavailableError,
        shrike_native.NativeInternalError,
        shrike_native.NativeBusyError,
    ):
        assert issubclass(exc, RuntimeError)
        assert not issubclass(exc, ValueError)


def test_the_four_native_kinds_are_distinct_types() -> None:
    # The four ErrorKind arms map to four distinct exception classes (no two
    # variants collapse onto one Python type, so a catch can discriminate them).
    kinds = {
        shrike_native.NativeInputError,
        shrike_native.NativeUnavailableError,
        shrike_native.NativeInternalError,
        shrike_native.NativeBusyError,
    }
    assert len(kinds) == 4
