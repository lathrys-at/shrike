"""The action-core registry seam (#331): the kernel's list of re-homed actions
and the Python binding's forwarding set must not drift silently."""

from __future__ import annotations

import pytest

shrike_native = pytest.importorskip("shrike_native")

# The actions whose bodies run in shrike_kernel::actions — actions.py forwards
# exactly these through the per-action bindings (slice 1: the read surface).
# Growing this list is deliberate: add the binding, rewire the action, then
# extend this pin alongside the kernel's REHOMED_ACTIONS.
REHOMED = ["collection_info", "list_notes", "collection_query"]


def test_kernel_and_binding_agree_on_the_rehomed_set() -> None:
    assert shrike_native.rehomed_actions() == REHOMED


def test_rehomed_bindings_exist() -> None:
    for name in REHOMED:
        assert hasattr(shrike_native, f"action_{name}"), name
