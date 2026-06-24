"""The action registry: shape and contract pins.

The behavioural gate is the tools-layer unit files + the integration suite
passing unmodified through ``register_tools`` (they do); this pins the registry
itself — the 27-action surface, name stability, and the translation-ready
contract (async impls with response-model return annotations, docs present).
"""

from __future__ import annotations

import inspect

import pytest

from shrike.api.actions import ActionContext, build_actions

EXPECTED_ACTIONS = {
    "collection_info",
    "list_profiles",
    "export_package",
    "list_notes",
    "search_notes",
    "collection_query",
    "upsert_notes",
    "upsert_note_types",
    "update_note_type_fields",
    "update_note_type_templates",
    "find_replace_note_types",
    "update_note_type_field_metadata",
    "update_note_tags",
    "rename_tag",
    "find_replace_notes",
    "migrate_note_type",
    "upsert_decks",
    "delete_decks",
    "delete_notes",
    "delete_note_types",
    "collection_prune",
    "collection_check",
    "import_package",
    "store_media",
    "fetch_media",
    "list_media",
    "delete_media",
}


def test_registry_carries_the_full_tool_surface(kharness) -> None:
    actions = build_actions(ActionContext(wrapper=kharness.wrapper, kernel=kharness.kernel))
    assert {a.name for a in actions} == EXPECTED_ACTIONS
    assert len(actions) == 27


def test_actions_are_translation_ready(kharness) -> None:
    # Coarse async impls with documented contracts and model return annotations —
    # what lets another adapter bind the same registry without FastMCP.
    ctx = ActionContext(wrapper=kharness.wrapper, kernel=kharness.kernel)
    for action in build_actions(ctx):
        assert inspect.iscoroutinefunction(action.impl), action.name
        assert action.doc, f"{action.name} has no doc"
        signature = inspect.signature(action.impl)
        assert signature.return_annotation is not inspect.Signature.empty, action.name


def test_build_actions_requires_a_kernel(wrapper) -> None:
    # A kernel-less context is a configuration error, surfaced loudly at
    # registry build time.
    import pytest

    with pytest.raises(ValueError, match="kernel mode"):
        build_actions(ActionContext(wrapper=wrapper))


# The action-core registry seam: the kernel's list of re-homed actions and the
# Python binding's forwarding set must not drift silently.

shrike_native = pytest.importorskip("shrike_native")

# The actions whose bodies run in shrike_kernel::actions — actions.py forwards
# exactly these through the per-action bindings (the read surface).
# Growing this list is deliberate: add the binding, rewire the action, then
# extend this pin alongside the kernel's REHOMED_ACTIONS.
REHOMED = ["collection_info", "list_notes", "collection_query", "search_notes"]


def test_kernel_and_binding_agree_on_the_rehomed_set() -> None:
    assert shrike_native.rehomed_actions() == REHOMED


def test_rehomed_bindings_exist() -> None:
    for name in REHOMED:
        assert hasattr(shrike_native, f"action_{name}"), name
