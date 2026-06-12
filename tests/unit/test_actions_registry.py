"""The action registry (#276): shape and contract pins.

The behavioural gate is the tools-layer unit files + the integration suite
passing unmodified through ``register_tools`` (they do); this pins the registry
itself — the 24-action surface, name stability, and the translation-ready
contract (async impls with response-model return annotations, docs present).
"""

from __future__ import annotations

import inspect

from shrike.actions import ActionContext, build_actions

EXPECTED_ACTIONS = {
    "collection_info",
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
    "store_media",
    "fetch_media",
    "list_media",
    "delete_media",
}


def test_registry_carries_the_full_tool_surface(kharness) -> None:
    actions = build_actions(ActionContext(wrapper=kharness.wrapper, kernel=kharness.kernel))
    assert {a.name for a in actions} == EXPECTED_ACTIONS
    assert len(actions) == 24


def test_actions_are_translation_ready(kharness) -> None:
    # Coarse async impls with documented contracts and model return annotations —
    # what lets another adapter (or the Rust registry at stretch slice 2) bind
    # the same registry without FastMCP.
    ctx = ActionContext(wrapper=kharness.wrapper, kernel=kharness.kernel)
    for action in build_actions(ctx):
        assert inspect.iscoroutinefunction(action.impl), action.name
        assert action.doc, f"{action.name} has no doc"
        signature = inspect.signature(action.impl)
        assert signature.return_annotation is not inspect.Signature.empty, action.name


def test_build_actions_requires_a_kernel(wrapper) -> None:
    # The standalone (facade) mode retired with #355: a kernel-less context is
    # a configuration error, surfaced loudly at registry build time.
    import pytest

    with pytest.raises(ValueError, match="kernel mode"):
        build_actions(ActionContext(wrapper=wrapper))
