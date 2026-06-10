from __future__ import annotations

import json

import pytest

from shrike.collection import CollectionWrapper


def make_notes(wrapper: CollectionWrapper, notes: list[dict]) -> list[dict]:
    """Create notes synchronously through the wrapper's worker thread.

    The shared setup helper: routes through the SAME native upsert the async
    API uses (so fixtures and tests exercise one code path), usable from sync
    and async tests alike.
    """
    return wrapper.run_sync(  # type: ignore[no-any-return]
        lambda c: json.loads(c.upsert_notes(json.dumps(notes), "error", False))
    )


@pytest.fixture()
def wrapper(tmp_path):
    """Create a CollectionWrapper backed by a fresh empty Anki collection."""
    path = str(tmp_path / "collection.anki2")
    w = CollectionWrapper(path)
    yield w
    w.close()


@pytest.fixture()
def basic_note(wrapper):
    """Create a single Basic note in the Test deck and return its ID.

    Synchronous so it can serve both sync and async tests. Routes through the
    wrapper's worker thread (the same serialized path the async API uses).
    """
    results = make_notes(
        wrapper,
        [
            {
                "deck": "Test",
                "note_type": "Basic",
                "fields": {"Front": "What is 2+2?", "Back": "4"},
                "tags": ["math", "easy"],
            }
        ],
    )
    return results[0]["id"]
