from __future__ import annotations

import pytest

from shrike.collection import CollectionWrapper


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
    results = wrapper.run_sync(
        lambda _c: wrapper._upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "What is 2+2?", "Back": "4"},
                    "tags": ["math", "easy"],
                }
            ]
        )
    )
    return results[0]["id"]
