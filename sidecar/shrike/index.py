from __future__ import annotations

from typing import Any


class VectorIndex:
    """Stub vector index. Returns empty results until embedding service is available."""

    def __init__(self, path: str | None = None):
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def search(
        self,
        queries: list[str] | None = None,
        ids: list[int] | None = None,
        top_k: int = 10,
        **filters: Any,
    ) -> dict[str, Any]:
        return {"results": []}

    def on_notes_changed(self, note_ids: list[int]) -> None:
        pass

    def on_notes_deleted(self, note_ids: list[int]) -> None:
        pass
