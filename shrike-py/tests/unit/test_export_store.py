"""The ExportStore: token lifecycle (claim/reap one-shot, close, TTL sweep)
and token unguessability.

The kernel-level export round-trip + symlink safety are pinned in
tests/native/test_export_package.py and the shrike-collection Rust tests.
"""

from __future__ import annotations

import os

from shrike.server.export_store import ExportStore


class TestExportStore:
    def test_claim_then_reap_is_one_shot(self, tmp_path) -> None:
        store = ExportStore(str(tmp_path / "cache"))
        token, path = store.new_temp_path(suffix=".apkg")
        with open(path, "wb") as f:
            f.write(b"pkg")
        store.register(token, path, "apkg")
        assert store.claim(token) == path
        store.reap(token)
        assert store.claim(token) is None
        assert not os.path.exists(path)

    def test_claim_is_one_shot(self, tmp_path) -> None:
        # The one-shot reap runs as a post-response background task, *after* the
        # first GET's body is streamed — so it can land after the client has
        # already issued a second GET. claim() consumes the token synchronously,
        # so the second GET misses here (a clean 404) instead of serving a file
        # the reap is concurrently removing → 500. No reap runs below — the
        # second claim must already miss on the strength of the mark alone.
        store = ExportStore(str(tmp_path / "cache"))
        token, path = store.new_temp_path(suffix=".apkg")
        with open(path, "wb") as f:
            f.write(b"pkg")
        store.register(token, path, "apkg")
        assert store.claim(token) == path  # first GET claims and serves
        assert store.claim(token) is None  # second GET misses (one-shot)
        # The claimed-but-unreaped file is not orphaned: close() still reaps it.
        store.close()
        assert not os.path.exists(path)

    def test_close_reaps_pending(self, tmp_path) -> None:
        store = ExportStore(str(tmp_path / "cache"))
        token, path = store.new_temp_path(suffix=".colpkg")
        with open(path, "wb") as f:
            f.write(b"pkg")
        store.register(token, path, "colpkg")
        store.close()
        assert not os.path.exists(path)
        assert store.claim(token) is None

    def test_ttl_sweep_reaps_expired(self, tmp_path) -> None:
        store = ExportStore(str(tmp_path / "cache"), ttl_seconds=0.0)
        token, path = store.new_temp_path(suffix=".apkg")
        with open(path, "wb") as f:
            f.write(b"pkg")
        store.register(token, path, "apkg")
        # ttl=0 → the next claim sweeps it as expired.
        assert store.claim(token) is None
        assert not os.path.exists(path)

    def test_token_is_unguessable(self, tmp_path) -> None:
        store = ExportStore(str(tmp_path / "cache"))
        t1, _ = store.new_temp_path(suffix=".apkg")
        t2, _ = store.new_temp_path(suffix=".apkg")
        assert t1 != t2
        assert len(t1) >= 24  # secrets.token_urlsafe(24) → ~32 chars
