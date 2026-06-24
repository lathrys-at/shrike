"""The ExportStore: token lifecycle (resolve/reap one-shot, close, TTL sweep)
and token unguessability.

The kernel-level export round-trip + symlink safety are pinned in
tests/native/test_export_package.py and the shrike-collection Rust tests.
"""

from __future__ import annotations

import os

import pytest

from shrike.server.export_store import ExportStore


class TestExportStore:
    def test_resolve_then_reap_is_one_shot(self, tmp_path) -> None:
        store = ExportStore(str(tmp_path / "cache"))
        token, path = store.new_temp_path(suffix=".apkg")
        with open(path, "wb") as f:
            f.write(b"pkg")
        store.register(token, path, "apkg")
        assert store.resolve(token) == path
        store.reap(token)
        assert store.resolve(token) is None
        assert not os.path.exists(path)

    @pytest.mark.xfail(
        strict=True,
        reason="#1016: resolve is non-consuming, so a fast-follow second GET still "
        "resolves the token and races the post-response reap into a 500",
    )
    def test_resolve_is_one_shot(self, tmp_path) -> None:
        # The one-shot reap runs as a post-response background task, *after* the
        # first GET's body is streamed — so it can land after the client has
        # already issued a second GET. If that second GET can still resolve the
        # token, it serves a file the reap is concurrently removing → 500 on the
        # vanished file. The store must consume the token on the first resolve so
        # the second GET deterministically misses (a clean 404), never a 500.
        store = ExportStore(str(tmp_path / "cache"))
        token, path = store.new_temp_path(suffix=".apkg")
        with open(path, "wb") as f:
            f.write(b"pkg")
        store.register(token, path, "apkg")
        assert store.resolve(token) == path  # first GET resolves and serves
        assert store.resolve(token) is None  # second GET must miss (one-shot)

    def test_close_reaps_pending(self, tmp_path) -> None:
        store = ExportStore(str(tmp_path / "cache"))
        token, path = store.new_temp_path(suffix=".colpkg")
        with open(path, "wb") as f:
            f.write(b"pkg")
        store.register(token, path, "colpkg")
        store.close()
        assert not os.path.exists(path)
        assert store.resolve(token) is None

    def test_ttl_sweep_reaps_expired(self, tmp_path) -> None:
        store = ExportStore(str(tmp_path / "cache"), ttl_seconds=0.0)
        token, path = store.new_temp_path(suffix=".apkg")
        with open(path, "wb") as f:
            f.write(b"pkg")
        store.register(token, path, "apkg")
        # ttl=0 → the next resolve sweeps it as expired.
        assert store.resolve(token) is None
        assert not os.path.exists(path)

    def test_token_is_unguessable(self, tmp_path) -> None:
        store = ExportStore(str(tmp_path / "cache"))
        t1, _ = store.new_temp_path(suffix=".apkg")
        t2, _ = store.new_temp_path(suffix=".apkg")
        assert t1 != t2
        assert len(t1) >= 24  # secrets.token_urlsafe(24) → ~32 chars
