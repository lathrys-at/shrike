"""The ExportStore: token lifecycle (claim/reap one-shot, close, TTL sweep)
and token unguessability.

The kernel-level export round-trip + symlink safety are pinned in
tests/native/test_export_package.py and the shrike-collection Rust tests.
"""

from __future__ import annotations

import logging
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
        # claim() consumes the token synchronously, so a second GET misses on the
        # strength of the mark alone — no reap runs below. (The real reap is a
        # post-response background task that can land after the client's second
        # GET; consuming on claim is what makes that second GET a clean 404 rather
        # than a race against the reap.)
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

    def test_ttl_sweep_reaps_a_claimed_entry(self, tmp_path) -> None:
        # An aborted stream leaves the token claimed but unreaped (its reap never
        # runs). The TTL sweep must still collect its temp — it reaps on age, not
        # on claimed state — so a collection-bearing file can't linger past the
        # TTL when the post-response reap is skipped.
        ttl = 3600.0
        store = ExportStore(str(tmp_path / "cache"), ttl_seconds=ttl)
        token, path = store.new_temp_path(suffix=".apkg")
        with open(path, "wb") as f:
            f.write(b"pkg")
        store.register(token, path, "apkg")
        assert store.claim(token) == path  # claimed; the reap never runs
        # Age the claimed entry past the TTL, then sweep (mutating created is the
        # deterministic stand-in for wall-clock passing — claim() can't be reached
        # with ttl=0 since it sweeps before it marks).
        store._pending[token].created -= ttl + 1.0
        store._sweep_expired()
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

    def test_dir_is_the_exports_subdir(self, tmp_path) -> None:
        store = ExportStore(str(tmp_path / "cache"))
        assert store.dir == os.path.join(str(tmp_path / "cache"), "exports")

    def test_resolve_drops_the_entry_when_the_file_vanished(self, tmp_path) -> None:
        # The file is deleted out from under the store (a manual delete or a
        # prior reap racing this resolve): resolve must return None AND forget
        # the entry, so a second resolve is a clean None, never a KeyError.
        store = ExportStore(str(tmp_path / "cache"))
        token, path = store.new_temp_path(suffix=".apkg")
        with open(path, "wb") as f:
            f.write(b"pkg")
        store.register(token, path, "apkg")
        os.remove(path)
        assert store.resolve(token) is None
        assert store.resolve(token) is None

    def test_reap_of_an_unknown_token_is_a_noop(self, tmp_path) -> None:
        # Reaping a token that was never registered (or already reaped) must not
        # raise — the BackgroundTask reap fires unconditionally after a stream.
        store = ExportStore(str(tmp_path / "cache"))
        store.reap("never-registered")

    def test_reap_swallows_a_non_filenotfound_oserror_and_warns(self, tmp_path, caplog) -> None:
        # The registered path is a directory, so os.remove raises an OSError
        # that is NOT FileNotFoundError — the warn-and-continue arm (a leftover
        # temp the reap couldn't clear must never propagate out of the
        # BackgroundTask and 500 the next request).
        store = ExportStore(str(tmp_path / "cache"))
        token, path = store.new_temp_path(suffix=".apkg")
        os.makedirs(path)
        with open(os.path.join(path, "child"), "wb") as f:
            f.write(b"x")
        store.register(token, path, "apkg")
        with caplog.at_level(logging.WARNING, logger="shrike.export"):
            store.reap(token)  # must not raise
        assert any("could not reap" in r.message for r in caplog.records)
