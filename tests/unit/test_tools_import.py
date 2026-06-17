"""The import_package action — path-root gate + routing to the kernel op (#72 S2).

The drift-rebuild correctness is pinned natively (tests/native/test_import_package.py);
this pins the action layer: the server-local-path gate (off by default, containment),
the conflict-option passthrough, and the response shape.
"""

from __future__ import annotations

import json

import pytest

from shrike.api.actions import ActionContext, ToolInputError, build_actions


class _StubKernel:
    """Records the import_package call and returns a canned summary."""

    def __init__(self):
        self.calls: list[tuple] = []

    async def import_package(self, path, update_notes, update_notetypes, with_sched, merge):
        self.calls.append((path, update_notes, update_notetypes, with_sched, merge))
        summary = {
            "new": 3,
            "updated": 1,
            "duplicate": 0,
            "conflicting": 0,
            "first_field_match": 0,
            "missing_notetype": 0,
            "missing_deck": 0,
            "empty_first_field": 0,
            "found_notes": 4,
        }
        return (json.dumps(summary), True)


def _import_action(kernel, import_roots, *, purely_local=True):
    ctx = ActionContext(
        wrapper=object(),
        kernel=kernel,
        server_import_path_roots=import_roots,
        server_purely_local=purely_local,
    )
    return {a.name: a for a in build_actions(ctx)}["import_package"].impl


class TestPathRootGate:
    async def test_disabled_when_no_root_configured(self):
        # No root configured → the read gate's empty-roots branch denies.
        kernel = _StubKernel()
        action = _import_action(kernel, None)
        with pytest.raises(ToolInputError, match="not permitted"):
            await action(path="/anywhere/deck.apkg")
        assert kernel.calls == []  # never reached the kernel

    async def test_disabled_when_not_purely_local(self, tmp_path):
        # A configured root but a non-purely-local server → denied (the
        # server-level resolution would have emptied the roots, but the action
        # also requires server_purely_local — defense in depth).
        root = tmp_path / "imports"
        root.mkdir()
        pkg = root / "deck.apkg"
        pkg.write_bytes(b"PK\x03\x04")
        kernel = _StubKernel()
        action = _import_action(kernel, [str(root)], purely_local=False)
        with pytest.raises(ToolInputError, match="not permitted"):
            await action(path=str(pkg))
        assert kernel.calls == []

    async def test_path_outside_root_rejected(self, tmp_path):
        root = tmp_path / "imports"
        root.mkdir()
        kernel = _StubKernel()
        action = _import_action(kernel, [str(root)])
        with pytest.raises(ToolInputError, match="not permitted"):
            await action(path=str(tmp_path / "elsewhere" / "deck.apkg"))
        assert kernel.calls == []

    async def test_path_within_root_routes_to_kernel(self, tmp_path):
        root = tmp_path / "imports"
        root.mkdir()
        pkg = root / "deck.apkg"
        pkg.write_bytes(b"PK\x03\x04")  # a file under the root (content irrelevant here)
        kernel = _StubKernel()
        action = _import_action(kernel, [str(root)])

        resp = await action(path=str(pkg))
        assert len(kernel.calls) == 1
        # Defaults flow through: if_newer / if_newer / no scheduling / no merge.
        called_path, un, unt, sched, merge = kernel.calls[0]
        assert un == "if_newer" and unt == "if_newer" and sched is False and merge is False
        # The response folds the summary + the reindexed flag.
        assert resp.new == 3
        assert resp.updated == 1
        assert resp.found_notes == 4
        assert resp.reindexed is True

    async def test_traversal_escape_rejected(self, tmp_path):
        # A `..` that climbs out of the root must not pass (commonpath on the
        # realpath'd sides, not a prefix string match).
        root = tmp_path / "imports"
        root.mkdir()
        kernel = _StubKernel()
        action = _import_action(kernel, [str(root)])
        with pytest.raises(ToolInputError, match="not permitted"):
            await action(path=str(root / ".." / "secret.apkg"))
        assert kernel.calls == []


class TestConflictOptions:
    async def test_options_passed_through(self, tmp_path):
        root = tmp_path / "imports"
        root.mkdir()
        pkg = root / "deck.apkg"
        pkg.write_bytes(b"PK\x03\x04")
        kernel = _StubKernel()
        action = _import_action(kernel, [str(root)])
        await action(
            path=str(pkg),
            update_notes="always",
            update_notetypes="never",
            with_scheduling=True,
            merge_notetypes=True,
        )
        _, un, unt, sched, merge = kernel.calls[0]
        assert (un, unt, sched, merge) == ("always", "never", True, True)
