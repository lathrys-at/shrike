"""The shared server-local path-safety mechanism (#71 S1).

These helpers gate every server-local filesystem capability (store_media #164,
export #71, import #72). The read gate (``path_within_any_root``) and the
purely-local / validate-root helpers are exercised through server.py's aliases
in test_server_security.py; this pins the mechanism directly, with focus on the
NEW write gate (``output_path_within_any_root``) — the export #71 case where the
target file does not exist yet.
"""

from __future__ import annotations

import os

import pytest

from shrike import pathsafety


class TestServerIsPurelyLocal:
    def test_default_loopback_is_purely_local(self) -> None:
        assert pathsafety.server_is_purely_local(
            "127.0.0.1",
            allow_remote=False,
            no_dns_rebinding_protection=False,
            allowed_hosts=[],
            allowed_origins=[],
        )

    @pytest.mark.parametrize(
        "override",
        [
            {"allow_remote": True},
            {"no_dns_rebinding_protection": True},
            {"allowed_hosts": ["proxy.internal"]},
            {"allowed_origins": ["https://proxy.internal"]},
        ],
    )
    def test_any_remote_signal_disables_it(self, override) -> None:
        base = {
            "allow_remote": False,
            "no_dns_rebinding_protection": False,
            "allowed_hosts": [],
            "allowed_origins": [],
        }
        base.update(override)
        assert not pathsafety.server_is_purely_local("127.0.0.1", **base)

    def test_non_loopback_bind_is_not_purely_local(self) -> None:
        assert not pathsafety.server_is_purely_local(
            "0.0.0.0",
            allow_remote=False,
            no_dns_rebinding_protection=False,
            allowed_hosts=[],
            allowed_origins=[],
        )


class TestValidatePathRoot:
    def test_accepts_an_existing_dir_realpathed(self, tmp_path) -> None:
        root = tmp_path / "exports"
        root.mkdir()
        assert pathsafety.validate_path_root(str(root)) == os.path.realpath(str(root))

    def test_rejects_filesystem_root(self) -> None:
        with pytest.raises(ValueError, match="confines nothing"):
            pathsafety.validate_path_root("/")

    def test_rejects_missing_dir(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="not an existing directory"):
            pathsafety.validate_path_root(str(tmp_path / "nope"))


class TestReadGate:
    """``path_within_any_root`` — an existing file inside a root."""

    def test_contained_existing_file(self, tmp_path) -> None:
        root = tmp_path / "media"
        root.mkdir()
        f = root / "a.png"
        f.write_text("x")
        assert pathsafety.path_within_any_root(str(f), [str(root)])

    def test_outside_root_rejected(self, tmp_path) -> None:
        root = tmp_path / "media"
        root.mkdir()
        other = tmp_path / "other.png"
        other.write_text("x")
        assert not pathsafety.path_within_any_root(str(other), [str(root)])

    def test_missing_file_rejected(self, tmp_path) -> None:
        root = tmp_path / "media"
        root.mkdir()
        assert not pathsafety.path_within_any_root(str(root / "ghost.png"), [str(root)])

    def test_empty_roots_deny_everything(self, tmp_path) -> None:
        f = tmp_path / "a"
        f.write_text("x")
        assert not pathsafety.path_within_any_root(str(f), [])

    def test_prefix_sibling_not_contained(self, tmp_path) -> None:
        # /media must NOT match /media-evil (commonpath, not startswith).
        root = tmp_path / "media"
        root.mkdir()
        evil = tmp_path / "media-evil"
        evil.mkdir()
        f = evil / "x"
        f.write_text("x")
        assert not pathsafety.path_within_any_root(str(f), [str(root)])


class TestWriteGate:
    """``output_path_within_any_root`` — a to-be-created file inside a root
    (the export #71 case; the target does not exist yet)."""

    def test_nonexistent_target_in_existing_root_is_allowed(self, tmp_path) -> None:
        root = tmp_path / "exports"
        root.mkdir()
        # The .apkg doesn't exist yet — the write gate checks the parent dir.
        target = root / "deck.apkg"
        assert not target.exists()
        assert pathsafety.output_path_within_any_root(str(target), [str(root)])

    def test_target_in_a_subdir_of_root(self, tmp_path) -> None:
        root = tmp_path / "exports"
        (root / "sub").mkdir(parents=True)
        target = root / "sub" / "deck.apkg"
        assert pathsafety.output_path_within_any_root(str(target), [str(root)])

    def test_target_outside_root_rejected(self, tmp_path) -> None:
        root = tmp_path / "exports"
        root.mkdir()
        target = tmp_path / "elsewhere.apkg"
        assert not pathsafety.output_path_within_any_root(str(target), [str(root)])

    def test_dotdot_escape_rejected(self, tmp_path) -> None:
        root = tmp_path / "exports"
        root.mkdir()
        # A ..-escape resolves out of the root → rejected (the parent realpath
        # collapses the ..).
        target = root / "sub" / ".." / ".." / "escaped.apkg"
        (root / "sub").mkdir()
        assert not pathsafety.output_path_within_any_root(str(target), [str(root)])

    def test_symlinked_parent_escape_rejected(self, tmp_path) -> None:
        # A symlink whose target is outside the root must not let a write escape:
        # realpath of the parent collapses the symlink before containment.
        root = tmp_path / "exports"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        link = root / "link"
        link.symlink_to(outside)  # exports/link -> ../outside
        target = link / "deck.apkg"
        assert not pathsafety.output_path_within_any_root(str(target), [str(root)])

    def test_missing_parent_dir_rejected(self, tmp_path) -> None:
        # The destination directory must already exist (fail-closed).
        root = tmp_path / "exports"
        root.mkdir()
        target = root / "no-such-subdir" / "deck.apkg"
        assert not pathsafety.output_path_within_any_root(str(target), [str(root)])

    def test_empty_roots_deny_everything(self, tmp_path) -> None:
        target = tmp_path / "deck.apkg"
        assert not pathsafety.output_path_within_any_root(str(target), [])

    def test_prefix_sibling_root_not_matched(self, tmp_path) -> None:
        root = tmp_path / "exports"
        root.mkdir()
        sibling = tmp_path / "exports-evil"
        sibling.mkdir()
        target = sibling / "deck.apkg"
        assert not pathsafety.output_path_within_any_root(str(target), [str(root)])
