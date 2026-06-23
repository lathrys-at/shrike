"""Unit coverage for server.py module-level helpers and the non-loopback bind
guard. (The async route handlers and full startup are covered by the integration
suite.)"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from shrike.harness.collection import collect_embed_inputs
from shrike.server import main
from shrike.server.server import (
    _host_header_form,
    _make_image_resolver,
    _positive_int,
)


class TestMakeImageResolver:
    """The lock-free (read, exists) pair the index uses to fold media into a
    note's embed hash. Both sanitize to a basename inside the media dir, so a
    name can only ever resolve under the folder."""

    def test_reads_existing_file_inside_dir(self, tmp_path: Path) -> None:
        (tmp_path / "pic.png").write_bytes(b"PNGDATA")
        read, exists = _make_image_resolver(str(tmp_path))
        assert read("pic.png") == b"PNGDATA"
        assert exists("pic.png") is True

    def test_missing_file_reads_none_and_does_not_exist(self, tmp_path: Path) -> None:
        read, exists = _make_image_resolver(str(tmp_path))
        assert read("absent.png") is None
        assert exists("absent.png") is False

    def test_traversal_name_is_confined_to_the_dir(self, tmp_path: Path) -> None:
        # A traversal attempt is reduced to a basename, so it can never escape the
        # media dir to read an arbitrary file — it just misses inside the dir.
        outside = tmp_path.parent / "secret.txt"
        outside.write_bytes(b"TOPSECRET")
        media = tmp_path / "media"
        media.mkdir()
        read, exists = _make_image_resolver(str(media))
        assert read("../secret.txt") is None
        assert exists("../secret.txt") is False

    def test_empty_name_resolves_to_nothing(self, tmp_path: Path) -> None:
        # _safe_media_name('') yields no name → both short-circuit to absent.
        read, exists = _make_image_resolver(str(tmp_path))
        assert read("") is None
        assert exists("") is False


class TestHostHeaderForm:
    """The Host-header spelling of a bind host for the allow-list: an IPv6 literal
    is canonicalized + bracketed, an IPv4 address canonicalized, a name passed
    through verbatim — all wildcarded on port."""

    def test_ipv6_is_canonicalized_and_bracketed(self) -> None:
        assert _host_header_form("::1") == "[::1]:*"
        # An already-bracketed bind host is stripped before canonicalization.
        assert _host_header_form("[::1]") == "[::1]:*"

    def test_ipv4_is_canonicalized(self) -> None:
        assert _host_header_form("127.0.0.2") == "127.0.0.2:*"

    def test_name_is_passed_through_verbatim(self) -> None:
        # A non-address bind host (e.g. "localhost") can't be parsed as an IP, so
        # it's passed through unchanged — the ValueError fallback branch.
        assert _host_header_form("localhost") == "localhost:*"


class TestPositiveInt:
    """The argparse type for >= 1 integer flags (e.g. --embedding-batch-size)."""

    def test_accepts_one_and_above(self) -> None:
        assert _positive_int("1") == 1
        assert _positive_int("42") == 42

    @pytest.mark.parametrize("raw", ["0", "-1"])
    def test_rejects_below_one(self, raw: str) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int(raw)


class TestCollectForRebuild:
    def test_gathers_ids_mod_and_texts(self, wrapper, basic_note):
        inputs, col_mod = wrapper.run_sync(collect_embed_inputs)
        assert [i.note_id for i in inputs] == [basic_note]
        assert isinstance(col_mod, int)
        assert len(inputs) == 1
        assert "2+2" in inputs[0].text

    def test_empty_collection(self, wrapper):
        inputs, _col_mod = wrapper.run_sync(collect_embed_inputs)
        assert inputs == []


class TestNonLoopbackGuard:
    def test_refuses_non_loopback_without_allow_remote(self, tmp_path):
        argv = [
            "shrike-server",
            "--collection",
            str(tmp_path / "c.anki2"),
            "--host",
            "0.0.0.0",
        ]
        with (
            patch("sys.argv", argv),
            patch("shrike.server.server.configure_logging", return_value=tmp_path),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 1
