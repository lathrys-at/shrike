"""`shrike collection export` CLI (#71 S2): the dual download/server-path strategy + flags.

The CLI talks to a mocked client; the export action + route are covered in
test_tools_export.py / test_export.py (integration). These pin the CLI's own
logic: format inference, the deck/ids and colpkg-scope guards, downloading
to DEST vs --server-path, and the JSON output shape.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from shrike.cli import cli
from shrike.schemas import ExportPackagePath, ExportPackageUrl


def _run(tmp_path, args, *, export_return=None, download_bytes=b"PKGDATA"):
    cfg = tmp_path / "config.yml"
    cfg.write_text("")
    fake = MagicMock()
    fake.export_package.return_value = export_return
    fake.download_export.return_value = download_bytes
    with patch("shrike.client.ShrikeClient", return_value=fake):
        result = CliRunner().invoke(cli, ["--config", str(cfg), *args])
    return result, fake


def _url_result(fmt="apkg"):
    return ExportPackageUrl(
        delivery="url", note_count=3, bytes=7, format=fmt, url="http://h/export/tok"
    )


def _path_result(path, fmt="colpkg"):
    return ExportPackagePath(delivery="path", note_count=3, bytes=7, format=fmt, path=path)


class TestDownloadDelivery:
    def test_downloads_url_to_dest(self, tmp_path):
        dest = tmp_path / "out.apkg"
        result, fake = _run(
            tmp_path,
            ["collection", "export", str(dest)],
            export_return=_url_result(),
            download_bytes=b"ZIPDATA",
        )
        assert result.exit_code == 0, result.output
        # The CLI called export (no output_path → download), then wrote DEST.
        assert fake.export_package.call_args.kwargs["output_path"] is None
        assert dest.read_bytes() == b"ZIPDATA"
        assert "Exported" in result.output and "3" in result.output

    def test_format_inferred_from_dest_extension(self, tmp_path):
        dest = tmp_path / "backup.colpkg"
        result, fake = _run(
            tmp_path, ["collection", "export", str(dest)], export_return=_url_result("colpkg")
        )
        assert result.exit_code == 0, result.output
        assert fake.export_package.call_args.kwargs["format"] == "colpkg"

    def test_deck_scope_passed_through(self, tmp_path):
        dest = tmp_path / "d.apkg"
        result, fake = _run(
            tmp_path,
            ["collection", "export", str(dest), "--deck", "Spanish"],
            export_return=_url_result(),
        )
        assert result.exit_code == 0, result.output
        assert fake.export_package.call_args.kwargs["deck"] == "Spanish"

    def test_json_output(self, tmp_path):
        dest = tmp_path / "out.apkg"
        result, _ = _run(
            tmp_path,
            ["--json", "collection", "export", str(dest)],
            export_return=_url_result(),
            download_bytes=b"ABCD",
        )
        assert result.exit_code == 0, result.output
        assert '"note_count": 3' in result.output
        assert '"bytes": 4' in result.output  # len(b"ABCD")


class TestServerPathDelivery:
    def test_server_path_no_download(self, tmp_path):
        result, fake = _run(
            tmp_path,
            ["collection", "export", "--server-path", "/srv/exports/b.colpkg"],
            export_return=_path_result("/srv/exports/b.colpkg"),
        )
        assert result.exit_code == 0, result.output
        # output_path is passed; no download happens.
        assert fake.export_package.call_args.kwargs["output_path"] == "/srv/exports/b.colpkg"
        fake.download_export.assert_not_called()
        assert "on the server" in result.output


class TestValidation:
    def test_deck_and_ids_mutually_exclusive(self, tmp_path):
        result, _ = _run(tmp_path, ["collection", "export", "x.apkg", "--deck", "A", "--ids", "1"])
        assert result.exit_code != 0
        assert "at most one of --deck or --ids" in result.output

    def test_colpkg_with_deck_rejected(self, tmp_path):
        result, _ = _run(tmp_path, ["collection", "export", "x.colpkg", "--deck", "A"])
        assert result.exit_code != 0
        assert "whole-collection backup" in result.output

    def test_requires_dest_or_server_path(self, tmp_path):
        result, _ = _run(tmp_path, ["collection", "export"])
        assert result.exit_code != 0
        assert "DEST" in result.output or "server-path" in result.output
