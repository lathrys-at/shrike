"""Export integration — the full HTTP path.

Drives `export_package` over the real server: the default download-url delivery
(export → GET the url → valid package bytes → reaped one-shot) and the
server-local `output_path` delivery (gated by --export-path-root on a
purely-local daemon). Uses dedicated servers (export touches collection-wide
state and the path-root one needs a startup flag), module-scoped where possible.
"""

from __future__ import annotations

import io
import zipfile

import httpx
import pytest

from shrike.client import ShrikeClient
from tests.integration.conftest import ServerInfo

pytestmark = pytest.mark.integration


def _seed(client: ShrikeClient) -> None:
    client.upsert_notes(
        [
            {"deck": "ExportTest", "note_type": "Basic", "fields": {"Front": "q1", "Back": "a1"}},
            {"deck": "ExportTest", "note_type": "Basic", "fields": {"Front": "q2", "Back": "a2"}},
        ],
        on_duplicate="allow",
    )


def _is_apkg(data: bytes) -> bool:
    """A .apkg/.colpkg is a zip carrying a collection db."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return any(n.startswith("collection.anki2") for n in z.namelist())
    except zipfile.BadZipFile:
        return False


@pytest.fixture(scope="module")
def export_server(server_factory) -> ServerInfo:
    return server_factory("export-module")


class TestDownloadUrlDelivery:
    def test_export_then_download_round_trips(self, export_server: ServerInfo) -> None:
        with ShrikeClient(export_server.url, autostart=False) as client:
            _seed(client)
            result = client.export_package(format="apkg")
            assert result.delivery == "url"
            assert result.note_count >= 2
            assert result.url and "/export/" in result.url

            # GET the url → real package bytes (a valid zip with the collection).
            data = client.download_export(result.url)
            assert _is_apkg(data)
            assert len(data) == result.bytes

            # One-shot: the token is reaped after download → a re-GET 404s.
            assert httpx.get(result.url).status_code == 404

    def test_deck_scoped_download(self, export_server: ServerInfo) -> None:
        with ShrikeClient(export_server.url, autostart=False) as client:
            _seed(client)
            result = client.export_package(deck="ExportTest", format="apkg")
            assert result.delivery == "url"
            data = client.download_export(result.url)
            assert _is_apkg(data)

    def test_unknown_export_token_404s(self, export_server: ServerInfo) -> None:
        base = export_server.url.rsplit("/", 1)[0]
        assert httpx.get(f"{base}/export/no-such-token").status_code == 404


class TestServerLocalPathDelivery:
    @pytest.fixture(scope="class")
    def root_server(self, server_factory, tmp_path_factory) -> tuple[ServerInfo, str]:
        root = tmp_path_factory.mktemp("export-root")
        server = server_factory("export-rooted", extra_args=["--export-path-root", str(root)])
        return server, str(root)

    def test_output_path_writes_on_the_server(self, root_server) -> None:
        server, root = root_server
        with ShrikeClient(server.url, autostart=False) as client:
            _seed(client)
            import os

            out = os.path.join(root, "backup.colpkg")
            result = client.export_package(format="colpkg", output_path=out)
            assert result.delivery == "path"
            assert result.path == out
            # The server wrote it (this test shares the server's disk).
            assert os.path.isfile(out)
            with open(out, "rb") as f:
                assert _is_apkg(f.read())

    def test_output_path_outside_root_is_rejected(self, root_server, tmp_path) -> None:
        server, _root = root_server
        from shrike.client import ServerError

        with ShrikeClient(server.url, autostart=False) as client:
            out = str(tmp_path / "escape.apkg")  # outside any export root
            with pytest.raises(ServerError, match="output_path is not permitted"):
                client.export_package(format="apkg", output_path=out)
