"""Import integration — the full HTTP path.

The end-to-end counterpart of `test_export.py`: drives `import_package` over the
real server. Import reads an `.apkg`/`.colpkg` from a **server-local path**, gated
by `--import-path-root` on a purely-local daemon (the read counterpart of export's
`--export-path-root`), and merges it into the collection. Because it touches
collection-wide state and needs a startup flag, it uses a dedicated, class-scoped
server.
"""

from __future__ import annotations

import os

import pytest

from shrike.client import ServerError, ShrikeClient
from tests.integration.conftest import ServerInfo

pytestmark = pytest.mark.integration


def _seed(client: ShrikeClient) -> None:
    client.upsert_notes(
        [
            {"deck": "ImportTest", "note_type": "Basic", "fields": {"Front": "iq1", "Back": "ia1"}},
            {"deck": "ImportTest", "note_type": "Basic", "fields": {"Front": "iq2", "Back": "ia2"}},
        ],
        on_duplicate="allow",
    )


@pytest.fixture(scope="module")
def rooted_server(server_factory, tmp_path_factory) -> tuple[ServerInfo, str]:
    # One root serves both legs of the round-trip: export writes the package into
    # it, import reads from it. Both capabilities are gated on this root AND the
    # server being purely-local (loopback, no --allow-remote, …).
    root = tmp_path_factory.mktemp("import-root")
    server = server_factory(
        "import-rooted",
        extra_args=["--export-path-root", str(root), "--import-path-root", str(root)],
    )
    return server, str(root)


class TestServerLocalImport:
    def test_export_then_import_round_trips(self, rooted_server: tuple[ServerInfo, str]) -> None:
        server, root = rooted_server
        with ShrikeClient(server.url, autostart=False) as client:
            _seed(client)
            pkg = os.path.join(root, "round-trip.colpkg")
            exported = client.export_package(format="colpkg", output_path=pkg)
            assert exported.delivery == "path"
            assert os.path.isfile(pkg)

            # Import the package back. It is a MERGE into the same collection — the
            # notes already exist by GUID, so they land in the updated/duplicate
            # buckets rather than `new`, but the importer ran end-to-end over the
            # real HTTP path and accounted for every note the package carried.
            result = client.import_package(pkg)
            assert result.found_notes >= 2
            buckets = (
                result.new
                + result.updated
                + result.duplicate
                + result.conflicting
                + result.first_field_match
                + result.missing_notetype
                + result.missing_deck
                + result.empty_first_field
            )
            assert buckets == result.found_notes, (
                "every note in the package must land in exactly one bucket"
            )

    def test_import_outside_root_is_rejected(
        self, rooted_server: tuple[ServerInfo, str], tmp_path
    ) -> None:
        server, _root = rooted_server
        with ShrikeClient(server.url, autostart=False) as client:
            # Outside any import root — the capability gate refuses it before any
            # read, so the file need not even exist (the path is the attack surface,
            # not its contents).
            outside = str(tmp_path / "outside.colpkg")
            with pytest.raises(ServerError, match="not permitted"):
                client.import_package(outside)

    def test_import_of_a_non_package_file_errors(
        self, rooted_server: tuple[ServerInfo, str]
    ) -> None:
        server, root = rooted_server
        with ShrikeClient(server.url, autostart=False) as client:
            # Inside the root (passes the path-root gate) but not a valid package:
            # anki's importer must reject it with a clean error, not crash the server.
            junk = os.path.join(root, "not-a-package.colpkg")
            with open(junk, "wb") as f:
                f.write(b"this is not a zip nor an anki package")
            with pytest.raises(ServerError):
                client.import_package(junk)
