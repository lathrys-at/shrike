"""The legacy-config degrade-on-both-paths contract (CLI + daemon --config)."""

from __future__ import annotations

import sys

import pytest

from shrike.cli.config import save_config


@pytest.fixture(autouse=True)
def _clean_embedding_env(monkeypatch) -> None:
    """Keep resolve tests independent of the ambient environment."""
    for var in (
        "SHRIKE_EMBEDDING_MODEL",
        "SHRIKE_EMBEDDING_PORT",
        "SHRIKE_EMBEDDING_POOLING",
        "SHRIKE_EMBEDDING_ARGS",
        "LLAMA_SERVER_PATH",
        "SHRIKE_CACHE_DIR",
        "SHRIKE_INDEX_SAVE_DELAY",
        "SHRIKE_INDEX_SAVE_THRESHOLD",
        "SHRIKE_ALLOWED_HOSTS",
        "SHRIKE_ALLOWED_ORIGINS",
        "SHRIKE_NO_DNS_REBINDING_PROTECTION",
    ):
        monkeypatch.delenv(var, raising=False)


class TestLegacyConfigSameOutcomeBothPaths:
    """The SAME legacy config must DEGRADE on BOTH launch paths.

    A legacy config (no v2 sections) whose synthesized capabilities the build
    can't serve — e.g. ``backend: llama`` (→ a ``remote`` entry needing the
    managed llama-server) on a build lacking that engine — must degrade on both
    paths: the CLI's ``resolve_embedding_profile`` short-circuits ``caps.legacy``
    to the no-build-validation ``resolve_embedding`` (degrade-with-warning), and
    the daemon ``--config`` path must degrade too rather than running the build-
    validating ``resolve_profile`` → ``ProfileError`` → refuse to boot.
    """

    # backend defaults to "llama" when a model is present → a remote entry that
    # needs the managed llama-server, which a mismatched build doesn't compile.
    LEGACY = {"embedding": {"model": "/m.gguf", "pooling": "last"}}
    # A build that compiles onnx but NOT the remote/managed-llama runtime the
    # migrated legacy entry needs → resolve_profile would raise on it.
    MISMATCHED_BUILD = ("engine-ort",)

    def test_migrated_legacy_would_fail_build_validation(self) -> None:
        # The same caps that the legacy path degrades WOULD raise via
        # resolve_profile on the mismatched build.
        from shrike.harness.profiles import ProfileError, parse_capabilities, resolve_profile

        caps = parse_capabilities(dict(self.LEGACY))
        assert caps.legacy is True
        with pytest.raises(ProfileError):
            resolve_profile(caps, self.MISMATCHED_BUILD)

    def test_cli_path_degrades_the_legacy_config(self, capsys) -> None:
        # The CLI launch path: degrade-with-warning, no build validation.
        from shrike.cli.config import resolve_embedding_profile

        resolved = resolve_embedding_profile(dict(self.LEGACY), None)
        assert resolved["model"] == "/m.gguf"
        assert resolved["pooling"] == "last"
        assert "deprecated" in capsys.readouterr().err

    def test_daemon_config_path_degrades_not_refuse_boot(self, monkeypatch, tmp_path) -> None:
        # The daemon --config launch path drives the REAL server.main() far
        # enough to resolve the config, on the mismatched build. The legacy
        # branch must degrade and let boot proceed to building the
        # EmbeddingRuntime — which we intercept with a sentinel to stop main()
        # before it opens a collection / starts asyncio.
        import shrike.server.server as server

        config_path = tmp_path / "legacy.yml"
        save_config(dict(self.LEGACY), config_path)

        # A build that can't serve the migrated legacy entry.
        monkeypatch.setattr(
            server.shrike_native, "build_features", lambda: set(self.MISMATCHED_BUILD)
        )

        # Let the daemon lock acquire as a no-op (don't touch real state files).
        class _FakeLock:
            def __init__(self, *a, **k) -> None:
                pass

            def acquire(self, *a, **k) -> None:
                pass

        monkeypatch.setattr(server, "ServerLock", _FakeLock)

        # Intercept right after resolution: EmbeddingRuntime construction is the
        # first thing main() does past the --config resolution block.
        class _Reached(Exception):
            pass

        captured: dict[str, object] = {}

        def _sentinel(*args, **kwargs):
            captured["backend"] = kwargs.get("backend")
            captured["model"] = kwargs.get("model")
            captured["pooling"] = kwargs.get("pooling")
            raise _Reached

        monkeypatch.setattr(server, "EmbeddingRuntime", _sentinel)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "shrike-server",
                "--collection",
                str(tmp_path / "c.anki2"),
                "--config",
                str(config_path),
            ],
        )

        # Reaching the sentinel proves the daemon DEGRADED (did not refuse-boot):
        # a SystemExit from parser.error would mean it still refused.
        with pytest.raises(_Reached):
            server.main()

        # And it degraded to the SAME params the CLI path produces.
        from shrike.cli.config import resolve_embedding_profile

        cli = resolve_embedding_profile(dict(self.LEGACY), None, quiet=True)
        assert captured["model"] == cli["model"] == "/m.gguf"
        assert captured["pooling"] == cli["pooling"] == "last"

    def test_v2_config_error_still_refuses_boot(self, monkeypatch, tmp_path) -> None:
        # The boundary: a REAL v2-config error must still refuse-boot (resolve_profile
        # validation is unchanged for v2). A v2 onnx entry on a build without the onnx
        # engine raises ProfileError → parser.error → SystemExit.
        import shrike.server.server as server

        v2 = {"embedders": [{"modalities": ["text"], "runtime": "onnx", "model": "/m"}]}
        config_path = tmp_path / "v2.yml"
        save_config(dict(v2), config_path)

        monkeypatch.setattr(server.shrike_native, "build_features", lambda: {"manage-llama"})

        class _FakeLock:
            def __init__(self, *a, **k) -> None:
                pass

            def acquire(self, *a, **k) -> None:
                pass

        monkeypatch.setattr(server, "ServerLock", _FakeLock)
        # If main() reached runtime construction, the v2 error wasn't caught — fail loud.
        monkeypatch.setattr(
            server,
            "EmbeddingRuntime",
            lambda *a, **k: pytest.fail("v2 config error must refuse-boot, not reach runtime"),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "shrike-server",
                "--collection",
                str(tmp_path / "c.anki2"),
                "--config",
                str(config_path),
            ],
        )

        with pytest.raises(SystemExit):
            server.main()
