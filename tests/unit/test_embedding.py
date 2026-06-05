"""Tests for the shrike.embedding module."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from shrike.embed_text import EMBED_TEXT_VERSION
from shrike.embedding import EmbeddingRuntime, EmbeddingService


@pytest.fixture()
def svc(tmp_path: Path) -> EmbeddingService:
    return EmbeddingService(
        model="/fake/model.gguf",
        port=19999,
        log_dir=tmp_path / "logs",
    )


def _set_running(svc: EmbeddingService, pid: int = 123) -> None:
    """Make a service look like it has a live llama-server subprocess."""
    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = pid
    svc._process = proc


class TestInit:
    def test_defaults(self) -> None:
        svc = EmbeddingService(model="/path/model.gguf")
        assert svc.url == "http://127.0.0.1:8373"
        assert svc.running is False

    def test_custom_host_port(self) -> None:
        svc = EmbeddingService(model="/m.gguf", host="0.0.0.0", port=9000)
        assert svc.url == "http://0.0.0.0:9000"

    def test_running_false_before_start(self, svc: EmbeddingService) -> None:
        assert svc.running is False


class TestFindLlamaServer:
    def test_env_var_valid(self, svc: EmbeddingService, tmp_path: Path) -> None:
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        with patch.dict(os.environ, {"LLAMA_SERVER_PATH": str(binary)}):
            assert svc._find_llama_server() == str(binary)

    def test_env_var_invalid(self, svc: EmbeddingService) -> None:
        with (
            patch.dict(os.environ, {"LLAMA_SERVER_PATH": "/nonexistent/binary"}),
            pytest.raises(FileNotFoundError, match="does not point to an executable"),
        ):
            svc._find_llama_server()

    def test_path_lookup(self, svc: EmbeddingService) -> None:
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("shutil.which", return_value="/usr/bin/llama-server"),
        ):
            os.environ.pop("LLAMA_SERVER_PATH", None)
            assert svc._find_llama_server() == "/usr/bin/llama-server"

    def test_not_found(self, svc: EmbeddingService) -> None:
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("shutil.which", return_value=None),
        ):
            os.environ.pop("LLAMA_SERVER_PATH", None)
            with pytest.raises(FileNotFoundError, match="llama-server not found"):
                svc._find_llama_server()


class TestBuildCommand:
    def test_minimal(self) -> None:
        svc = EmbeddingService(model="/m.gguf", port=9000)
        cmd = svc._build_command("/bin/llama-server")
        assert cmd == [
            "/bin/llama-server",
            "--model",
            "/m.gguf",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--embeddings",
        ]

    def test_all_options(self, tmp_path: Path) -> None:
        svc = EmbeddingService(
            model="/m.gguf",
            host="0.0.0.0",
            port=9001,
            log_dir=tmp_path,
            context_size=2048,
            threads=4,
            gpu_layers=33,
        )
        cmd = svc._build_command("/bin/ls")
        assert "--ctx-size" in cmd
        assert "2048" in cmd
        assert "--threads" in cmd
        assert "4" in cmd
        assert "--gpu-layers" in cmd
        assert "33" in cmd
        assert "--log-file" in cmd
        assert str(tmp_path / "llama-server.log") in cmd

    def test_no_log_dir_omits_log_file(self) -> None:
        svc = EmbeddingService(model="/m.gguf")
        cmd = svc._build_command("/bin/ls")
        assert "--log-file" not in cmd

    def test_pooling_flag(self) -> None:
        svc = EmbeddingService(model="/m.gguf", pooling="last")
        cmd = svc._build_command("/bin/ls")
        assert "--pooling" in cmd
        assert cmd[cmd.index("--pooling") + 1] == "last"

    def test_no_pooling_omits_flag(self) -> None:
        svc = EmbeddingService(model="/m.gguf")
        assert "--pooling" not in svc._build_command("/bin/ls")

    def test_extra_args_appended_and_split(self) -> None:
        # Raw entries are shlex-split and appended verbatim, in order, last.
        svc = EmbeddingService(model="/m.gguf", extra_args=["--flash-attn", "--ubatch-size 256"])
        cmd = svc._build_command("/bin/ls")
        assert cmd[-3:] == ["--flash-attn", "--ubatch-size", "256"]

    def test_extra_args_reject_reserved_flags(self) -> None:
        # Shrike-owned flags are stripped (with their value), the rest survive.
        svc = EmbeddingService(
            model="/m.gguf",
            extra_args=["--host 0.0.0.0", "--port 1", "--model /evil", "--embeddings", "--keep"],
        )
        cmd = svc._build_command("/bin/ls")
        # --host's contract value is still loopback; the override didn't leak in.
        assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
        assert "0.0.0.0" not in cmd
        assert "/evil" not in cmd
        assert cmd.count("--embeddings") == 1  # only Shrike's own
        assert cmd[-1] == "--keep"  # the one non-reserved passthrough token

    def test_extra_args_reject_equals_form(self) -> None:
        svc = EmbeddingService(model="/m.gguf", extra_args=["--host=0.0.0.0", "--ok"])
        cmd = svc._build_command("/bin/ls")
        assert "--host=0.0.0.0" not in cmd
        assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
        assert cmd[-1] == "--ok"

    def test_no_extra_args_appends_nothing(self) -> None:
        svc = EmbeddingService(model="/m.gguf", port=9000)
        # Identical to the minimal command — passthrough is empty.
        assert svc._build_command("/bin/llama-server")[-1] == "--embeddings"


class TestStart:
    def test_start_spawns_and_waits(self, svc: EmbeddingService) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345

        with (
            patch.object(svc, "_find_llama_server", return_value="/bin/llama-server"),
            patch("shrike.embedding.subprocess.Popen", return_value=mock_proc),
            patch.object(svc, "_wait_healthy", return_value=True),
        ):
            svc.start()

        assert svc._process is mock_proc

    def test_start_creates_log_dir(self, svc: EmbeddingService) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345

        with (
            patch.object(svc, "_find_llama_server", return_value="/bin/llama-server"),
            patch("shrike.embedding.subprocess.Popen", return_value=mock_proc),
            patch.object(svc, "_wait_healthy", return_value=True),
        ):
            svc.start()

        assert svc._log_dir is not None
        assert svc._log_dir.exists()

    def test_start_raises_on_health_timeout(self, svc: EmbeddingService) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.pid = 12345
        mock_proc.wait.return_value = 1

        with (
            patch.object(svc, "_find_llama_server", return_value="/bin/llama-server"),
            patch("shrike.embedding.subprocess.Popen", return_value=mock_proc),
            patch.object(svc, "_wait_healthy", return_value=False),
            pytest.raises(RuntimeError, match="failed to become healthy"),
        ):
            svc.start()

        assert svc._process is None

    def test_start_raises_on_missing_binary(self, svc: EmbeddingService) -> None:
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("shutil.which", return_value=None),
        ):
            os.environ.pop("LLAMA_SERVER_PATH", None)
            with pytest.raises(FileNotFoundError):
                svc.start()

    def test_start_noop_if_already_running(self, svc: EmbeddingService) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 999
        svc._process = mock_proc

        with patch.object(svc, "_find_llama_server") as find:
            svc.start()
            find.assert_not_called()


class TestWaitHealthy:
    def test_returns_true_on_200(self, svc: EmbeddingService) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        svc._process = mock_proc

        mock_resp = Mock()
        mock_resp.status_code = 200

        with patch("shrike.embedding.httpx.get", return_value=mock_resp):
            assert svc._wait_healthy() is True

    def test_returns_false_on_process_exit(self, svc: EmbeddingService) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        svc._process = mock_proc

        with patch("shrike.embedding.HEALTH_TIMEOUT", 0.1):
            assert svc._wait_healthy() is False

    def test_retries_on_connect_error(self, svc: EmbeddingService) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        svc._process = mock_proc

        mock_resp = Mock()
        mock_resp.status_code = 200

        call_count = 0

        def side_effect(*args: Any, **kwargs: Any) -> Mock:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("refused")
            return mock_resp

        with (
            patch("shrike.embedding.httpx.get", side_effect=side_effect),
            patch("shrike.embedding.HEALTH_POLL_INTERVAL", 0.01),
        ):
            assert svc._wait_healthy() is True
            assert call_count == 3


class TestStop:
    def test_stop_noop_if_not_started(self, svc: EmbeddingService) -> None:
        svc.stop()
        assert svc._process is None

    def test_stop_already_exited(self, svc: EmbeddingService) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.pid = 123
        svc._process = mock_proc

        svc.stop()
        assert svc._process is None

    def test_stop_sends_sigterm(self, svc: EmbeddingService) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, None]
        mock_proc.pid = 123
        mock_proc.wait.return_value = 0
        svc._process = mock_proc

        with patch("os.kill") as mock_kill:
            svc.stop()
            mock_kill.assert_called_once_with(123, signal.SIGTERM)

        assert svc._process is None

    def test_stop_escalates_to_sigkill(self, svc: EmbeddingService) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, None]
        mock_proc.pid = 123
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), 0]
        svc._process = mock_proc

        kills: list[int] = []

        def track_kill(pid: int, sig: int) -> None:
            kills.append(sig)

        with patch("os.kill", side_effect=track_kill):
            svc.stop()

        assert signal.SIGTERM in kills
        assert signal.SIGKILL in kills
        assert svc._process is None


class TestHealth:
    def test_not_running(self, svc: EmbeddingService) -> None:
        result = svc.health()
        assert result == {"available": False}

    def test_running_and_healthy(self, svc: EmbeddingService) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 456
        svc._process = mock_proc

        mock_resp = Mock()
        mock_resp.status_code = 200

        with patch("shrike.embedding.httpx.get", return_value=mock_resp):
            result = svc.health()

        assert result["available"] is True
        assert result["pid"] == 456
        assert result["model"] == "/fake/model.gguf"

    def test_running_but_unhealthy(self, svc: EmbeddingService) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 456
        svc._process = mock_proc

        with patch("shrike.embedding.httpx.get", side_effect=httpx.ConnectError("refused")):
            result = svc.health()

        assert result["available"] is False
        assert result["pid"] == 456


class TestEmbed:
    def test_raises_when_not_running(self, svc: EmbeddingService) -> None:
        with pytest.raises(RuntimeError, match="not running"):
            svc.embed(["hello"])

    def test_returns_vectors(self, svc: EmbeddingService) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        svc._process = mock_proc

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }
        mock_resp.raise_for_status = Mock()

        with patch("shrike.embedding.httpx.post", return_value=mock_resp):
            result = svc.embed(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    def test_propagates_http_errors(self, svc: EmbeddingService) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        svc._process = mock_proc

        mock_resp = Mock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=Mock(), response=Mock()
        )

        with (
            patch("shrike.embedding.httpx.post", return_value=mock_resp),
            pytest.raises(httpx.HTTPStatusError),
        ):
            svc.embed(["hello"])


class TestModelInfo:
    def test_not_running_returns_empty(self, svc: EmbeddingService) -> None:
        assert svc.model_info() == {}

    def test_parses_v1_models(self, svc: EmbeddingService) -> None:
        _set_running(svc)
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_resp.json.return_value = {
            "data": [{"id": "m.gguf", "meta": {"n_embd": 384, "size": 100}}]
        }
        with patch("shrike.embedding.httpx.get", return_value=mock_resp):
            info = svc.model_info()
        assert info["id"] == "m.gguf"
        assert info["meta"]["n_embd"] == 384

    def test_empty_data_returns_empty(self, svc: EmbeddingService) -> None:
        _set_running(svc)
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_resp.json.return_value = {"data": []}
        with patch("shrike.embedding.httpx.get", return_value=mock_resp):
            assert svc.model_info() == {}

    def test_http_error_returns_empty(self, svc: EmbeddingService) -> None:
        _set_running(svc)
        with patch("shrike.embedding.httpx.get", side_effect=httpx.ConnectError("x")):
            assert svc.model_info() == {}


class TestEmbeddingDim:
    _META = {"n_params": 1, "n_embd": 384, "n_vocab": 3, "n_ctx_train": 4, "size": 5}

    def test_from_meta(self, svc: EmbeddingService) -> None:
        with patch.object(svc, "model_info", return_value={"id": "m", "meta": self._META}):
            assert svc.embedding_dim() == 384

    def test_probe_fallback_when_meta_missing(self, svc: EmbeddingService) -> None:
        # No n_embd in meta → probe with a tiny embed and measure the width.
        with (
            patch.object(svc, "model_info", return_value={"id": "m", "meta": {}}),
            patch.object(svc, "embed", return_value=[[0.0] * 16]) as embed,
        ):
            assert svc.embedding_dim() == 16
        embed.assert_called_once()

    def test_none_when_both_routes_fail(self, svc: EmbeddingService) -> None:
        with (
            patch.object(svc, "model_info", return_value={}),
            patch.object(svc, "embed", side_effect=RuntimeError("down")),
        ):
            assert svc.embedding_dim() is None


class TestModelFingerprint:
    _META = {"n_params": 1, "n_embd": 2, "n_vocab": 3, "n_ctx_train": 4, "size": 5}
    # The note-text normalization version is appended to every fingerprint.
    _TP = f":textprep={EMBED_TEXT_VERSION}"

    def test_from_meta(self, svc: EmbeddingService) -> None:
        with patch.object(svc, "model_info", return_value={"id": "m", "meta": self._META}):
            assert svc.model_fingerprint() == "meta:1:2:3:4:5" + self._TP

    def test_name_excluded(self, svc: EmbeddingService) -> None:
        # Same numeric meta, different name → identical fingerprint.
        with patch.object(svc, "model_info", return_value={"id": "A", "meta": self._META}):
            fp_a = svc.model_fingerprint()
        with patch.object(svc, "model_info", return_value={"id": "B", "meta": self._META}):
            fp_b = svc.model_fingerprint()
        assert fp_a == fp_b

    def test_fallback_to_file_size(self, tmp_path: Path) -> None:
        model = tmp_path / "model.gguf"
        model.write_bytes(b"x" * 100)
        svc = EmbeddingService(model=str(model))
        with patch.object(svc, "model_info", return_value={}):
            assert svc.model_fingerprint() == "file:model.gguf:100" + self._TP

    def test_fallback_missing_file(self, svc: EmbeddingService) -> None:
        with patch.object(svc, "model_info", return_value={}):
            assert svc.model_fingerprint() == "file:model.gguf:-1" + self._TP

    def test_pooling_folded_in(self) -> None:
        svc = EmbeddingService(model="/m.gguf", pooling="last")
        with patch.object(svc, "model_info", return_value={"id": "m", "meta": self._META}):
            assert svc.model_fingerprint() == "meta:1:2:3:4:5:pool=last" + self._TP

    def test_pooling_changes_fingerprint(self) -> None:
        # Different pooling on the same model → different identity → rebuild.
        mean = EmbeddingService(model="/m.gguf", pooling="mean")
        last = EmbeddingService(model="/m.gguf", pooling="last")
        with (
            patch.object(mean, "model_info", return_value={"id": "m", "meta": self._META}),
            patch.object(last, "model_info", return_value={"id": "m", "meta": self._META}),
        ):
            assert mean.model_fingerprint() != last.model_fingerprint()

    def test_unset_pooling_adds_no_pool_token(self, svc: EmbeddingService) -> None:
        # No pooling set → no pool= token (only the always-present textprep tail).
        with patch.object(svc, "model_info", return_value={"id": "m", "meta": self._META}):
            assert svc.model_fingerprint() == "meta:1:2:3:4:5" + self._TP

    def test_extra_args_folded_in(self) -> None:
        svc = EmbeddingService(model="/m.gguf", extra_args=["--flash-attn"])
        with patch.object(svc, "model_info", return_value={"id": "m", "meta": self._META}):
            assert svc.model_fingerprint() == "meta:1:2:3:4:5:args=--flash-attn" + self._TP

    def test_extra_args_change_fingerprint(self) -> None:
        a = EmbeddingService(model="/m.gguf", extra_args=["--flash-attn"])
        b = EmbeddingService(model="/m.gguf", extra_args=["--ubatch-size 256"])
        with (
            patch.object(a, "model_info", return_value={"id": "m", "meta": self._META}),
            patch.object(b, "model_info", return_value={"id": "m", "meta": self._META}),
        ):
            assert a.model_fingerprint() != b.model_fingerprint()

    def test_reserved_extra_args_excluded_from_fingerprint(self) -> None:
        # A stripped reserved flag never reaches llama-server, so it must not
        # appear in the fingerprint (and thus can't force a needless rebuild).
        svc = EmbeddingService(model="/m.gguf", extra_args=["--host 0.0.0.0"])
        with patch.object(svc, "model_info", return_value={"id": "m", "meta": self._META}):
            assert svc.model_fingerprint() == "meta:1:2:3:4:5" + self._TP

    def test_pooling_and_extra_args_both_folded(self) -> None:
        svc = EmbeddingService(model="/m.gguf", pooling="last", extra_args=["--flash-attn"])
        with patch.object(svc, "model_info", return_value={"id": "m", "meta": self._META}):
            assert (
                svc.model_fingerprint() == "meta:1:2:3:4:5:pool=last:args=--flash-attn" + self._TP
            )


class TestEmbedModelPinning:
    def _fake_post(self, captured: dict[str, Any]):
        def post(url: str, json: dict[str, Any], timeout: float) -> Mock:
            captured["json"] = json
            r = Mock()
            r.raise_for_status = Mock()
            r.json.return_value = {"data": [{"embedding": [0.1]}]}
            return r

        return post

    def test_pins_model_name(self, svc: EmbeddingService) -> None:
        _set_running(svc)
        svc._model_name = "m.gguf"
        captured: dict[str, Any] = {}
        with patch("shrike.embedding.httpx.post", side_effect=self._fake_post(captured)):
            svc.embed(["hi"])
        assert captured["json"]["model"] == "m.gguf"
        assert captured["json"]["input"] == ["hi"]

    def test_no_pin_when_name_unknown(self, svc: EmbeddingService) -> None:
        _set_running(svc)
        svc._model_name = None
        captured: dict[str, Any] = {}
        with patch("shrike.embedding.httpx.post", side_effect=self._fake_post(captured)):
            svc.embed(["hi"])
        assert "model" not in captured["json"]


class TestEmbeddingRuntime:
    def test_start_constructs_and_attaches(self) -> None:
        index = MagicMock()
        runtime = EmbeddingRuntime(index=index, model="/m.gguf")
        fake_svc = MagicMock()
        fake_svc.running = True
        with patch("shrike.embedding.EmbeddingService", return_value=fake_svc) as ctor:
            runtime.start()
        ctor.assert_called_once()
        fake_svc.start.assert_called_once()
        index.set_embedding_service.assert_called_once_with(fake_svc)
        assert runtime.service is fake_svc
        assert runtime.running is True

    def test_start_no_model_raises(self) -> None:
        runtime = EmbeddingRuntime(index=MagicMock(), model=None)
        with pytest.raises(ValueError, match="No embedding model"):
            runtime.start()

    def test_start_applies_override(self) -> None:
        runtime = EmbeddingRuntime(index=MagicMock(), model=None)
        fake_svc = MagicMock()
        fake_svc.running = True
        with patch("shrike.embedding.EmbeddingService", return_value=fake_svc):
            runtime.start(model="/override.gguf")
        assert runtime.model == "/override.gguf"

    def test_start_passes_extra_args_to_service(self) -> None:
        runtime = EmbeddingRuntime(index=MagicMock(), model="/m.gguf", extra_args=["--flash-attn"])
        fake_svc = MagicMock()
        fake_svc.running = True
        with patch("shrike.embedding.EmbeddingService", return_value=fake_svc) as ctor:
            runtime.start()
        assert ctor.call_args.kwargs["extra_args"] == ["--flash-attn"]

    def test_start_noop_if_running(self) -> None:
        runtime = EmbeddingRuntime(index=MagicMock(), model="/m.gguf")
        existing = MagicMock()
        existing.running = True
        runtime._service = existing
        with patch("shrike.embedding.EmbeddingService") as ctor:
            svc = runtime.start()
        ctor.assert_not_called()
        assert svc is existing

    def test_stop_detaches_and_stops(self) -> None:
        index = MagicMock()
        runtime = EmbeddingRuntime(index=index, model="/m.gguf")
        fake_svc = MagicMock()
        fake_svc.running = True
        runtime._service = fake_svc
        assert runtime.stop() is True
        index.set_embedding_service.assert_called_once_with(None)
        fake_svc.stop.assert_called_once()
        assert runtime.service is None

    def test_stop_noop_if_not_running(self) -> None:
        runtime = EmbeddingRuntime(index=MagicMock(), model="/m.gguf")
        assert runtime.stop() is False

    def test_health_no_service(self) -> None:
        runtime = EmbeddingRuntime(index=MagicMock())
        health = runtime.health()
        assert health["available"] is False
        assert health["state"] == "not_configured"

    def test_state_transitions(self) -> None:
        # No model → not_configured.
        assert EmbeddingRuntime(index=MagicMock()).state == "not_configured"
        # Model present but not started → stopped.
        assert EmbeddingRuntime(index=MagicMock(), model="/m.gguf").state == "stopped"

    def test_state_failed_after_start_error(self) -> None:
        runtime = EmbeddingRuntime(index=MagicMock(), model="/m.gguf")
        fake_svc = MagicMock()
        fake_svc.start.side_effect = RuntimeError("boom")
        with (
            patch("shrike.embedding.EmbeddingService", return_value=fake_svc),
            pytest.raises(RuntimeError),
        ):
            runtime.start()
        assert runtime.state == "failed"


class TestProcessHelpers:
    """Low-level helpers behind orphan reaping."""

    def test_pid_alive_for_current_process(self) -> None:
        from shrike.embedding import _pid_alive

        assert _pid_alive(os.getpid()) is True

    def test_pid_alive_false_for_nonexistent(self) -> None:
        from shrike.embedding import _pid_alive

        # An implausibly high PID that won't exist.
        assert _pid_alive(2_147_483_646) is False

    def test_pid_alive_false_for_nonpositive(self) -> None:
        from shrike.embedding import _pid_alive

        assert _pid_alive(0) is False
        assert _pid_alive(-1) is False

    def test_port_in_use_detects_listener(self) -> None:
        import socket

        from shrike.embedding import _port_in_use

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        try:
            assert _port_in_use("127.0.0.1", port) is True
        finally:
            sock.close()
        assert _port_in_use("127.0.0.1", port) is False


class TestPidFileLifecycle:
    def test_pid_file_written_on_start(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "embedding.pid"
        svc = EmbeddingService(model="/m.gguf", port=19998, pid_file=pid_file)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 54321

        with (
            patch.object(svc, "_find_llama_server", return_value="/bin/llama-server"),
            patch("shrike.embedding.subprocess.Popen", return_value=mock_proc),
            patch.object(svc, "_wait_healthy", return_value=True),
            patch.object(svc, "_reap_orphan"),
        ):
            svc.start()

        assert pid_file.read_text() == "54321"

    def test_pid_file_removed_on_stop(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "embedding.pid"
        pid_file.write_text("54321")
        svc = EmbeddingService(model="/m.gguf", port=19998, pid_file=pid_file)
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 54321
        svc._process = proc

        with patch("shrike.embedding.os.kill"):
            svc.stop()

        assert not pid_file.exists()

    def test_pid_file_removed_when_already_exited(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "embedding.pid"
        pid_file.write_text("54321")
        svc = EmbeddingService(model="/m.gguf", port=19998, pid_file=pid_file)
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 0
        proc.pid = 54321
        svc._process = proc

        svc.stop()
        assert not pid_file.exists()

    def test_no_pid_file_is_noop(self, tmp_path: Path) -> None:
        # A service without a pid_file must never write/read/raise.
        svc = EmbeddingService(model="/m.gguf", port=19998)
        svc._reap_orphan()
        svc._write_pid_file()
        svc._clear_pid_file()

    def test_start_reaps_before_spawning(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "embedding.pid"
        pid_file.write_text("99999")
        svc = EmbeddingService(model="/m.gguf", port=19998, pid_file=pid_file)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 111

        with (
            patch.object(svc, "_find_llama_server", return_value="/bin/llama-server"),
            patch("shrike.embedding.subprocess.Popen", return_value=mock_proc),
            patch.object(svc, "_wait_healthy", return_value=True),
            patch.object(svc, "_reap_orphan") as reap,
        ):
            svc.start()

        reap.assert_called_once()
        assert pid_file.read_text() == "111"


class TestReapOrphan:
    def test_reaps_when_alive_and_port_held(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "embedding.pid"
        pid_file.write_text("99999")
        svc = EmbeddingService(model="/m.gguf", host="127.0.0.1", port=19998, pid_file=pid_file)

        killed: list[tuple[int, int]] = []
        # Port held during the reap check, then free after the kill.
        port_states = iter([True, False])

        def record_kill(pid: int, sig: int) -> None:
            killed.append((pid, sig))

        with (
            patch("shrike.embedding._pid_alive", return_value=True),
            patch("shrike.embedding._port_in_use", side_effect=lambda h, p: next(port_states)),
            patch("shrike.embedding.os.kill", side_effect=record_kill),
        ):
            svc._reap_orphan()

        assert (99999, signal.SIGTERM) in killed
        assert not pid_file.exists()

    def test_escalates_to_sigkill_if_port_stays_held(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "embedding.pid"
        pid_file.write_text("99999")
        svc = EmbeddingService(model="/m.gguf", host="127.0.0.1", port=19998, pid_file=pid_file)

        killed: list[int] = []

        with (
            patch("shrike.embedding._pid_alive", return_value=True),
            patch("shrike.embedding._port_in_use", return_value=True),  # never frees
            # Zero both port-free deadlines: with time.sleep a no-op, any positive
            # timeout busy-spins on time.monotonic() for that wall-clock duration.
            patch("shrike.embedding.SHUTDOWN_TIMEOUT", 0.0),
            patch("shrike.embedding.SIGKILL_PORT_TIMEOUT", 0.0),
            patch("shrike.embedding.time.sleep", lambda *_: None),
            patch("shrike.embedding.os.kill", side_effect=lambda pid, sig: killed.append(sig)),
        ):
            svc._reap_orphan()

        assert signal.SIGTERM in killed
        assert signal.SIGKILL in killed

    def test_no_reap_when_port_free(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "embedding.pid"
        pid_file.write_text("99999")
        svc = EmbeddingService(model="/m.gguf", port=19998, pid_file=pid_file)

        with (
            patch("shrike.embedding._pid_alive", return_value=True),
            patch("shrike.embedding._port_in_use", return_value=False),
            patch("shrike.embedding.os.kill") as kill,
        ):
            svc._reap_orphan()

        kill.assert_not_called()
        # A stale file is still cleaned even when there's nothing to reap.
        assert not pid_file.exists()

    def test_no_reap_when_pid_dead(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "embedding.pid"
        pid_file.write_text("99999")
        svc = EmbeddingService(model="/m.gguf", port=19998, pid_file=pid_file)

        with (
            patch("shrike.embedding._pid_alive", return_value=False),
            patch("shrike.embedding._port_in_use", return_value=True),
            patch("shrike.embedding.os.kill") as kill,
        ):
            svc._reap_orphan()

        kill.assert_not_called()
        assert not pid_file.exists()

    def test_garbage_pid_file_is_cleaned(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "embedding.pid"
        pid_file.write_text("not-a-pid")
        svc = EmbeddingService(model="/m.gguf", port=19998, pid_file=pid_file)

        with patch("shrike.embedding.os.kill") as kill:
            svc._reap_orphan()  # must not raise

        kill.assert_not_called()
        assert not pid_file.exists()

    def test_missing_pid_file_is_noop(self, tmp_path: Path) -> None:
        svc = EmbeddingService(model="/m.gguf", port=19998, pid_file=tmp_path / "absent.pid")
        with patch("shrike.embedding.os.kill") as kill:
            svc._reap_orphan()
        kill.assert_not_called()
