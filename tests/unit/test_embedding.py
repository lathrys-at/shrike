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

from shrike.embedding import EmbeddingService


@pytest.fixture()
def svc(tmp_path: Path) -> EmbeddingService:
    return EmbeddingService(
        model="/fake/model.gguf",
        port=19999,
        log_dir=tmp_path / "logs",
    )


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
