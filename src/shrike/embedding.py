"""Embedding service — manages a llama-server subprocess for local embeddings.

The Shrike server owns the llama-server process as a direct child. No separate
lock/PID machinery is needed: the child inherits the parent's process group
and is terminated on shutdown.

The service exposes a simple sync interface:
    svc = EmbeddingService(model="/path/to/model.gguf", log_dir="/path/to/logs")
    svc.start()          # spawns llama-server, waits for health
    vecs = svc.embed(["hello", "world"])  # list[list[float]]
    svc.stop()           # SIGTERM → SIGKILL fallback
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("shrike.embedding")

DEFAULT_PORT = 8373
DEFAULT_HOST = "127.0.0.1"
HEALTH_TIMEOUT = 30.0
HEALTH_POLL_INTERVAL = 0.25
SHUTDOWN_TIMEOUT = 5.0


class EmbeddingService:
    """Manages a llama-server subprocess for computing embeddings."""

    def __init__(
        self,
        *,
        model: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        log_dir: str | Path | None = None,
        context_size: int | None = None,
        threads: int | None = None,
        gpu_layers: int | None = None,
        llama_server: str | None = None,
    ) -> None:
        self._model = model
        self._host = host
        self._port = port
        self._log_dir = Path(log_dir) if log_dir else None
        self._context_size = context_size
        self._threads = threads
        self._gpu_layers = gpu_layers
        self._llama_server_override = llama_server
        self._process: subprocess.Popen[bytes] | None = None
        self._base_url = f"http://{host}:{port}"

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def url(self) -> str:
        return self._base_url

    def _find_llama_server(self) -> str:
        """Locate the llama-server binary.

        Priority: explicit override (--llama-server) > LLAMA_SERVER_PATH env > PATH.
        """
        env_path = self._llama_server_override or os.environ.get("LLAMA_SERVER_PATH")
        if env_path:
            p = Path(env_path)
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
            raise FileNotFoundError(f"LLAMA_SERVER_PATH={env_path} does not point to an executable")

        import shutil

        found = shutil.which("llama-server")
        if found:
            return found

        raise FileNotFoundError(
            "llama-server not found. Install llama.cpp or set LLAMA_SERVER_PATH."
        )

    def _build_command(self, binary: str) -> list[str]:
        cmd = [
            binary,
            "--model",
            self._model,
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--embeddings",
        ]
        if self._context_size is not None:
            cmd.extend(["--ctx-size", str(self._context_size)])
        if self._threads is not None:
            cmd.extend(["--threads", str(self._threads)])
        if self._gpu_layers is not None:
            cmd.extend(["--gpu-layers", str(self._gpu_layers)])
        if self._log_dir:
            cmd.extend(["--log-file", str(self._log_dir / "llama-server.log")])
        return cmd

    def start(self) -> None:
        """Spawn llama-server and wait for it to become healthy."""
        if self.running:
            logger.warning("Embedding service already running (PID %s)", self._process.pid)  # type: ignore[union-attr]
            return

        binary = self._find_llama_server()
        cmd = self._build_command(binary)

        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting llama-server: model=%s, host=%s, port=%s",
            self._model,
            self._host,
            self._port,
        )

        stderr_target: int | Any = subprocess.DEVNULL
        if self._log_dir:
            stderr_target = open(self._log_dir / "llama-server-stderr.log", "a")  # noqa: SIM115

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_target,
        )

        if not self._wait_healthy():
            rc = self._process.poll()
            self.stop()
            raise RuntimeError(
                f"llama-server failed to become healthy within {HEALTH_TIMEOUT}s (exit code: {rc})"
            )

        logger.info("Embedding service ready (PID %s)", self._process.pid)

    def _wait_healthy(self) -> bool:
        """Poll GET /health until it returns 200 or we time out."""
        deadline = time.monotonic() + HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                return False
            try:
                resp = httpx.get(f"{self._base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(HEALTH_POLL_INTERVAL)
        return False

    def stop(self) -> None:
        """Stop the llama-server subprocess."""
        if self._process is None:
            return

        pid = self._process.pid

        if self._process.poll() is not None:
            logger.info(
                "Embedding service already exited (PID %s, code %s)",
                pid,
                self._process.returncode,
            )
            self._process = None
            return

        logger.info("Stopping embedding service (PID %s)", pid)

        if sys.platform == "win32":
            self._process.terminate()
        else:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)

        try:
            self._process.wait(timeout=SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server did not exit after SIGTERM, sending SIGKILL")
            with contextlib.suppress(ProcessLookupError):
                if sys.platform == "win32":
                    self._process.terminate()
                else:
                    os.kill(pid, signal.SIGKILL)
            self._process.wait(timeout=2.0)

        logger.info("Embedding service stopped (PID %s)", pid)
        self._process = None

    def health(self) -> dict[str, Any]:
        """Return health status suitable for inclusion in /status responses."""
        if not self.running:
            return {"available": False}

        try:
            resp = httpx.get(f"{self._base_url}/health", timeout=2.0)
            return {
                "available": resp.status_code == 200,
                "pid": self._process.pid if self._process else None,
                "url": self._base_url,
                "model": self._model,
            }
        except (httpx.ConnectError, httpx.TimeoutException):
            return {"available": False, "pid": self._process.pid if self._process else None}

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a list of texts.

        Returns a list of float vectors, one per input text.
        Raises RuntimeError if the service is not running.
        """
        if not self.running:
            raise RuntimeError("Embedding service is not running")

        resp = httpx.post(
            f"{self._base_url}/v1/embeddings",
            json={"input": texts},
            timeout=60.0,
        )
        resp.raise_for_status()

        data = resp.json()
        results: list[list[float]] = [item["embedding"] for item in data["data"]]
        return results
