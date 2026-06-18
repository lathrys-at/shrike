"""Tests for the shrike.log module."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from shrike.platform.log import configure_logging, get_log_file, style_log_line


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset root logger state between tests."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)


class TestConfigureLogging:
    def test_creates_log_directory(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        result = configure_logging(log_dir_override=log_dir)
        assert result == log_dir
        assert log_dir.exists()

    def test_creates_log_file(self, tmp_path: Path) -> None:
        configure_logging(log_dir_override=tmp_path)
        log_file = tmp_path / "shrike.log"
        # File is created lazily by RotatingFileHandler on first write
        logging.getLogger("shrike").info("test message")
        assert log_file.exists()

    def test_log_format(self, tmp_path: Path) -> None:
        configure_logging(log_dir_override=tmp_path)
        logging.getLogger("shrike.test").info("hello world")
        content = (tmp_path / "shrike.log").read_text()
        # Should contain timestamp, level, logger name, message
        assert "INFO" in content
        assert "shrike.test" in content
        assert "hello world" in content

    def test_custom_process_name(self, tmp_path: Path) -> None:
        configure_logging(log_dir_override=tmp_path, process_name="llama")
        logging.getLogger("llama").info("test")
        assert (tmp_path / "llama.log").exists()

    def test_log_level_override(self, tmp_path: Path) -> None:
        configure_logging(log_dir_override=tmp_path, log_level_override="debug")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_log_level_from_config(self, tmp_path: Path) -> None:
        config = {"logging": {"level": "warning"}}
        configure_logging(config, log_dir_override=tmp_path)
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_cli_override_beats_config(self, tmp_path: Path) -> None:
        config = {"logging": {"level": "warning"}}
        configure_logging(config, log_dir_override=tmp_path, log_level_override="debug")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_per_logger_levels(self, tmp_path: Path) -> None:
        config = {
            "logging": {
                "levels": {
                    "shrike.collection": "debug",
                    "shrike.tools": "error",
                },
            },
        }
        configure_logging(config, log_dir_override=tmp_path)
        assert logging.getLogger("shrike.collection").level == logging.DEBUG
        assert logging.getLogger("shrike.tools").level == logging.ERROR

    def test_rotation_settings_from_config(self, tmp_path: Path) -> None:
        config = {
            "logging": {
                "max_bytes": 1024,
                "backup_count": 2,
            },
        }
        configure_logging(config, log_dir_override=tmp_path)
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].maxBytes == 1024
        assert file_handlers[0].backupCount == 2

    def test_foreground_adds_console_handler(self, tmp_path: Path) -> None:
        configure_logging(log_dir_override=tmp_path, foreground=True)
        root = logging.getLogger()
        # Should have both file and console handlers
        assert len(root.handlers) == 2

    def test_noisy_loggers_suppressed(self, tmp_path: Path) -> None:
        configure_logging(log_dir_override=tmp_path)
        assert logging.getLogger("httpx").level >= logging.WARNING
        assert logging.getLogger("httpcore").level >= logging.WARNING
        assert logging.getLogger("uvicorn.access").level >= logging.WARNING

    def test_clears_existing_handlers(self, tmp_path: Path) -> None:
        # Add a dummy handler
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        initial_count = len(root.handlers)
        assert initial_count > 0

        configure_logging(log_dir_override=tmp_path)
        # Should have exactly 1 handler (file only, no foreground)
        assert len(root.handlers) == 1


class TestGetLogFile:
    def test_default_path(self) -> None:
        path = get_log_file()
        assert path.name == "shrike.log"
        assert "shrike" in str(path.parent)

    def test_custom_process_name(self) -> None:
        path = get_log_file(process_name="llama")
        assert path.name == "llama.log"

    def test_config_dir(self, tmp_path: Path) -> None:
        config = {"logging": {"dir": str(tmp_path)}}
        path = get_log_file(config)
        assert path.parent == tmp_path

    def test_override_beats_config(self, tmp_path: Path) -> None:
        override = tmp_path / "override"
        config = {"logging": {"dir": str(tmp_path / "config")}}
        path = get_log_file(config, log_dir_override=override)
        assert path.parent == override


class TestStyleLogLine:
    def test_blank_line_returns_none(self) -> None:
        assert style_log_line("") is None
        assert style_log_line("   ") is None

    def test_info_line(self) -> None:
        styled = style_log_line("2026-05-24T16:00:06 INFO  shrike  Opening collection: /tmp/test")
        assert styled is not None
        plain = styled.plain
        assert "2026-05-24T16:00:06" in plain
        assert "INFO" in plain
        assert "shrike" in plain
        assert "Opening collection: /tmp/test" in plain

    def test_warning_line(self) -> None:
        styled = style_log_line("2026-05-24T16:00:06 WARNING shrike.tools  Something odd")
        assert styled is not None
        assert "WARNING" in styled.plain

    def test_unrecognized_format_returns_unstyled(self) -> None:
        styled = style_log_line("this is just some random text")
        assert styled is not None
        assert styled.plain == "this is just some random text"

    def test_no_message_part(self) -> None:
        styled = style_log_line("2026-05-24T16:00:06 DEBUG logger_only_no_message")
        assert styled is not None
        assert "logger_only_no_message" in styled.plain
