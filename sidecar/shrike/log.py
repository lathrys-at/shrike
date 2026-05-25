"""Centralized logging configuration for all Shrike processes.

Sets up structured plain-text log files with rotation, and optionally
a colored console handler for foreground mode.

Log format (files):
    2025-05-24T10:30:00.123 INFO  shrike.tools  Registered 6 tools
    2025-05-24T10:30:01.456 DEBUG shrike.collection  Opening /path/to/collection.anki2

Console format (foreground mode):
    Rich-colored output via RichHandler.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Any

DEFAULT_LOG_DIR = Path("~/.local/state/shrike/logs").expanduser()
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5
DEFAULT_LEVEL = "INFO"

FILE_FORMAT = "%(asctime)s %(levelname)-5s %(name)s  %(message)s"
FILE_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(
    config: dict[str, Any] | None = None,
    *,
    foreground: bool = False,
    log_dir_override: Path | str | None = None,
    log_level_override: str | None = None,
    process_name: str = "shrike",
) -> Path:
    """Set up logging for a Shrike process.

    Args:
        config: Parsed config dict (may contain a ``logging`` section).
        foreground: If True, add a colored console handler instead of (in
            addition to) the file handler.
        log_dir_override: CLI ``--log-dir`` override.
        log_level_override: CLI ``--log-level`` override.
        process_name: Base name for the log file (e.g. ``"shrike"``,
            ``"llama"``).  Produces ``<process_name>.log``.

    Returns:
        The resolved log directory path.
    """
    log_config = (config or {}).get("logging", {})

    # Resolve log directory
    log_dir = Path(log_dir_override or log_config.get("dir") or DEFAULT_LOG_DIR).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Resolve root level
    root_level_name = (log_level_override or log_config.get("level") or DEFAULT_LEVEL).upper()
    root_level = getattr(logging, root_level_name, logging.INFO)

    # Rotation settings
    max_bytes = int(log_config.get("max_bytes", DEFAULT_MAX_BYTES))
    backup_count = int(log_config.get("backup_count", DEFAULT_BACKUP_COUNT))

    # --- Configure root logger ---
    root = logging.getLogger()
    root.setLevel(root_level)

    # Remove any existing handlers (e.g. from basicConfig or previous calls)
    root.handlers.clear()

    # File handler with rotation
    log_file = log_dir / f"{process_name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=FILE_DATE_FORMAT))
    file_handler.setLevel(root_level)
    root.addHandler(file_handler)

    # Console handler for foreground mode
    if foreground:
        try:
            from rich.logging import RichHandler

            console_handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
                markup=False,
            )
        except ImportError:
            console_handler = logging.StreamHandler()  # type: ignore[assignment]
            console_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=FILE_DATE_FORMAT))
        console_handler.setLevel(root_level)
        root.addHandler(console_handler)

    # Per-logger level overrides
    per_logger_levels: dict[str, str] = log_config.get("levels", {})
    for logger_name, level_name in per_logger_levels.items():
        level = getattr(logging, level_name.upper(), None)
        if level is not None:
            logging.getLogger(logger_name).setLevel(level)

    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    return log_dir


_LEVEL_STYLES: dict[str, str] = {
    "DEBUG": "dim",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "bold red",
    "CRITICAL": "bold red reverse",
}

_LOUD_LEVELS = {"ERROR", "CRITICAL"}


def parse_log_line(line: str) -> dict[str, str] | None:
    """Parse a plain-text log line into its components.

    Returns a dict with ``timestamp``, ``level``, ``logger``, and
    ``message`` keys, or ``None`` if the line is blank or doesn't
    match the expected format.
    """
    stripped = line.strip()
    if not stripped:
        return None

    # Timestamp is always 19 chars (YYYY-MM-DDTHH:MM:SS), followed by a space.
    if len(stripped) < 21 or stripped[19] != " ":
        return None

    timestamp = stripped[:19]
    after_ts = stripped[20:]

    # Level is the next whitespace-delimited token.
    space_idx = after_ts.find(" ")
    if space_idx < 0:
        return None

    level = after_ts[:space_idx].strip()
    rest = after_ts[space_idx + 1 :].lstrip()

    # Logger name and message are separated by double-space.
    double_space = rest.find("  ")
    if double_space > 0:
        logger_name = rest[:double_space]
        message = rest[double_space + 2 :]
    else:
        logger_name = rest
        message = ""

    return {
        "timestamp": timestamp,
        "level": level.upper(),
        "logger": logger_name,
        "message": message,
    }


def style_log_line(line: str) -> Any:
    """Turn a plain-text log line into a styled ``rich.text.Text``.

    Returns ``None`` for blank lines.  Returns an unstyled ``Text`` for
    lines that don't match the expected format.
    """
    from rich.text import Text

    stripped = line.strip()
    if not stripped:
        return None

    parsed = parse_log_line(stripped)
    if parsed is None:
        return Text(stripped)

    level_style = _LEVEL_STYLES.get(parsed["level"], "")
    msg_style = level_style if parsed["level"] in _LOUD_LEVELS else ""

    styled = Text(no_wrap=True, overflow="ellipsis")
    styled.append(parsed["timestamp"], style="dim")
    styled.append(" ")
    styled.append(f"{parsed['level']:<8s}", style=level_style)
    styled.append(parsed["logger"], style="cyan dim")
    styled.append("  ", style="")
    styled.append(parsed["message"], style=msg_style)
    return styled


def get_log_file(
    config: dict[str, Any] | None = None,
    *,
    log_dir_override: Path | str | None = None,
    process_name: str = "shrike",
) -> Path:
    """Return the path to a process's log file without configuring logging.

    Useful for CLI commands that need to read/tail the log.
    """
    log_config = (config or {}).get("logging", {})
    log_dir = Path(log_dir_override or log_config.get("dir") or DEFAULT_LOG_DIR).expanduser()
    return log_dir / f"{process_name}.log"
