"""Platform-canonical directory paths for Shrike.

Uses platformdirs to resolve directories appropriate for the current OS:
  - macOS: ~/Library/Application Support, ~/Library/Logs, ~/Library/Caches
  - Linux: XDG dirs (~/.config, ~/.local/state, ~/.cache)
  - Windows: %APPDATA%, %LOCALAPPDATA%

All functions return Path objects. Directories are NOT created automatically —
callers should mkdir(parents=True, exist_ok=True) as needed.
"""

from __future__ import annotations

from pathlib import Path

from platformdirs import PlatformDirs

_dirs = PlatformDirs("shrike", appauthor=False)


def config_dir() -> Path:
    """Config file directory (config.yml lives here)."""
    return Path(_dirs.user_config_dir)


def config_file() -> Path:
    """Default path to the config file."""
    return config_dir() / "config.yml"


def state_dir() -> Path:
    """Runtime state directory (PID files, server metadata)."""
    return Path(_dirs.user_state_dir)


def log_dir() -> Path:
    """Log file directory."""
    return Path(_dirs.user_log_dir)


def cache_dir() -> Path:
    """Cache directory (embeddings index, etc.)."""
    return Path(_dirs.user_cache_dir)
