"""Platform-canonical directory paths for Shrike.

Uses platformdirs to resolve directories appropriate for the current OS:
  - macOS: ~/Library/Application Support, ~/Library/Logs, ~/Library/Caches
  - Linux: XDG dirs (~/.config, ~/.local/state, ~/.cache)
  - Windows: %APPDATA%, %LOCALAPPDATA%

All functions return Path objects. Directories are NOT created automatically —
callers should mkdir(parents=True, exist_ok=True) as needed.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from dataclasses import dataclass
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


# -- Anki base-dir discovery -------------------------------------------
# These are Shrike's *own* directories above; the functions below locate
# *Anki's* base directory and read its profile registry, so `shrike profile
# list --discover` can surface a machine's Anki profiles without the user
# hunting for paths. They mirror Anki's own conventions (aqt/profiles.py:
# ``_default_base`` + the folder-per-profile ``collectionPath``), deliberately
# rather than reusing platformdirs — Anki's Windows base is Roaming %APPDATA%
# and Anki honors an ``ANKI_BASE`` override, neither of which a generic
# platformdirs call reproduces. Read-only and best-effort throughout: a missing
# base dir or unreadable prefs is an empty result, never an error.


@dataclass(frozen=True)
class AnkiProfile:
    """A profile discovered in Anki's base directory.

    ``name`` is the profile name from ``prefs21.db``; ``collection_path`` is the
    folder-per-profile ``collection.anki2`` Anki would open for it
    (``<base>/<name>/collection.anki2``); ``exists`` records whether that file
    is actually present on disk (a registered profile may never have been
    opened, or its media may live elsewhere).
    """

    name: str
    collection_path: str
    exists: bool


def anki_base_dir() -> Path:
    """Anki's base directory (the folder holding ``prefs21.db`` and the
    per-profile collection folders).

    Mirrors Anki's ``ProfileManager._default_base`` precedence: the
    ``ANKI_BASE`` environment override wins; otherwise the platform default —
    ``~/Library/Application Support/Anki2`` (macOS), Roaming ``%APPDATA%\\Anki2``
    (Windows), ``$XDG_DATA_HOME/Anki2`` or ``~/.local/share/Anki2`` (Linux/other).

    Returns a path that may not exist — callers check (``discover_anki_profiles``
    degrades to an empty list when it is absent).
    """
    override = os.environ.get("ANKI_BASE")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path("~/Library/Application Support/Anki2").expanduser()
    if sys.platform.startswith("win"):
        # Anki uses the Roaming AppData folder (CSIDL_APPDATA), i.e. %APPDATA%,
        # not %LOCALAPPDATA%.
        appdata = os.environ.get("APPDATA")
        root = Path(appdata) if appdata else Path("~").expanduser()
        return root / "Anki2"
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home) if data_home else Path("~/.local/share").expanduser()
    return root / "Anki2"


def anki_prefs_db(base: Path | None = None) -> Path:
    """Path to Anki's profile database (``<base>/prefs21.db``)."""
    return (base or anki_base_dir()) / "prefs21.db"


def discover_anki_profiles(base: Path | None = None) -> list[AnkiProfile]:
    """Read Anki's ``prefs21.db`` and return its profiles, newest-Anki-faithful.

    Each profile maps to its conventional collection path
    (``<base>/<name>/collection.anki2`` — Anki stores no path in the blob, it's
    folder-per-profile), with an ``exists`` flag. The ``_global`` meta row is
    excluded. The pickled ``data`` blob is never unpickled (it carries Qt/sip
    types and the path is convention-derived — opening it buys nothing and
    risks an untrusted-pickle decode), so only ``name`` is read.

    Best-effort and read-only: a missing base dir / prefs file, or an
    unreadable/locked database, yields an empty list rather than raising —
    discovery is a convenience over manual registration, never a hard
    dependency. The connection is opened with SQLite's URI ``mode=ro`` so it
    cannot write or create the file and never blocks a running Anki.
    """
    prefs = anki_prefs_db(base)
    if not prefs.is_file():
        return []

    base_dir = prefs.parent
    try:
        # mode=ro: read-only, fails if the file is absent (never creates it),
        # takes only shared locks. NOT immutable=1 — Anki may be mid-write, and
        # immutable would risk a torn read. A short timeout rides out a
        # transient commit lock; we never write, so we never block Anki.
        uri = f"file:{prefs.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            rows = conn.execute(
                "select name from profiles where name != '_global' order by name"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        # Locked, corrupt, or not the schema we expect → no discovery, no crash.
        return []

    profiles: list[AnkiProfile] = []
    for (name,) in rows:
        if not name:
            continue
        coll = base_dir / str(name) / "collection.anki2"
        profiles.append(
            AnkiProfile(
                name=str(name),
                collection_path=str(coll),
                exists=coll.is_file(),
            )
        )
    return profiles
