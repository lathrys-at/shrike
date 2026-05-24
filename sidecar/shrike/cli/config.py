from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("~/.config/shrike/config.yml").expanduser()

DEFAULT_CONFIG = {
    "collection": None,
    "server": {
        "host": "127.0.0.1",
        "port": 8372,
    },
}


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load config from a YAML file, falling back to defaults.

    Returns a dict with all keys populated (missing keys filled from defaults).
    """
    config = _deep_copy_defaults()
    filepath = path or DEFAULT_CONFIG_PATH

    if filepath.exists():
        with open(filepath) as f:
            file_config = yaml.safe_load(f) or {}
        _merge(config, file_config)

    return config


def save_config(config: dict[str, Any], path: Path | None = None) -> Path:
    """Write config to a YAML file. Creates parent directories if needed."""
    filepath = path or DEFAULT_CONFIG_PATH
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Only write non-default, meaningful values
    output: dict[str, Any] = {}
    if config.get("collection"):
        output["collection"] = config["collection"]

    server = config.get("server", {})
    server_out: dict[str, Any] = {}
    if server.get("host") and server["host"] != "127.0.0.1":
        server_out["host"] = server["host"]
    if server.get("port") and server["port"] != 8372:
        server_out["port"] = server["port"]
    if server_out:
        output["server"] = server_out

    with open(filepath, "w") as f:
        f.write("# Shrike configuration\n")
        f.write("# See: https://github.com/lathrys-at/shrike\n\n")
        if output:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    return filepath


def resolve_url(config: dict[str, Any], url_override: str | None = None) -> str:
    """Determine the server URL from config, env, or explicit override."""
    if url_override:
        return url_override

    env_url = os.environ.get("SHRIKE_URL")
    if env_url:
        return env_url

    server = config.get("server", {})
    host = server.get("host", "127.0.0.1")
    port = server.get("port", 8372)
    return f"http://{host}:{port}/mcp"


def resolve_collection(
    config: dict[str, Any], collection_override: str | None = None
) -> str | None:
    """Determine the collection path from config, env, or explicit override."""
    if collection_override:
        return os.path.expanduser(collection_override)

    env_collection = os.environ.get("SHRIKE_COLLECTION")
    if env_collection:
        return os.path.expanduser(env_collection)

    config_collection = config.get("collection")
    if config_collection:
        return str(os.path.expanduser(config_collection))

    return None


def _deep_copy_defaults() -> dict[str, Any]:
    return {
        "collection": DEFAULT_CONFIG["collection"],
        "server": dict(DEFAULT_CONFIG["server"]),  # type: ignore[arg-type]
    }


def _merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value)
        else:
            base[key] = value
