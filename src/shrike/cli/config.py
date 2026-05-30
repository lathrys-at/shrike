from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from shrike.paths import config_file
from shrike.paths import log_dir as _default_log_dir

if TYPE_CHECKING:
    from shrike.client import ServerSpec

DEFAULT_CONFIG_PATH = config_file()

DEFAULT_CONFIG: dict[str, Any] = {
    "collection": None,
    "server": {
        "host": "127.0.0.1",
        "port": 8372,
    },
    "embedding": {
        "model": None,
        "port": 8373,
        "context_size": None,
        "threads": None,
        "gpu_layers": None,
    },
    "logging": {
        "dir": str(_default_log_dir()),
        "level": "info",
        "levels": {},
        "max_bytes": 10485760,
        "backup_count": 5,
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

    # Persist embedding settings so `shrike embedding start` can find the model
    # after a server start that configured one (also avoids dropping the model
    # on first-run auto-save).
    emb = config.get("embedding", {})
    if emb.get("model"):
        emb_out: dict[str, Any] = {"model": emb["model"]}
        if emb.get("port") and emb["port"] != 8373:
            emb_out["port"] = emb["port"]
        for key in ("context_size", "threads", "gpu_layers", "llama_server"):
            if emb.get(key):
                emb_out[key] = emb[key]
        output["embedding"] = emb_out

    with open(filepath, "w") as f:
        f.write("# Shrike configuration\n")
        f.write("# See: https://github.com/lathrys-at/shrike\n\n")
        if output:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    return filepath


def resolve_embedding(
    config: dict[str, Any],
    *,
    model: str | None = None,
    port: int | None = None,
    context_size: int | None = None,
    threads: int | None = None,
    gpu_layers: int | None = None,
    llama_server: str | None = None,
) -> dict[str, Any]:
    """Resolve embedding parameters via the config → env → flag cascade.

    Flag overrides win, then environment variables, then config values. This is
    the same precedence ``shrike server start`` uses for the collection path.
    Environment variables: ``SHRIKE_EMBEDDING_MODEL``, ``SHRIKE_EMBEDDING_PORT``,
    and ``LLAMA_SERVER_PATH`` (binary). Paths are user-expanded.
    """
    emb = config.get("embedding", {})

    env_port = os.environ.get("SHRIKE_EMBEDDING_PORT")
    resolved: dict[str, Any] = {
        "model": model or os.environ.get("SHRIKE_EMBEDDING_MODEL") or emb.get("model"),
        "llama_server": (
            llama_server or os.environ.get("LLAMA_SERVER_PATH") or emb.get("llama_server")
        ),
        "port": port or (int(env_port) if env_port else None) or emb.get("port"),
        "context_size": context_size or emb.get("context_size"),
        "threads": threads or emb.get("threads"),
        "gpu_layers": gpu_layers or emb.get("gpu_layers"),
    }

    if resolved["model"]:
        resolved["model"] = os.path.expanduser(str(resolved["model"]))
    if resolved["llama_server"]:
        resolved["llama_server"] = os.path.expanduser(str(resolved["llama_server"]))

    return resolved


def embedding_args(resolved: dict[str, Any], *, no_embedding: bool = False) -> list[str]:
    """Build server CLI args from already-resolved embedding params.

    ``resolved`` comes from :func:`resolve_embedding` (config → env → flags).
    """
    args: list[str] = []
    if resolved.get("llama_server"):
        args.extend(["--llama-server", str(resolved["llama_server"])])
    if resolved.get("model"):
        args.extend(["--embedding-model", str(resolved["model"])])
    if resolved.get("port"):
        args.extend(["--embedding-port", str(resolved["port"])])
    if resolved.get("context_size"):
        args.extend(["--embedding-context-size", str(resolved["context_size"])])
    if resolved.get("threads"):
        args.extend(["--embedding-threads", str(resolved["threads"])])
    if resolved.get("gpu_layers"):
        args.extend(["--embedding-gpu-layers", str(resolved["gpu_layers"])])
    if no_embedding:
        args.append("--no-embedding")
    return args


def build_server_spec(
    config: dict[str, Any],
    *,
    collection: str | None = None,
    host: str | None = None,
    port: int | None = None,
    log_dir: str | None = None,
    log_level: str | None = None,
    no_embedding: bool = False,
    embedding_overrides: dict[str, Any] | None = None,
) -> ServerSpec | None:
    """Resolve a launch spec for the local daemon, or None if no collection.

    Centralizes config → env → flag resolution so callers (CLI commands, the
    auto-start client) hand a fully-formed, config-agnostic spec to the client.
    """
    from shrike.client import ServerSpec

    coll = resolve_collection(config, collection)
    if not coll:
        return None

    server = config.get("server", {})
    log_config = config.get("logging", {})
    resolved_log_dir = str(
        Path(log_dir or log_config.get("dir") or str(_default_log_dir())).expanduser()
    )
    resolved_emb = resolve_embedding(config, **(embedding_overrides or {}))

    return ServerSpec(
        collection=coll,
        host=host or server.get("host", "127.0.0.1"),
        port=port or server.get("port", 8372),
        allow_remote=bool(server.get("allow_remote", False)),
        log_dir=resolved_log_dir,
        log_level=log_level or log_config.get("level", "info"),
        embedding_args=embedding_args(resolved_emb, no_embedding=no_embedding),
    )


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
    logging_defaults = dict(DEFAULT_CONFIG["logging"])  # type: ignore[arg-type]
    logging_defaults["levels"] = dict(logging_defaults.get("levels", {}))
    return {
        "collection": DEFAULT_CONFIG["collection"],
        "server": dict(DEFAULT_CONFIG["server"]),  # type: ignore[arg-type]
        "embedding": dict(DEFAULT_CONFIG["embedding"]),  # type: ignore[arg-type]
        "logging": logging_defaults,
    }


def _merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value)
        else:
            base[key] = value
