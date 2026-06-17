from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from shrike.platform.paths import config_file
from shrike.platform.paths import log_dir as _default_log_dir

if TYPE_CHECKING:
    from shrike.client import ServerSpec

DEFAULT_CONFIG_PATH = config_file()

DEFAULT_CONFIG: dict[str, Any] = {
    "collection": None,
    # Vector-index cache location and flush tuning. ``None`` means "let the
    # server pick" — the platform cache dir, and the kernel saver defaults
    # (60s idle / 100-change burst). Kept None here so config.py needn't import
    # the heavy index module just to know the numbers.
    "cache_dir": None,
    "server": {
        "host": "127.0.0.1",
        "port": 8372,
        # Transport-security additions. ``allowed_hosts``/``allowed_origins`` are
        # extra trusted Host/Origin values beyond loopback (a reverse-proxy or
        # VPN hostname); ``no_dns_rebinding_protection`` disables the guard for
        # network-is-the-boundary deployments (behind Caddy / on a tailnet).
        "allowed_hosts": [],
        "allowed_origins": [],
        "no_dns_rebinding_protection": False,
    },
    "embedding": {
        # Backend kind: "llama" (llama-server, GGUF/MLX) or "onnx" (in-process
        # onnxruntime, needs the 'onnx' extra). ``None`` means "unset" so it
        # resolves to None and isn't transmitted as an override by `embedding
        # start`; the built-in "llama" default is applied at the consumption sites
        # (server `main()` and `embedding_args`).
        "backend": None,
        "model": None,
        "port": 8373,
        # onnxruntime execution providers (onnx backend only), in priority order.
        # Empty means CPUExecutionProvider.
        "onnx_providers": [],
        # Optional cap on the embedding batch size (any backend). None = batch as
        # large as the startup batch-safety probe proves safe.
        "batch_size": None,
        "context_size": None,
        "threads": None,
        "gpu_layers": None,
        # llama-server pooling type (mean|last|cls|none). None means "use the
        # model's GGUF default" — required as `last` for last-token models
        # (Jina v5, Qwen3-Embedding) whose metadata omits it.
        "pooling": None,
        # Generic llama-server arg passthrough — a list of raw token strings
        # appended verbatim (e.g. ["--flash-attn", "--ubatch-size 256"]). For
        # runtime-only flags; vector-affecting flags belong in typed settings.
        "extra_args": [],
    },
    "index": {
        "save_delay": None,
        "save_threshold": None,
    },
    # The collection/profile registry (#66): a name -> collection-path mapping
    # (any .anki2 path), plus the active-default name. ``entries`` is an ordered
    # list; ``default`` names one of them (or is None). Empty by default; the
    # ``shrike profile`` commands manage it. See shrike.registry.
    "profiles": {
        "entries": [],
        "default": None,
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
    if config.get("cache_dir"):
        output["cache_dir"] = config["cache_dir"]

    server = config.get("server", {})
    server_out: dict[str, Any] = {}
    if server.get("host") and server["host"] != "127.0.0.1":
        server_out["host"] = server["host"]
    if server.get("port") and server["port"] != 8372:
        server_out["port"] = server["port"]
    if server.get("allow_remote"):
        server_out["allow_remote"] = True
    if server.get("allowed_hosts"):
        server_out["allowed_hosts"] = list(server["allowed_hosts"])
    if server.get("allowed_origins"):
        server_out["allowed_origins"] = list(server["allowed_origins"])
    if server.get("no_dns_rebinding_protection"):
        server_out["no_dns_rebinding_protection"] = True
    if server.get("cooperative_lock"):
        server_out["cooperative_lock"] = True
    if server.get("lock_hold_seconds") is not None:
        server_out["lock_hold_seconds"] = server["lock_hold_seconds"]
    if server.get("media_allow_private_fetch"):
        server_out["media_allow_private_fetch"] = True
    if server.get("public_url"):
        server_out["public_url"] = server["public_url"]
    if server.get("media_path_roots"):
        server_out["media_path_roots"] = list(server["media_path_roots"])
    if server_out:
        output["server"] = server_out

    # Persist embedding settings so `shrike embedding start` can find the model
    # after a server start that configured one (also avoids dropping the model
    # on first-run auto-save).
    emb = config.get("embedding", {})
    if emb.get("model"):
        emb_out: dict[str, Any] = {"model": emb["model"]}
        # Only persist a non-default backend, so a llama-only config stays clean.
        if emb.get("backend") and emb["backend"] != "llama":
            emb_out["backend"] = emb["backend"]
        if emb.get("port") and emb["port"] != 8373:
            emb_out["port"] = emb["port"]
        persisted_keys = (
            "context_size",
            "threads",
            "gpu_layers",
            "pooling",
            "extra_args",
            "llama_server",
            "onnx_providers",
            "batch_size",
        )
        for key in persisted_keys:
            if emb.get(key):
                emb_out[key] = emb[key]
        output["embedding"] = emb_out

    idx = config.get("index", {})
    idx_out = {k: idx[k] for k in ("save_delay", "save_threshold") if idx.get(k) is not None}
    if idx_out:
        output["index"] = idx_out

    # The collection/profile registry (#66): persisted only when non-empty, via
    # the registry model so the on-disk shape stays the single round-trip
    # contract (ordered entries + the active-default name; optional per-profile
    # fields emitted only when set).
    from shrike.harness.registry import Registry

    profiles_out = Registry.from_config(config).to_config_section()
    if profiles_out:
        output["profiles"] = profiles_out

    # The v2 capability sections (#498) pass through verbatim — structured,
    # user-managed config; --save-config must never drop or rewrite them.
    for key in ("embedders", "recognizers", "managed"):
        if config.get(key) is not None:
            output[key] = config[key]

    with open(filepath, "w") as f:
        f.write("# Shrike configuration\n")
        f.write("# See: https://github.com/lathrys-at/shrike\n\n")
        if output:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    return filepath


def _resolve_extra_args(flag_args: Sequence[str] | None, emb: dict[str, Any]) -> list[str]:
    """Passthrough token strings via flag → env → config (flag wins).

    The CLI flag is a list of raw entries; the ``SHRIKE_EMBEDDING_ARGS`` env var
    is one shlex-split string. Each entry stays a raw token string here — the
    embedding service shlex-splits and guardrails them at command-build time.
    """
    if flag_args:
        return list(flag_args)
    env = os.environ.get("SHRIKE_EMBEDDING_ARGS")
    if env:
        return shlex.split(env)
    return list(emb.get("extra_args") or [])


def _resolve_onnx_providers(flag_providers: Sequence[str] | None, emb: dict[str, Any]) -> list[str]:
    """onnxruntime providers via flag → env → config (flag wins).

    ``SHRIKE_EMBEDDING_ONNX_PROVIDERS`` is comma-separated. Empty means "let
    onnxruntime default" (CPUExecutionProvider).
    """
    if flag_providers:
        return list(flag_providers)
    env = os.environ.get("SHRIKE_EMBEDDING_ONNX_PROVIDERS")
    if env:
        return [p.strip() for p in env.split(",") if p.strip()]
    return list(emb.get("onnx_providers") or [])


def resolve_embedding(
    config: dict[str, Any],
    *,
    backend: str | None = None,
    model: str | None = None,
    port: int | None = None,
    context_size: int | None = None,
    threads: int | None = None,
    gpu_layers: int | None = None,
    pooling: str | None = None,
    extra_args: Sequence[str] | None = None,
    llama_server: str | None = None,
    onnx_providers: Sequence[str] | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Resolve embedding parameters via the config → env → flag cascade.

    Flag overrides win, then environment variables, then config values. This is
    the same precedence ``shrike server start`` uses for the collection path.
    Environment variables: ``SHRIKE_EMBEDDING_BACKEND``, ``SHRIKE_EMBEDDING_MODEL``,
    ``SHRIKE_EMBEDDING_PORT``, ``SHRIKE_EMBEDDING_POOLING``, ``SHRIKE_EMBEDDING_ARGS``
    (shlex-split passthrough), ``SHRIKE_EMBEDDING_ONNX_PROVIDERS`` (comma-separated),
    ``SHRIKE_EMBEDDING_BATCH_SIZE``, and ``LLAMA_SERVER_PATH`` (binary). Paths are
    user-expanded.
    """
    emb = config.get("embedding", {})

    env_port = os.environ.get("SHRIKE_EMBEDDING_PORT")
    resolved: dict[str, Any] = {
        # Resolves to None when unspecified — deliberately NOT defaulted to "llama"
        # here. `shrike embedding start` transmits only non-None resolved values, so
        # a None backend lets a running server keep the backend it booted with
        # (matching how model/pooling behave). The "llama" default is applied at the
        # consumption sites: `embedding_args` emits the flag only for a non-llama
        # backend, and server `main()` uses `args.embedding_backend or DEFAULT_BACKEND`.
        "backend": (backend or os.environ.get("SHRIKE_EMBEDDING_BACKEND") or emb.get("backend")),
        "model": model or os.environ.get("SHRIKE_EMBEDDING_MODEL") or emb.get("model"),
        "llama_server": (
            llama_server or os.environ.get("LLAMA_SERVER_PATH") or emb.get("llama_server")
        ),
        "port": port or (int(env_port) if env_port else None) or emb.get("port"),
        "context_size": context_size or emb.get("context_size"),
        "threads": threads or emb.get("threads"),
        "gpu_layers": gpu_layers or emb.get("gpu_layers"),
        "pooling": (pooling or os.environ.get("SHRIKE_EMBEDDING_POOLING") or emb.get("pooling")),
        "extra_args": _resolve_extra_args(extra_args, emb),
        "onnx_providers": _resolve_onnx_providers(onnx_providers, emb),
        "batch_size": (
            batch_size
            or (int(env_bs) if (env_bs := os.environ.get("SHRIKE_EMBEDDING_BATCH_SIZE")) else None)
            or emb.get("batch_size")
        ),
    }

    if resolved["model"]:
        resolved["model"] = os.path.expanduser(str(resolved["model"]))
    if resolved["llama_server"]:
        resolved["llama_server"] = os.path.expanduser(str(resolved["llama_server"]))

    # Reject rather than silently drop/serialize a bad cap (CLI flags are already
    # IntRange-bounded; this catches a hand-edited config value, where 0 would otherwise
    # be swallowed by the `or`-cascade and a negative would flow straight through).
    bs = resolved["batch_size"]
    if bs is not None and int(bs) < 1:
        raise ValueError(f"embedding.batch_size must be >= 1 (got {bs})")

    return resolved


def resolve_recognition(
    config: dict[str, Any], *, ocr_backend: str | None = None
) -> dict[str, Any]:
    """Resolve recognition (OCR/ASR) parameters via the flag → env → config
    cascade (#221/#223), mirroring ``resolve_embedding``.

    ``None`` for ``ocr_backend`` means recognition is off — today's behaviour
    byte-for-byte. Env var: ``SHRIKE_OCR_BACKEND``. Config: ``recognition.ocr``.
    """
    rec = config.get("recognition", {})
    backend = ocr_backend or os.environ.get("SHRIKE_OCR_BACKEND") or rec.get("ocr")
    return {"ocr": backend or None}


def resolve_cache_dir(config: dict[str, Any], cache_dir_override: str | None = None) -> str | None:
    """Vector-index cache directory via flag → env → config.

    ``None`` means "use the platform default" (the server resolves it). Env var:
    ``SHRIKE_CACHE_DIR``. Paths are user-expanded.
    """
    if cache_dir_override:
        return os.path.expanduser(cache_dir_override)
    env = os.environ.get("SHRIKE_CACHE_DIR")
    if env:
        return os.path.expanduser(env)
    cfg = config.get("cache_dir")
    if cfg:
        return os.path.expanduser(str(cfg))
    return None


def resolve_cross_space_margin(config: dict[str, Any]) -> float:
    """The cross-space image floor margin via env → config → default (#580).

    The floor is ``mean + margin·std`` of a secondary image space's typical best
    match (#201b); ``margin`` is the precision/recall dial — higher → a stricter
    floor → more precision, less recall. Lives under ``search.cross_space_fusion``
    in config; env var ``SHRIKE_CROSS_SPACE_FLOOR_MARGIN``. Default ``1.0``
    (``ACTIVATION_MARGIN`` — today's behaviour). This is an operational knob (no
    v2 capability section), so it keeps the env→config→default cascade past
    #523, like the cache dir / index-flush tuning. An unparseable value falls
    back to the default rather than failing boot."""
    from shrike.harness.index import ACTIVATION_MARGIN

    env = os.environ.get("SHRIKE_CROSS_SPACE_FLOOR_MARGIN")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    cfg = (config.get("search", {}) or {}).get("cross_space_fusion", {}) or {}
    raw = cfg.get("margin")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    return ACTIVATION_MARGIN


def resolve_index_save(
    config: dict[str, Any],
    *,
    save_delay: float | None = None,
    save_threshold: int | None = None,
) -> dict[str, Any]:
    """Resolve index-flush tuning via flag → env → config.

    Each value is ``None`` when unset, leaving the server to apply its own
    default (idle debounce / burst cap). Env vars: ``SHRIKE_INDEX_SAVE_DELAY``,
    ``SHRIKE_INDEX_SAVE_THRESHOLD``.
    """
    idx = config.get("index", {})
    env_delay = os.environ.get("SHRIKE_INDEX_SAVE_DELAY")
    env_threshold = os.environ.get("SHRIKE_INDEX_SAVE_THRESHOLD")

    delay = save_delay
    if delay is None and env_delay:
        delay = float(env_delay)
    if delay is None:
        delay = idx.get("save_delay")

    threshold = save_threshold
    if threshold is None and env_threshold:
        threshold = int(env_threshold)
    if threshold is None:
        threshold = idx.get("save_threshold")

    return {"save_delay": delay, "save_threshold": threshold}


def resolve_transport(
    config: dict[str, Any],
    *,
    allowed_hosts: list[str] | None = None,
    allowed_origins: list[str] | None = None,
    no_dns_rebinding_protection: bool | None = None,
) -> dict[str, Any]:
    """Resolve transport-security settings via flag → env → config.

    ``allowed_hosts``/``allowed_origins`` are *additive* trusted Host/Origin
    values (beyond loopback); ``no_dns_rebinding_protection`` disables the guard
    entirely (the network-is-the-boundary deployment). Env vars:
    ``SHRIKE_ALLOWED_HOSTS`` / ``SHRIKE_ALLOWED_ORIGINS`` (comma-separated) and
    ``SHRIKE_NO_DNS_REBINDING_PROTECTION`` (truthy: 1/true/yes/on).
    """
    server = config.get("server", {})

    def _env_list(name: str) -> list[str] | None:
        raw = os.environ.get(name)
        if raw is None:
            return None
        return [v.strip() for v in raw.split(",") if v.strip()]

    hosts = allowed_hosts or _env_list("SHRIKE_ALLOWED_HOSTS") or server.get("allowed_hosts") or []
    origins = (
        allowed_origins
        or _env_list("SHRIKE_ALLOWED_ORIGINS")
        or server.get("allowed_origins")
        or []
    )

    if no_dns_rebinding_protection is None:
        env_flag = os.environ.get("SHRIKE_NO_DNS_REBINDING_PROTECTION")
        if env_flag is not None:
            no_dns_rebinding_protection = env_flag.strip().lower() in ("1", "true", "yes", "on")
        else:
            no_dns_rebinding_protection = bool(server.get("no_dns_rebinding_protection", False))

    return {
        "allowed_hosts": list(hosts),
        "allowed_origins": list(origins),
        "no_dns_rebinding_protection": no_dns_rebinding_protection,
    }


def resolve_locking(
    config: dict[str, Any],
    *,
    cooperative: bool | None = None,
    hold_seconds: float | None = None,
) -> dict[str, Any]:
    """Resolve cooperative-locking settings via flag → env → config.

    ``cooperative`` enables open-on-demand/idle-release; ``hold_seconds`` is the
    idle window before releasing. Env vars: ``SHRIKE_COOPERATIVE_LOCK`` (truthy:
    1/true/yes/on) and ``SHRIKE_LOCK_HOLD_SECONDS``. A ``hold_seconds`` of None
    means "use the server's built-in default".
    """
    server = config.get("server", {})

    if cooperative is None:
        env_flag = os.environ.get("SHRIKE_COOPERATIVE_LOCK")
        if env_flag is not None:
            cooperative = env_flag.strip().lower() in ("1", "true", "yes", "on")
        else:
            cooperative = bool(server.get("cooperative_lock", False))

    if hold_seconds is None:
        env_hold = os.environ.get("SHRIKE_LOCK_HOLD_SECONDS")
        hold_seconds = float(env_hold) if env_hold else server.get("lock_hold_seconds")

    return {"cooperative": cooperative, "hold_seconds": hold_seconds}


def locking_args(resolved: dict[str, Any]) -> list[str]:
    """Build server CLI args from resolved locking params (see resolve_locking)."""
    args: list[str] = []
    if resolved.get("cooperative"):
        args.append("--cooperative-lock")
    if resolved.get("hold_seconds") is not None:
        args.extend(["--lock-hold-seconds", str(resolved["hold_seconds"])])
    return args


def index_args(resolved: dict[str, Any]) -> list[str]:
    """Build server CLI args from resolved index-flush params (see resolve_index_save)."""
    args: list[str] = []
    if resolved.get("save_delay") is not None:
        args.extend(["--index-save-delay", str(resolved["save_delay"])])
    if resolved.get("save_threshold") is not None:
        args.extend(["--index-save-threshold", str(resolved["save_threshold"])])
    return args


def transport_args(resolved: dict[str, Any]) -> list[str]:
    """Build server CLI args from resolved transport params (see resolve_transport)."""
    args: list[str] = []
    for host in resolved.get("allowed_hosts") or []:
        args.extend(["--allowed-host", str(host)])
    for origin in resolved.get("allowed_origins") or []:
        args.extend(["--allowed-origin", str(origin)])
    if resolved.get("no_dns_rebinding_protection"):
        args.append("--no-dns-rebinding-protection")
    return args


def embedding_args(resolved: dict[str, Any], *, no_embedding: bool = False) -> list[str]:
    """Build server CLI args from already-resolved embedding params.

    ``resolved`` comes from :func:`resolve_embedding` (config → env → flags).
    """
    args: list[str] = []
    # Only emit a non-default backend so existing llama-only command lines stay
    # byte-for-byte unchanged.
    if resolved.get("backend") and resolved["backend"] != "llama":
        args.extend(["--embedding-backend", str(resolved["backend"])])
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
    if resolved.get("pooling"):
        args.extend(["--embedding-pooling", str(resolved["pooling"])])
    for token in resolved.get("extra_args") or []:
        args.extend(["--embedding-arg", str(token)])
    for provider in resolved.get("onnx_providers") or []:
        args.extend(["--embedding-onnx-provider", str(provider)])
    if resolved.get("batch_size"):
        args.extend(["--embedding-batch-size", str(resolved["batch_size"])])
    if no_embedding:
        args.append("--no-embedding")
    return args


def resolve_embedding_profile(
    config: dict[str, Any],
    embedding_overrides: dict[str, Any] | None,
    *,
    quiet: bool = False,
) -> dict[str, Any]:
    """Resolve the embedding params, v2-first (#498 slice 1).

    A config declaring the v2 sections (``embedders:``/``recognizers:``/
    ``managed:``) is parsed, resolved against the build's compiled features,
    and bridged onto the legacy param shape today's runtime consumes —
    migration warnings print to stderr. The ``--embedding-*`` flags (and
    their env twins) are incompatible with a v2 config: the config file is
    the only home for the structured sections (docs/distribution.md); the
    flags survive one release for legacy configs and then go entirely.
    Without v2 sections, the legacy cascade runs unchanged (a deprecated
    legacy ``embedding:``/``recognition:`` section warns via the migration).

    ``quiet`` suppresses the deprecation/ignored-env warnings — set by the
    passive resolution every client command performs for auto-start
    (``build_server_spec``), so only the explicit ``server start`` /
    ``embedding start`` paths warn, and exactly once.
    """
    from shrike.harness.profiles import (
        ProfileError,
        parse_capabilities,
        plan_to_runtime_params,
        resolve_profile,
    )

    caps = parse_capabilities(config)
    if caps.legacy:
        if not quiet:
            for warning in caps.warnings:
                print(f"warning: {warning}", file=sys.stderr)
        return resolve_embedding(config, **(embedding_overrides or {}))

    overrides = {k: v for k, v in (embedding_overrides or {}).items() if v not in (None, [], ())}
    if overrides:
        raise ProfileError(
            f"--embedding-*/--llama-server flags ({', '.join(sorted(overrides))}) are "
            "incompatible with a config declaring embedders:/managed: — the config file "
            "is the only home for these (docs/distribution.md)"
        )
    # The legacy env twins don't apply under v2 either — warn rather than let
    # an ambient variable silently do nothing (the cross-talk #498 kills).
    legacy_env = [
        name
        for name in (
            "SHRIKE_EMBEDDING_BACKEND",
            "SHRIKE_EMBEDDING_MODEL",
            "SHRIKE_EMBEDDING_PORT",
            "SHRIKE_EMBEDDING_POOLING",
            "SHRIKE_EMBEDDING_ARGS",
            "SHRIKE_EMBEDDING_ONNX_PROVIDERS",
            "SHRIKE_EMBEDDING_BATCH_SIZE",
            "LLAMA_SERVER_PATH",
        )
        if os.environ.get(name)
    ]
    if legacy_env and not quiet:
        print(
            f"warning: {', '.join(legacy_env)} ignored — the config declares "
            "embedders:/managed:, which is the only home for embedding settings "
            "(docs/distribution.md)",
            file=sys.stderr,
        )

    import shrike_native  # lazy: keeps plain client commands import-light

    plan = resolve_profile(caps, shrike_native.build_features())
    if not quiet:
        for warning in plan.warnings:
            print(f"warning: {warning}", file=sys.stderr)
    resolved = plan_to_runtime_params(plan)
    for key in ("model", "llama_server"):
        if resolved.get(key):
            resolved[key] = os.path.expanduser(str(resolved[key]))
    if resolved.get("mmprojs"):
        resolved["mmprojs"] = [os.path.expanduser(str(p)) for p in resolved["mmprojs"]]
    return resolved


def build_server_spec(
    config: dict[str, Any],
    *,
    config_path: Path | str | None = None,
    collection: str | None = None,
    host: str | None = None,
    port: int | None = None,
    log_dir: str | None = None,
    log_level: str | None = None,
    cache_dir: str | None = None,
    no_embedding: bool = False,
    embedding_overrides: dict[str, Any] | None = None,
    index_save_overrides: dict[str, Any] | None = None,
    transport_overrides: dict[str, Any] | None = None,
    locking_overrides: dict[str, Any] | None = None,
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
    # quiet: this resolution runs passively on every client command (the
    # auto-start spec); the explicit start commands own the warnings.
    resolved_emb = resolve_embedding_profile(config, embedding_overrides, quiet=True)
    # A v2 config rides --config (#498): the daemon resolves the structured
    # sections itself (remote endpoints have no flag spelling), so the spec
    # carries the config path and NO embedding flags.
    is_v2 = any(config.get(k) is not None for k in ("embedders", "recognizers", "managed"))
    v2_config_path = str(config_path or DEFAULT_CONFIG_PATH) if is_v2 else None
    resolved_index = resolve_index_save(config, **(index_save_overrides or {}))
    resolved_transport = resolve_transport(config, **(transport_overrides or {}))
    resolved_locking = resolve_locking(config, **(locking_overrides or {}))

    env_private_media = os.environ.get("SHRIKE_MEDIA_ALLOW_PRIVATE_FETCH")
    allow_private_media = (
        env_private_media.strip().lower() in ("1", "true", "yes", "on")
        if env_private_media is not None
        else bool(server.get("media_allow_private_fetch", False))
    )
    public_url = os.environ.get("SHRIKE_PUBLIC_URL") or server.get("public_url") or None
    env_roots = os.environ.get("SHRIKE_MEDIA_PATH_ROOTS")
    media_path_roots = (
        [p for p in env_roots.split(os.pathsep) if p]
        if env_roots is not None
        else list(server.get("media_path_roots") or [])
    )

    return ServerSpec(
        collection=coll,
        host=host or server.get("host", "127.0.0.1"),
        port=port or server.get("port", 8372),
        allow_remote=bool(server.get("allow_remote", False)),
        allow_private_media_fetch=allow_private_media,
        public_url=public_url,
        media_path_roots=media_path_roots,
        allowed_hosts=resolved_transport["allowed_hosts"],
        allowed_origins=resolved_transport["allowed_origins"],
        no_dns_rebinding_protection=resolved_transport["no_dns_rebinding_protection"],
        log_dir=resolved_log_dir,
        log_level=log_level or log_config.get("level", "info"),
        cache_dir=resolve_cache_dir(config, cache_dir),
        embedding_args=(
            (["--no-embedding"] if no_embedding else [])
            if is_v2
            else embedding_args(resolved_emb, no_embedding=no_embedding)
        ),
        config_path=v2_config_path,
        index_args=index_args(resolved_index),
        locking_args=locking_args(resolved_locking),
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
    embedding_defaults = dict(DEFAULT_CONFIG["embedding"])  # type: ignore[arg-type]
    # Fresh list so a caller mutating extra_args can't poison the module default.
    embedding_defaults["extra_args"] = list(embedding_defaults.get("extra_args", []))
    profiles_defaults = dict(DEFAULT_CONFIG["profiles"])  # type: ignore[arg-type]
    # Fresh list so a loaded registry's entries can't poison the module default.
    profiles_defaults["entries"] = list(profiles_defaults.get("entries", []))
    return {
        "collection": DEFAULT_CONFIG["collection"],
        "cache_dir": DEFAULT_CONFIG["cache_dir"],
        "server": dict(DEFAULT_CONFIG["server"]),  # type: ignore[arg-type]
        "embedding": embedding_defaults,
        "index": dict(DEFAULT_CONFIG["index"]),  # type: ignore[arg-type]
        "profiles": profiles_defaults,
        "logging": logging_defaults,
    }


def _merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value)
        else:
            base[key] = value
