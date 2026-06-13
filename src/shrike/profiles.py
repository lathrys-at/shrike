"""Config model v2 (#498): capability declarations → a resolved profile plan.

The distribution-profiles design (docs/distribution.md, canonical) replaces
the backend knob with *capability declarations*: an ``embedders:`` list of
vector spaces (each declaring ``modalities`` + ``runtime``), a
``recognizers:`` map keyed by source (``ocr``/``asr``/``describe``), and a
``managed:`` section for manage-class components (llama_server, sync_server).
This module is the pure half: parse + validate the declarations, migrate the
legacy ``embedding:``/``recognition:`` sections (warn-and-map, one release),
and resolve the declared set against what the build actually compiled
(``shrike_native.build_features()``, passed in — this module imports nothing
native so it stays unit-testable everywhere).

The two-layer rule it enforces (#498):

- a ``runtime`` whose build feature is **not compiled in** is a
  :class:`ProfileError` naming the build profile — never a silent no-op
  (killing the silent-cross-talk era is the point);
- a capability the build *can* express but Shrike hasn't implemented yet is a
  :class:`ProfileError` naming the tracking issue (#229 multi-space, #485
  asr/describe integration, #502 remote OCR, #36 sync server) — declared
  config never silently does nothing.

The N=1 serving shapes map onto the runtime via :func:`plan_to_runtime_params`:
the ort backends keyed by modalities, the managed llama-server (``manage:
auto``), and the unmanaged ``remote`` backend — an explicit ``endpoint``
(cloud/tailnet, ``api_key_env``-authenticated) or ``manage: attach`` (an
existing llama-server Shrike never spawns or stops). Structured entries have
no flag spelling: a v2 config reaches the daemon as ``--config`` and the
daemon resolves it here itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MODALITIES = ("text", "image", "audio")
EMBEDDER_RUNTIMES = ("onnx", "remote", "platform")
RECOGNIZER_SOURCES = ("ocr", "asr", "describe")
RECOGNIZER_RUNTIMES = ("onnx", "remote", "platform")
MANAGE_MODES = ("auto", "attach", "off")

# runtime → the #499 build-matrix feature that provides it.
_RUNTIME_FEATURE = {
    "onnx": "engine-ort",
    "remote": "engine-remote",
    "platform": "engine-apple",
}


class ProfileError(ValueError):
    """A config-layer capability error: the declaration is invalid, or it
    names a runtime/capability this build or this release can't serve. Always
    loud and actionable — the message names the offending entry and either
    the build profile or the tracking issue."""


@dataclass(frozen=True)
class EmbedderEntry:
    """One vector space (#229/#235): what it embeds and where it runs."""

    modalities: tuple[str, ...]
    runtime: str
    model: str | None = None
    endpoint: str | None = None  # remote only; None = the managed llama-server
    api_key_env: str | None = None  # remote only; secrets are referenced, never inline
    pooling: str | None = None  # vector-affecting; folds into the entry's fingerprint
    providers: tuple[str, ...] = ()  # onnx execution providers, priority order
    batch_size: int | None = None


@dataclass(frozen=True)
class RecognizerEntry:
    """One recognition source (#485's engine map row)."""

    source: str
    runtime: str
    model: str | None = None
    endpoint: str | None = None
    api_key_env: str | None = None
    locale: str | None = None  # asr


@dataclass(frozen=True)
class ManagedLlama:
    """llama-server as a manage-class component — orthogonal to engines."""

    manage: str = "auto"  # auto = spawn/own a child; attach = existing; off = cloud
    binary: str | None = None
    args: tuple[str, ...] = ()
    port: int | None = None
    context_size: int | None = None
    threads: int | None = None
    gpu_layers: int | None = None


@dataclass(frozen=True)
class ManagedSync:
    """anki's sync server as a child process (#36) — server profile only."""

    manage: str = "off"


@dataclass(frozen=True)
class Capabilities:
    """The parsed declaration set, before build resolution. ``legacy`` marks
    a set synthesized from the old ``embedding:``/``recognition:`` sections
    (warn-and-map); ``warnings`` carries the migration messages to log."""

    embedders: tuple[EmbedderEntry, ...] = ()
    recognizers: tuple[RecognizerEntry, ...] = ()
    managed_llama: ManagedLlama | None = None
    managed_sync: ManagedSync | None = None
    legacy: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedProfile:
    """The declared set intersected with the build: what this process will
    actually serve. Today at most one embedder space (#229 is the multi-space
    substrate); ``warnings`` aggregates migration + degradation messages."""

    embedder: EmbedderEntry | None
    managed_llama: ManagedLlama | None
    warnings: tuple[str, ...] = ()


def _require_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProfileError(f"{where} must be a non-empty string (got {value!r})")
    return value


def _opt_str(raw: Mapping[str, Any], key: str, where: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    return _require_str(value, f"{where}.{key}")


def _opt_int(raw: Mapping[str, Any], key: str, where: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileError(f"{where}.{key} must be an integer (got {value!r})")
    return value


def _reject_unknown(raw: Mapping[str, Any], known: Iterable[str], where: str) -> None:
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ProfileError(
            f"{where} has unknown key(s) {', '.join(unknown)} — "
            f"the config model is docs/distribution.md's (#498)"
        )


def _parse_embedder(raw: Any, index: int) -> EmbedderEntry:
    where = f"embedders[{index}]"
    if not isinstance(raw, Mapping):
        raise ProfileError(f"{where} must be a mapping (got {type(raw).__name__})")
    _reject_unknown(
        raw,
        (
            "modalities",
            "runtime",
            "model",
            "endpoint",
            "api_key_env",
            "pooling",
            "providers",
            "batch_size",
        ),
        where,
    )

    modalities_raw = raw.get("modalities")
    if (
        not isinstance(modalities_raw, Sequence)
        or isinstance(modalities_raw, str)
        or not modalities_raw
    ):
        raise ProfileError(f"{where}.modalities must be a non-empty list from {MODALITIES}")
    modalities = tuple(str(m) for m in modalities_raw)
    bad = sorted(set(modalities) - set(MODALITIES))
    if bad:
        raise ProfileError(
            f"{where}.modalities has unknown modality {', '.join(bad)} "
            f"(choices: {', '.join(MODALITIES)})"
        )
    if len(set(modalities)) != len(modalities):
        raise ProfileError(f"{where}.modalities has duplicates")

    runtime = _require_str(raw.get("runtime"), f"{where}.runtime")
    if runtime not in EMBEDDER_RUNTIMES:
        raise ProfileError(
            f"{where}.runtime must be one of {', '.join(EMBEDDER_RUNTIMES)} (got {runtime!r})"
        )

    entry = EmbedderEntry(
        modalities=modalities,
        runtime=runtime,
        model=_opt_str(raw, "model", where),
        endpoint=_opt_str(raw, "endpoint", where),
        api_key_env=_opt_str(raw, "api_key_env", where),
        pooling=_opt_str(raw, "pooling", where),
        providers=tuple(str(p) for p in (raw.get("providers") or ())),
        batch_size=_opt_int(raw, "batch_size", where),
    )

    # Knobs are structurally scoped to their runtime — inapplicable knobs are
    # errors, not silent cross-talk (the disease #498 cures).
    if entry.runtime != "remote":
        for key in ("endpoint", "api_key_env"):
            if getattr(entry, key) is not None:
                raise ProfileError(f"{where}.{key} applies only to runtime: remote")
    if entry.runtime != "onnx" and entry.providers:
        raise ProfileError(f"{where}.providers applies only to runtime: onnx")
    if entry.runtime == "onnx" and entry.model is None:
        raise ProfileError(
            f"{where}.model is required for runtime: onnx (the model dir/file to load)"
        )
    if entry.batch_size is not None and entry.batch_size < 1:
        raise ProfileError(f"{where}.batch_size must be >= 1 (got {entry.batch_size})")
    return entry


def _parse_recognizer(source: str, raw: Any) -> RecognizerEntry:
    where = f"recognizers.{source}"
    if source not in RECOGNIZER_SOURCES:
        raise ProfileError(
            f"recognizers has unknown source {source!r} (choices: {', '.join(RECOGNIZER_SOURCES)})"
        )
    if not isinstance(raw, Mapping):
        raise ProfileError(f"{where} must be a mapping (got {type(raw).__name__})")
    _reject_unknown(raw, ("runtime", "model", "endpoint", "api_key_env", "locale"), where)
    runtime = _require_str(raw.get("runtime"), f"{where}.runtime")
    if runtime not in RECOGNIZER_RUNTIMES:
        raise ProfileError(
            f"{where}.runtime must be one of {', '.join(RECOGNIZER_RUNTIMES)} (got {runtime!r})"
        )
    entry = RecognizerEntry(
        source=source,
        runtime=runtime,
        model=_opt_str(raw, "model", where),
        endpoint=_opt_str(raw, "endpoint", where),
        api_key_env=_opt_str(raw, "api_key_env", where),
        locale=_opt_str(raw, "locale", where),
    )
    if entry.runtime != "remote":
        for key in ("endpoint", "api_key_env"):
            if getattr(entry, key) is not None:
                raise ProfileError(f"{where}.{key} applies only to runtime: remote")
    return entry


def _parse_managed(raw: Any) -> tuple[ManagedLlama | None, ManagedSync | None]:
    if raw is None:
        return None, None
    if not isinstance(raw, Mapping):
        raise ProfileError(f"managed must be a mapping (got {type(raw).__name__})")
    _reject_unknown(raw, ("llama_server", "sync_server"), "managed")

    llama = None
    if (lraw := raw.get("llama_server")) is not None:
        where = "managed.llama_server"
        if not isinstance(lraw, Mapping):
            raise ProfileError(f"{where} must be a mapping")
        _reject_unknown(
            lraw,
            ("manage", "binary", "args", "port", "context_size", "threads", "gpu_layers"),
            where,
        )
        manage = str(lraw.get("manage", "auto"))
        if manage not in MANAGE_MODES:
            raise ProfileError(
                f"{where}.manage must be one of {', '.join(MANAGE_MODES)} (got {manage!r})"
            )
        args_raw = lraw.get("args") or ()
        if isinstance(args_raw, str) or not isinstance(args_raw, Sequence):
            raise ProfileError(f"{where}.args must be a list of token strings")
        llama = ManagedLlama(
            manage=manage,
            binary=_opt_str(lraw, "binary", where),
            args=tuple(str(a) for a in args_raw),
            port=_opt_int(lraw, "port", where),
            context_size=_opt_int(lraw, "context_size", where),
            threads=_opt_int(lraw, "threads", where),
            gpu_layers=_opt_int(lraw, "gpu_layers", where),
        )

    sync = None
    if (sraw := raw.get("sync_server")) is not None:
        where = "managed.sync_server"
        if not isinstance(sraw, Mapping):
            raise ProfileError(f"{where} must be a mapping")
        _reject_unknown(sraw, ("manage",), where)
        manage = str(sraw.get("manage", "off"))
        if manage not in ("auto", "off"):
            raise ProfileError(f"{where}.manage must be auto or off (got {manage!r})")
        sync = ManagedSync(manage=manage)

    return llama, sync


def _migrate_legacy(config: Mapping[str, Any]) -> Capabilities:
    """Synthesize v2 capabilities from the legacy ``embedding:`` /
    ``recognition:`` sections — the one-release warn-and-map (#498). Legacy
    semantics are preserved (degrade, don't refuse): a legacy OCR selection
    the build can't serve becomes a warning + an absent capability, exactly
    the boot behavior the old flag had."""
    warnings: list[str] = []
    emb = config.get("embedding") or {}
    embedders: list[EmbedderEntry] = []
    llama: ManagedLlama | None = None

    model = emb.get("model")
    backend = emb.get("backend") or ("llama" if model else None)
    if model and backend:
        warnings.append(
            "config: the embedding: section is deprecated — declare an embedders: "
            "entry instead (docs/distribution.md; this mapping is removed after one release)"
        )
        providers = tuple(str(p) for p in (emb.get("onnx_providers") or ()))
        batch_size = emb.get("batch_size")
        if batch_size is not None and int(batch_size) < 1:
            # Same rule as _parse_embedder/resolve_embedding — a migrated
            # entry must never hold an illegal value.
            raise ProfileError(f"embedding.batch_size must be >= 1 (got {batch_size})")
        pooling = emb.get("pooling")
        if backend in ("onnx", "onnx-rs"):
            embedders.append(
                EmbedderEntry(
                    modalities=("text",),
                    runtime="onnx",
                    model=str(model),
                    pooling=pooling,
                    providers=providers,
                    batch_size=batch_size,
                )
            )
        elif backend in ("clip", "clip-rs"):
            embedders.append(
                EmbedderEntry(
                    modalities=("text", "image"),
                    runtime="onnx",
                    model=str(model),
                    providers=providers,
                    batch_size=batch_size,
                )
            )
        else:  # llama — the managed-child remote shape
            embedders.append(
                EmbedderEntry(
                    modalities=("text",),
                    runtime="remote",
                    model=str(model),
                    pooling=pooling,
                    batch_size=batch_size,
                )
            )
            llama = ManagedLlama(
                manage="auto",
                binary=emb.get("llama_server"),
                args=tuple(str(a) for a in (emb.get("extra_args") or ())),
                port=emb.get("port"),
                context_size=emb.get("context_size"),
                threads=emb.get("threads"),
                gpu_layers=emb.get("gpu_layers"),
            )

    rec = config.get("recognition") or {}
    if rec.get("ocr"):
        # Legacy degrade semantics: the platform OCR engine left the server
        # build (#496 boundary) — warn and drop rather than refuse boot.
        warnings.append(
            "config: recognition.ocr is deprecated and the platform OCR engine is "
            "not in the server build — recognition stays off (the replacement is "
            "the remote recognizer rows, #502)"
        )

    return Capabilities(
        embedders=tuple(embedders),
        recognizers=(),
        managed_llama=llama,
        managed_sync=None,
        legacy=True,
        warnings=tuple(warnings),
    )


def parse_capabilities(config: Mapping[str, Any]) -> Capabilities:
    """Parse the v2 sections from a loaded config mapping; if none are
    present, synthesize them from the legacy sections (warn-and-map).

    Declaring BOTH v2 and legacy sections is an error — one source of truth.
    """
    has_v2 = any(config.get(k) is not None for k in ("embedders", "recognizers", "managed"))
    has_legacy = bool((config.get("embedding") or {}).get("model")) or bool(
        (config.get("recognition") or {}).get("ocr")
    )
    if not has_v2:
        return _migrate_legacy(config)
    if has_legacy:
        raise ProfileError(
            "config declares both the v2 sections (embedders/recognizers/managed) and "
            "the legacy embedding:/recognition: sections — remove the legacy ones "
            "(they are deprecated and mapped only when v2 is absent)"
        )

    raw_embedders = config.get("embedders") or []
    if not isinstance(raw_embedders, Sequence) or isinstance(raw_embedders, (str, bytes)):
        raise ProfileError("embedders must be a list of entries")
    embedders = tuple(_parse_embedder(raw, i) for i, raw in enumerate(raw_embedders))

    raw_recognizers = config.get("recognizers") or {}
    if not isinstance(raw_recognizers, Mapping):
        raise ProfileError("recognizers must be a mapping keyed by source (ocr/asr/describe)")
    recognizers = tuple(_parse_recognizer(str(k), v) for k, v in raw_recognizers.items())

    llama, sync = _parse_managed(config.get("managed"))
    return Capabilities(
        embedders=embedders,
        recognizers=recognizers,
        managed_llama=llama,
        managed_sync=sync,
    )


def _profile_name(build_features: set[str]) -> str:
    return (
        "server" if "engine-ort" in build_features or "manage-llama" in build_features else "mobile"
    )


def resolve_profile(caps: Capabilities, build_features: Iterable[str]) -> ResolvedProfile:
    """Intersect the declared capabilities with what the build compiled.

    Implements the #498 rules: an uncompiled runtime is a ProfileError naming
    the build profile; a declared capability this release hasn't wired yet is
    a ProfileError naming the tracking issue. Legacy-synthesized sets keep
    legacy degrade semantics (handled in :func:`_migrate_legacy`).
    """
    features = set(build_features)
    profile = _profile_name(features)
    warnings = list(caps.warnings)

    if len(caps.embedders) > 1:
        raise ProfileError(
            f"{len(caps.embedders)} embedder entries declare multiple vector spaces — "
            "multi-space embedding is the #229 substrate and isn't built yet; declare "
            "one entry (one space) for now"
        )

    embedder = caps.embedders[0] if caps.embedders else None
    if embedder is not None:
        feature = _RUNTIME_FEATURE[embedder.runtime]
        if feature not in features:
            raise ProfileError(
                f"embedders[0].runtime: {embedder.runtime} needs the {feature} engine, "
                f"which the {profile} build does not compile"
                + (
                    " — platform engines are never in the server build, on any OS "
                    "(docs/distribution.md)"
                    if embedder.runtime == "platform"
                    else ""
                )
            )
        if embedder.runtime == "remote" and embedder.endpoint is None:
            llama = caps.managed_llama or ManagedLlama()
            if llama.manage == "off":
                raise ProfileError(
                    "embedders[0] declares runtime: remote with no endpoint (= the managed "
                    "llama-server) but managed.llama_server.manage is off — give the entry "
                    "an endpoint or let the manager run"
                )
            if llama.manage == "auto" and "manage-llama" not in features:
                raise ProfileError(
                    f"embedders[0] (remote, no endpoint) needs the managed llama-server, "
                    f"which the {profile} build does not compile (manage-llama) — "
                    "manage: attach (an existing server) works on any build"
                )
        if embedder.runtime == "remote" and embedder.pooling is not None:
            llama = caps.managed_llama or ManagedLlama()
            if embedder.endpoint is not None or llama.manage == "attach":
                # Pooling is applied by the server PRODUCING the vectors; an
                # endpoint Shrike doesn't launch owns its own pooling —
                # accepting the knob here would be a silent no-op.
                raise ProfileError(
                    "embedders[0].pooling applies only when Shrike launches the server "
                    "(managed llama_server, manage: auto) — an external endpoint or an "
                    "attached server owns its own pooling"
                )

    for rec in caps.recognizers:
        if rec.source in ("asr", "describe"):
            raise ProfileError(
                f"recognizers.{rec.source} is declared but the kernel integration for "
                f"asr/describe hasn't landed yet (#485) — remove the entry for now"
            )
        # ocr rows, by runtime:
        if rec.runtime == "platform":
            if _RUNTIME_FEATURE["platform"] not in features:
                raise ProfileError(
                    f"recognizers.ocr.runtime: platform needs the engine-apple engine, which "
                    f"the {profile} build does not compile — platform engines are never in "
                    "the server build, on any OS (docs/distribution.md); the server-profile "
                    "replacement is runtime: remote (#502)"
                )
        elif rec.runtime == "remote":
            raise ProfileError(
                "recognizers.ocr.runtime: remote is the #502 work and hasn't landed yet — "
                "remove the entry for now"
            )
        else:  # onnx
            raise ProfileError(
                "recognizers.ocr.runtime: onnx names a future eval-gated engine that does "
                "not exist yet (docs/distribution.md) — remove the entry for now"
            )

    if caps.managed_sync is not None and caps.managed_sync.manage == "auto":
        raise ProfileError(
            "managed.sync_server.manage: auto is the #36 work and hasn't landed yet — "
            "set it off or remove the entry"
        )

    managed_llama = caps.managed_llama
    if managed_llama is not None and managed_llama.manage in ("auto", "attach"):
        # The managed llama-server exists to serve a remote entry without an
        # endpoint — a section nothing consumes would be a silent no-op (the
        # cross-talk rule), and attach + an explicit endpoint would be two
        # sources for one address. manage: off is a valid explicit "nothing
        # managed" declaration alongside any embedder.
        consumed = (
            embedder is not None and embedder.runtime == "remote" and embedder.endpoint is None
        )
        if not consumed:
            raise ProfileError(
                f"managed.llama_server (manage: {managed_llama.manage}) is declared but "
                "nothing consumes it — it serves an embedders: entry with runtime: remote "
                "and no endpoint; remove the section or set manage: off"
            )
    if (
        managed_llama is not None
        and managed_llama.manage == "attach"
        and any(
            getattr(managed_llama, knob) is not None
            for knob in ("binary", "context_size", "threads", "gpu_layers")
        )
    ):
        # attach uses a server someone else launched — launch-time knobs
        # can't apply (the silent-cross-talk rule again). `port` stays: it's
        # WHERE to attach, not how to launch.
        raise ProfileError(
            "managed.llama_server.manage: attach uses an existing server — the launch "
            "knobs (binary/context_size/threads/gpu_layers/args) don't apply; set port "
            "to say where it listens"
        )
    if managed_llama is not None and managed_llama.manage == "attach" and managed_llama.args:
        raise ProfileError(
            "managed.llama_server.manage: attach uses an existing server — args don't apply"
        )

    return ResolvedProfile(
        embedder=embedder,
        managed_llama=managed_llama,
        warnings=tuple(warnings),
    )


#: Where an attached llama-server is assumed to listen when ``managed.
#: llama_server.port`` is unset — the manager's own default port.
ATTACH_DEFAULT_PORT = 8373


def plan_to_runtime_params(plan: ResolvedProfile) -> dict[str, Any]:
    """Adapt a resolved N=1 plan onto the runtime-params shape
    ``EmbeddingRuntime`` consumes (a superset of the legacy
    ``resolve_embedding`` dict — the ``remote`` kind plus ``endpoint``/
    ``api_key_env`` exist only here; no flag spells them, they ride
    ``--config``).

    The mapping: an onnx entry keys the ort backend by its modalities
    (text → ``onnx``, text+image → ``clip``); a remote entry WITH an
    endpoint — or under ``manage: attach`` — is the unmanaged ``remote``
    backend (Shrike never spawns/stops that server); a remote entry without
    one is the managed llama-server (``manage: auto``, today's behavior).
    """
    e = plan.embedder
    if e is None:
        return {"backend": None, "model": None}
    if e.runtime == "onnx":
        backend = "clip" if "image" in e.modalities else "onnx"
        return {
            "backend": backend,
            "model": e.model,
            "pooling": e.pooling,
            "onnx_providers": list(e.providers),
            "batch_size": e.batch_size,
        }
    if e.runtime == "remote":
        llama = plan.managed_llama or ManagedLlama()
        if e.endpoint is not None or llama.manage == "attach":
            endpoint = e.endpoint or f"http://127.0.0.1:{llama.port or ATTACH_DEFAULT_PORT}"
            return {
                "backend": "remote",
                "model": e.model,
                "endpoint": endpoint,
                "api_key_env": e.api_key_env,
                "batch_size": e.batch_size,
            }
        return {
            "backend": "llama",
            "model": e.model,
            "pooling": e.pooling,
            "batch_size": e.batch_size,
            "llama_server": llama.binary,
            "port": llama.port,
            "extra_args": list(llama.args),
            "context_size": llama.context_size,
            "threads": llama.threads,
            "gpu_layers": llama.gpu_layers,
        }
    raise ProfileError(f"unsupported embedder runtime {e.runtime!r} on this release")
