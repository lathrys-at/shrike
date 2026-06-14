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
    # Per-modality multimodal projectors (#501) — loaded with the managed
    # server so it can embed images/audio. Empty for a text-only server.
    mmprojs: tuple[str, ...] = ()


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
class RecognizerPlan:
    """One resolved recognition engine (#485): the harness-ready shape a
    ``recognizers:`` entry maps onto. ``purpose`` is the kernel routing key /
    derived source (``ocr``/``describe``/``asr``); ``kind`` is the construction
    selector (``apple`` for the platform OCR engine, ``describe-remote`` for the
    remote VLM). Remote engines carry their ``endpoint`` + optional
    ``api_key_env`` (resolved to a token at construction, never inlined)."""

    purpose: str
    kind: str
    model: str | None = None
    endpoint: str | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class ResolvedEmbedder:
    """One resolved embedding space (#233): the declared entry plus its
    routing **role** — the per-modality PRIMARY flags. A modality's primary is
    the FIRST entry (declaration order) that declares it, which mirrors the
    kernel's insertion-order primary (``EmbedSpaces::primary``). The role is
    metadata this PR; the index-narrow / query-wide fan-out it feeds is PR-B/C
    (#232/#234)."""

    entry: EmbedderEntry
    #: The note modalities this space is PRIMARY for (it is the first declared
    #: space carrying each). The index fan-out routes a note item to its
    #: modality's primary space.
    primary_modalities: frozenset[str]

    @property
    def runtime(self) -> str:
        return self.entry.runtime

    @property
    def modalities(self) -> tuple[str, ...]:
        return self.entry.modalities

    @property
    def text_capable(self) -> bool:
        """Whether this space embeds the TEXT modality — the query-routing
        flag (a query fans out to every text-capable space in PR-C)."""
        return "text" in self.entry.modalities


@dataclass(frozen=True)
class ResolvedProfile:
    """The declared set intersected with the build: what this process will
    actually serve. ``embedders`` is the ordered set of resolved spaces (#233 —
    the multi-space substrate; an empty tuple means no embedder, one entry is
    the N=1 case); ``recognizers`` is the per-purpose recognition set (#485);
    ``warnings`` aggregates migration + degradation messages."""

    embedders: tuple[ResolvedEmbedder, ...]
    managed_llama: ManagedLlama | None
    recognizers: tuple[RecognizerEntry, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def embedder(self) -> EmbedderEntry | None:
        """The PRIMARY (first) embedder entry, or ``None`` — the N=1
        back-compat accessor. The index path consumes one engine until the
        fan-out lands (PR-B/C), so the primary entry is the load-bearing one."""
        return self.embedders[0].entry if self.embedders else None


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
            (
                "manage",
                "binary",
                "args",
                "port",
                "context_size",
                "threads",
                "gpu_layers",
                "mmprojs",
            ),
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
        mmprojs_raw = lraw.get("mmprojs") or ()
        if isinstance(mmprojs_raw, str) or not isinstance(mmprojs_raw, Sequence):
            raise ProfileError(f"{where}.mmprojs must be a list of projector paths")
        llama = ManagedLlama(
            manage=manage,
            binary=_opt_str(lraw, "binary", where),
            args=tuple(str(a) for a in args_raw),
            port=_opt_int(lraw, "port", where),
            context_size=_opt_int(lraw, "context_size", where),
            threads=_opt_int(lraw, "threads", where),
            gpu_layers=_opt_int(lraw, "gpu_layers", where),
            mmprojs=tuple(str(m) for m in mmprojs_raw),
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


def _resolve_embedder_roles(
    embedders: tuple[EmbedderEntry, ...],
) -> tuple[ResolvedEmbedder, ...]:
    """Assign each space its per-modality PRIMARY role (#233): a modality's
    primary is the FIRST entry (declaration order) that declares it, mirroring
    the kernel's insertion-order primary. So the first text space is primary
    for text, the first image space primary for image — and a single entry is
    primary for every modality it carries (the N=1 case)."""
    seen: set[str] = set()
    resolved: list[ResolvedEmbedder] = []
    for entry in embedders:
        primary = frozenset(m for m in entry.modalities if m not in seen)
        seen.update(entry.modalities)
        resolved.append(ResolvedEmbedder(entry=entry, primary_modalities=primary))
    return tuple(resolved)


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

    # Multi-space is built since #233: each declared entry is its own vector
    # space, validated independently against the build + this release. The
    # per-modality PRIMARY is the FIRST entry carrying the modality (mirrors the
    # kernel's insertion-order primary). The remote/managed-llama coupling is
    # validated per entry below; the managed-llama-consumption check (further
    # down) looks across ALL entries.
    for index, embedder in enumerate(caps.embedders):
        feature = _RUNTIME_FEATURE[embedder.runtime]
        if feature not in features:
            raise ProfileError(
                f"embedders[{index}].runtime: {embedder.runtime} needs the {feature} engine, "
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
                    f"embedders[{index}] declares runtime: remote with no endpoint (= the "
                    "managed llama-server) but managed.llama_server.manage is off — give the "
                    "entry an endpoint or let the manager run"
                )
            if llama.manage == "auto" and "manage-llama" not in features:
                raise ProfileError(
                    f"embedders[{index}] (remote, no endpoint) needs the managed llama-server, "
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
                    f"embedders[{index}].pooling applies only when Shrike launches the server "
                    "(managed llama_server, manage: auto) — an external endpoint or an "
                    "attached server owns its own pooling"
                )

    # At most ONE managed llama-server backs the embedder set: a remote
    # no-endpoint entry consumes it. Two such entries would both bind the one
    # managed server, which is ambiguous — reject rather than silently share.
    managed_remote_entries = sum(
        1 for e in caps.embedders if e.runtime == "remote" and e.endpoint is None
    )
    if managed_remote_entries > 1:
        raise ProfileError(
            f"{managed_remote_entries} embedder entries are remote with no endpoint — each "
            "would bind the single managed llama-server; give all but one an explicit endpoint"
        )

    # At most ONE IMAGE-embedding space (#580). Cross-space fusion admits a
    # secondary image space on its own calibrated floor (floor-admission, the
    # production mechanism since #580) — the relative winner-take-all gate that
    # used to bound MULTIPLICITY is retired. With a single image space there is
    # no multiplicity to bound, so retiring it is sound; declaring two image
    # spaces would reintroduce the N≥2 flood the gate guarded against (the eval
    # showed text recall collapses), with no mechanism left to stop it. So it is
    # a config error, not a silent degrade.
    image_entries = [i for i, e in enumerate(caps.embedders) if "image" in e.modalities]
    if len(image_entries) > 1:
        raise ProfileError(
            f"embedders[{', '.join(str(i) for i in image_entries)}] each declare the image "
            "modality — Shrike supports at most ONE image-embedding space (#580: cross-space "
            "fusion admits a single image space on its calibrated floor; two would flood the "
            "fusion with no gate to bound them). Keep one image space; a second text-only space "
            "is fine"
        )

    for rec in caps.recognizers:
        if rec.source == "asr":
            raise ProfileError(
                "recognizers.asr is declared but the kernel integration for asr hasn't "
                "landed yet (#485 PR2) — remove the entry for now"
            )
        if rec.source == "describe":
            # describe is attachable now (#485 PR1) — VLM image→text into the
            # embedding space (vector-only). The remote runtime (any
            # OpenAI-compatible vision endpoint) is the wired shape; platform
            # (engine-apple) and onnx describe engines don't exist yet.
            if rec.runtime != "remote":
                raise ProfileError(
                    f"recognizers.describe.runtime: {rec.runtime} is not a describe engine — "
                    "the wired shape is runtime: remote (any OpenAI-compatible vision "
                    "endpoint); declare endpoint + optional api_key_env"
                )
            if _RUNTIME_FEATURE["remote"] not in features:
                raise ProfileError(
                    f"recognizers.describe.runtime: remote needs the engine-remote engine, "
                    f"which the {profile} build does not compile"
                )
            if rec.endpoint is None:
                raise ProfileError(
                    "recognizers.describe.runtime: remote needs an endpoint (the "
                    "OpenAI-compatible vision server — a managed describe server is a "
                    "future capability; point at a running endpoint for now)"
                )
            continue
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
        consumed = any(e.runtime == "remote" and e.endpoint is None for e in caps.embedders)
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
    if managed_llama is not None and managed_llama.manage == "attach" and managed_llama.mmprojs:
        # An attached server loads its own projectors at its own launch.
        raise ProfileError(
            "managed.llama_server.manage: attach uses an existing server — mmprojs don't "
            "apply (the server you attach to loads its own); declare the embedder's "
            "image modality and start that server with its --mmproj"
        )
    if (
        managed_llama is not None
        and managed_llama.mmprojs
        and not any("image" in e.modalities for e in caps.embedders)
    ):
        # Projectors loaded for a text-only space are never used — a silent
        # no-op (the cross-talk rule). The mmprojs serve the image half. (The
        # managed server backs the one remote-no-endpoint entry; an image
        # modality on ANY declared space satisfies the consumer.)
        raise ProfileError(
            "managed.llama_server.mmprojs is set but no embedder declares an image "
            "modality — add image to an entry's modalities, or drop the projectors"
        )

    return ResolvedProfile(
        embedders=_resolve_embedder_roles(caps.embedders),
        managed_llama=managed_llama,
        recognizers=caps.recognizers,
        warnings=tuple(warnings),
    )


#: Where an attached llama-server is assumed to listen when ``managed.
#: llama_server.port`` is unset — the manager's own default port.
ATTACH_DEFAULT_PORT = 8373


def _entry_to_runtime_params(
    e: EmbedderEntry, managed_llama: ManagedLlama | None
) -> dict[str, Any]:
    """Map ONE resolved embedder entry onto the runtime-params dict
    ``EmbeddingRuntime`` consumes — the per-entry mapping shared by the N=1
    primary accessor (:func:`plan_to_runtime_params`) and the N-dict set
    (:func:`plan_to_runtime_params_set`). Reused verbatim per #233's scope (the
    multi-space change is fanning OUT this mapping, not altering it).

    The mapping: an onnx entry keys the ort backend by its modalities
    (text → ``onnx``, text+image → ``clip``); a remote entry WITH an
    endpoint — or under ``manage: attach`` — is the unmanaged ``remote``
    backend (Shrike never spawns/stops that server); a remote entry without
    one is the managed llama-server (``manage: auto``, today's behavior).
    """
    # The space's modalities flow to the backend (#501): an image space
    # composes the image half + reports image coverage.
    modalities = frozenset(e.modalities)
    if e.runtime == "onnx":
        backend = "clip" if "image" in e.modalities else "onnx"
        return {
            "backend": backend,
            "model": e.model,
            "pooling": e.pooling,
            "onnx_providers": list(e.providers),
            "batch_size": e.batch_size,
            "modalities": modalities,
        }
    if e.runtime == "remote":
        llama = managed_llama or ManagedLlama()
        if e.endpoint is not None or llama.manage == "attach":
            endpoint = e.endpoint or f"http://127.0.0.1:{llama.port or ATTACH_DEFAULT_PORT}"
            return {
                "backend": "remote",
                "model": e.model,
                "endpoint": endpoint,
                "api_key_env": e.api_key_env,
                "batch_size": e.batch_size,
                "modalities": modalities,
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
            "modalities": modalities,
            "mmprojs": list(llama.mmprojs),
        }
    raise ProfileError(f"unsupported embedder runtime {e.runtime!r} on this release")


def plan_to_runtime_params(plan: ResolvedProfile) -> dict[str, Any]:
    """The PRIMARY embedder's runtime-params dict (the N=1 accessor) — what
    the index/search paths' single engine consume this release. With one
    declared embedder it is the sole space, so the dict is byte-identical to
    the single-space era; the multi-space fan-out reads
    :func:`plan_to_runtime_params_set`.
    """
    e = plan.embedder
    if e is None:
        return {"backend": None, "model": None}
    return _entry_to_runtime_params(e, plan.managed_llama)


def plan_to_runtime_params_set(plan: ResolvedProfile) -> tuple[dict[str, Any], ...]:
    """The runtime-params dict for EVERY resolved space, in declaration order
    (#233): the harness/``EmbeddingRuntime`` fan-out attaches one backend per
    dict, each to its own kernel embed space. An empty plan yields an empty
    tuple. Each dict is the same per-entry mapping :func:`plan_to_runtime_params`
    emits for the primary — N=1 yields a 1-tuple whose sole element equals the
    primary dict, so the single-space runtime is unchanged."""
    return tuple(_entry_to_runtime_params(re.entry, plan.managed_llama) for re in plan.embedders)


#: A resolved recognizer entry's runtime → the harness construction kind.
_RECOGNIZER_KIND = {
    ("ocr", "platform"): "apple",
    ("describe", "remote"): "describe-remote",
}


def recognizer_plans(plan: ResolvedProfile) -> tuple[RecognizerPlan, ...]:
    """Adapt the resolved recognizers onto the harness-ready
    :class:`RecognizerPlan` shape (#485) — the recognizer analogue of
    :func:`plan_to_runtime_params`. ``resolve_profile`` has already validated
    each entry against the build and this release, so this is a pure mapping;
    an entry it doesn't know how to construct is a ProfileError (the
    validation and the mapping must stay in lockstep)."""
    plans: list[RecognizerPlan] = []
    for rec in plan.recognizers:
        kind = _RECOGNIZER_KIND.get((rec.source, rec.runtime))
        if kind is None:
            raise ProfileError(
                f"recognizers.{rec.source}.runtime: {rec.runtime} has no construction "
                "path on this release"
            )
        plans.append(
            RecognizerPlan(
                purpose=rec.source,
                kind=kind,
                model=rec.model,
                endpoint=rec.endpoint,
                api_key_env=rec.api_key_env,
            )
        )
    return tuple(plans)
