#!/usr/bin/env python
"""``//scripts:serve`` — the consolidated dogfooding launcher (#565/#656).

Boots a real Shrike server against a **fresh** collection using a checked-in,
path-free capability *profile* (``scripts/profiles/<name>.yml``), with the
profile's models materialized from Bazel externals — zero new download code, no
URL re-spelling. It is the spine the offline-integration milestone layers on; it
retires ``scripts/launch-qa-server.sh`` (whose job is now
``serve --profile <name> --seed qa``).

Usage (under Bazel — the model externals ride the binary's ``data`` deps):

    ./bazel run //scripts:serve -- --profile text-onnx [--seed qa] [--foreground|--daemon]

Profiles are **path-free** (the hard invariant): no ``collection:`` key, no
machine-absolute paths. An onnx embedder's ``model:`` is a bare *dir-name* (e.g.
``all-MiniLM-L6-v2-onnx-int8``); the launcher materializes that dir into the
per-run model tree and rewrites the entry's ``model:`` to the absolute path
before handing the effective config to the server. Run paths (collection, cache,
logs) ride as flags, not config.

Model materialization mirrors ``tests/integration/conftest.py``'s
``_populate_bazel_model_dir`` but for ``bazel run`` (there is no ``TEST_TMPDIR``):
under Bazel each profile-named model is located in the runfiles and copied into
``<run>/models/<dir-name>/...``; off Bazel (a plain checkout) the matching
``model_cache.cached_*_model_dir`` fetches it (the single Python fetch source).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from shrike.embedding_onnx_common import resolve_execution_providers

logger = logging.getLogger("shrike.serve")

# The repo root: scripts/serve.py → repo root is one level up. Under Bazel this
# is the runfiles-relative source location; only used for the non-Bazel fallback
# paths (profiles dir, scripts/.run), all of which are source-tree relative.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROFILES_DIR = _REPO_ROOT / "scripts" / "profiles"

# Per-run output home for the non-Bazel (script-fallback) lane. Gitignored.
_SCRIPT_RUN_ROOT = _REPO_ROOT / "scripts" / ".run"


# -- Model materialization: profile dir-name → how to source it ----------------
#
# A profile names each onnx model by its model_cache *_DIR_NAME (the same layout
# the integration conftest assembles into SHRIKE_TEST_MODEL_DIR). For each such
# dir-name we record BOTH source paths so neither lane re-spells a URL:
#   - bazel: the http_file external's runfiles paths → the dir/<file> layout
#     (mirrors conftest's _BAZEL_MODELS, keyed the other way: dir-name → files).
#   - script fallback: the model_cache.cached_*_model_dir function that fetches it.
#
# Add a row here (not a new download path) when a profile reuses another pinned
# external. The keys are model_cache's *_DIR_NAME constants, imported so a rename
# there can't silently drift this map.


def _model_sources() -> dict[str, dict[str, Any]]:
    """The dir-name → {bazel runfiles map, script fetch fn} table.

    Imported lazily so the pure-logic surface (arg parsing, profile load,
    config composition) stays importable without ``tests`` on ``sys.path`` —
    the Bazel unit test exercises that surface with this table stubbed.
    """
    # tests.integration.model_cache is the single source of model dir-names +
    # the script-fallback fetchers. imports=["../.."] in the test BUILD puts the
    # repo root on sys.path under Bazel; the non-Bazel lane runs from a checkout.
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from tests.integration import model_cache as mc

    return {
        mc.ONNX_MODEL_DIR_NAME: {
            # runfiles path (MODULE.bazel http_file) → file name within the dir
            "bazel": {
                "model_minilm_int8_onnx/file/model.onnx": "model.onnx",
                "model_minilm_tokenizer/file/tokenizer.json": "tokenizer.json",
            },
            "fetch": mc.cached_onnx_model_dir,
        },
    }


def _runfiles() -> Any | None:
    """The Bazel runfiles resolver, or ``None`` off Bazel (the script lane)."""
    if not (os.environ.get("RUNFILES_DIR") or os.environ.get("RUNFILES_MANIFEST_FILE")):
        return None
    try:
        from python.runfiles import runfiles  # type: ignore[import-not-found]
    except ImportError:
        return None
    return runfiles.Create()


def materialize_model(dir_name: str, models_root: Path) -> Path:
    """Materialize the model named *dir_name* into ``<models_root>/<dir_name>/``.

    Under Bazel, copy each of the model's files out of the runfiles into the run
    tree (so the server reads a stable on-disk dir, not a sandbox path that may
    vanish). Off Bazel, delegate to the matching ``model_cache`` fetcher (the
    single Python download source). Returns the materialized model dir.
    """
    if Path(dir_name).is_absolute():
        # A profile must name a model by bare dir-name, not a machine path — the
        # path-free invariant the launcher exists to enforce. Catch it here with a
        # pointed message rather than the misleading "don't know how to materialize".
        raise SystemExit(
            f"profile names model {dir_name!r} as an absolute path — profiles are path-free; "
            f"name the model by its bare dir-name (the launcher materializes it and rewrites "
            f"the path)"
        )
    sources = _model_sources()
    if dir_name not in sources:
        raise SystemExit(
            f"profile names model {dir_name!r} which the launcher doesn't know how to "
            f"materialize — add it to scripts/serve.py's model-source table "
            f"(known: {', '.join(sorted(sources)) or 'none'})"
        )
    spec = sources[dir_name]
    model_dir = models_root / dir_name

    r = _runfiles()
    if r is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
        missing: list[str] = []
        for src, file_name in spec["bazel"].items():
            loc = r.Rlocation(src)
            if not loc or not os.path.exists(loc):
                missing.append(src)
                continue
            dest = model_dir / file_name
            if not (dest.exists() and dest.stat().st_size > 0):
                tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
                shutil.copy(loc, tmp)
                os.replace(tmp, dest)
        # EVERY declared file must be present, or the materialized dir is partial
        # and the failure would surface late + cryptically at the backend's
        # _resolve_files ("no tokenizer.json" / "no text+vision pair"). Fail loud
        # HERE — a missing file means a forgotten @model_* data dep (bites the
        # 4-file CLIP dir of the profile-(c) wave; harmless silent for 2-file MVP).
        if missing:
            raise SystemExit(
                f"model {dir_name!r}: {len(missing)}/{len(spec['bazel'])} file(s) not found in "
                f"Bazel runfiles ({', '.join(missing)}) — add the corresponding @model_* "
                f"external(s) to //scripts:serve's `data` deps (and confirm scripts/serve.py's "
                f"model-source table names them)"
            )
        logger.info("materialized model %s from Bazel runfiles → %s", dir_name, model_dir)
        return model_dir

    # Non-Bazel: fetch via model_cache (the single download source).
    logger.info("materializing model %s via model_cache fetch → %s", dir_name, model_dir)
    fetched = spec["fetch"](models_root)
    return Path(fetched)


# -- ONNX execution-provider auto-detect (#569) --------------------------------
#
# Profiles are provider-FREE (portable, one file per capability shape — provider
# is orthogonal to a profile's identity). The launcher detects the providers at
# materialize time and overlays them onto every ONNX entry, so a GPU host gets
# GPU acceleration with zero per-platform profile drift. Detection rides the SAME
# source the backend uses (`onnxruntime.get_available_providers` →
# `resolve_execution_providers`), so the printed list can't disagree with what
# the backend will actually run; detection failure degrades to CPU (slower but
# correct), never a crash or wrong vectors.

#: GPU/accelerated execution providers in PRIORITY order — the candidate request
#: list before intersecting with what onnxruntime actually has. CPU is appended
#: by the shared resolver, never listed here. Mirrors the native EP mapping the
#: backends document (CUDA→TensorRT on onnxruntime-gpu; CoreML on the macOS base
#: wheel; DirectML on onnxruntime-directml).
_PROVIDER_PRIORITY = (
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
)

#: Accelerated providers whose ABSENCE on a GPU-looking host means the wrong
#: onnxruntime carrier is installed (the one thing auto-detect can't fix).
_NVIDIA_PROVIDERS = frozenset({"CUDAExecutionProvider", "TensorrtExecutionProvider"})


def _available_providers() -> list[str] | None:
    """``onnxruntime.get_available_providers()``, or ``None`` if onnxruntime is
    absent. The single source of truth the backend intersects against too."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    return list(ort.get_available_providers())


def _nvidia_gpu_present() -> bool:
    """A best-effort 'this host has an NVIDIA GPU' probe — ``nvidia-smi`` on PATH.

    Used ONLY to decide whether to emit the carrier-mismatch remedy; it never
    changes the resolved provider list (detection rides onnxruntime, not this)."""
    return shutil.which("nvidia-smi") is not None


def detect_providers() -> list[str]:
    """The resolved ONNX execution-provider list for this host (priority,
    intersected with what onnxruntime has, CPU last).

    Returns ``["CPUExecutionProvider"]`` when onnxruntime is absent (the backend
    would fail to start anyway, but the list stays well-formed). Emits the
    carrier-mismatch warning when a GPU-looking host lacks its GPU EP."""
    available = _available_providers()
    if available is None:
        logger.warning(
            "onnxruntime not importable — defaulting ONNX providers to CPU "
            "(the embedding backend needs the onnxruntime wheel to start)"
        )
        return ["CPUExecutionProvider"]
    # The shared resolver is the same code the backend runs: keep requested-and-
    # available in priority order, append CPU, dedup. So the printed list IS the
    # backend's list.
    resolved, _dropped = resolve_execution_providers(available, list(_PROVIDER_PRIORITY))
    _warn_on_carrier_mismatch(available)
    return resolved


def _warn_on_carrier_mismatch(available: list[str]) -> None:
    """Warn when the host looks GPU-capable but the installed onnxruntime carrier
    can't use it — the one failure auto-detect cannot repair, so it must tell the
    user the exact remedy."""
    has_nvidia_ep = any(p in available for p in _NVIDIA_PROVIDERS)
    if _nvidia_gpu_present() and not has_nvidia_ep:
        logger.warning(
            "an NVIDIA GPU is present (nvidia-smi) but the installed onnxruntime has no "
            "CUDA/TensorRT execution provider — embedding will run on CPU. Install the GPU "
            "carrier to use it:  pip uninstall onnxruntime && pip install onnxruntime-gpu"
        )


def resolve_providers(args: argparse.Namespace) -> list[str] | None:
    """The launcher's provider decision for this run, or ``None`` to leave a
    profile's own ``providers:`` untouched (an explicit profile choice wins).

    ``--cpu`` forces CPU-only; ``--providers A,B`` forces an explicit list
    (skipping detection); otherwise auto-detect. The return is the list to
    OVERLAY onto onnx entries that don't already declare ``providers:``."""
    if args.cpu:
        return ["CPUExecutionProvider"]
    if args.providers:
        # Explicit override: take the user's order verbatim (the backend still
        # intersects + CPU-falls-back at start, so a typo degrades, not crashes).
        return list(args.providers)
    return detect_providers()


# -- Profile loading + effective-config composition ----------------------------


def profile_path(name: str) -> Path:
    """Resolve a ``--profile NAME`` to its checked-in YAML path."""
    path = _PROFILES_DIR / f"{name}.yml"
    if not path.is_file():
        available = sorted(p.stem for p in _PROFILES_DIR.glob("*.yml"))
        raise SystemExit(
            f"unknown profile {name!r} — no {path.name} under scripts/profiles/ "
            f"(available: {', '.join(available) or 'none'})"
        )
    return path


def check_path_free(name: str, raw: Mapping[str, Any]) -> None:
    """Enforce the path-free invariant on a loaded profile mapping.

    A profile must not carry a ``collection:`` key: run paths ride as flags,
    models are bare dir-names rewritten to absolute paths at compose time.
    """
    if "collection" in raw:
        raise SystemExit(
            f"profile {name!r} declares collection: — profiles are path-free; the "
            f"launcher supplies the run collection as a flag"
        )


def load_profile(name: str) -> dict[str, Any]:
    """Load a profile YAML and enforce the path-free invariant."""
    raw = yaml.safe_load(profile_path(name).read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"profile {name!r} must be a YAML mapping")
    check_path_free(name, raw)
    return raw


def _model_names_in_profile(profile: Mapping[str, Any]) -> list[str]:
    """Every onnx embedder model dir-name the profile names, in order.

    Only onnx embedders carry a materializable model dir-name (the MVP surface).
    Remote/managed entries name endpoints/binaries, not bundled model dirs, and
    are left to later waves.
    """
    names: list[str] = []
    for entry in profile.get("embedders") or []:
        if isinstance(entry, Mapping) and entry.get("runtime") == "onnx":
            model = entry.get("model")
            if isinstance(model, str) and model:
                names.append(model)
    return names


def compose_effective_config(
    profile: Mapping[str, Any],
    resolved_models: Mapping[str, str],
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Rewrite a path-free profile into a server-ready effective config.

    Each onnx embedder's ``model:`` (a bare dir-name) is replaced with the
    absolute materialized path from *resolved_models*. Everything else passes
    through unchanged. A model with no resolved path is a programming error
    (the caller materializes every name from :func:`_model_names_in_profile`).

    *providers* (when not ``None``) is the launcher-resolved ONNX execution
    provider list (#569), overlaid onto every **onnx** entry that does not
    already declare its own ``providers:`` — an explicit profile choice wins over
    detection. Remote/platform entries are untouched (``providers:`` is onnx-only,
    mirroring the guard at ``profiles.py:291``).
    """
    config: dict[str, Any] = {k: v for k, v in profile.items() if k != "embedders"}
    embedders: list[Any] = []
    for entry in profile.get("embedders") or []:
        if isinstance(entry, Mapping):
            new_entry = dict(entry)
            if new_entry.get("runtime") == "onnx":
                model = new_entry.get("model")
                if isinstance(model, str) and model:
                    if model not in resolved_models:
                        raise KeyError(f"no materialized path for model {model!r}")
                    new_entry["model"] = resolved_models[model]
                # Overlay detected providers ONLY when the profile didn't set its
                # own (a checked-in profile carries none; an explicit one wins).
                if providers is not None and "providers" not in new_entry:
                    new_entry["providers"] = list(providers)
            embedders.append(new_entry)
        else:
            embedders.append(entry)
    if embedders or "embedders" in profile:
        config["embedders"] = embedders
    return config


# -- Run layout + collection seeding -------------------------------------------


def run_root() -> Path:
    """The per-run isolated output dir, created fresh.

    Under ``bazel run`` use ``$TEST_TMPDIR`` if present (rare) else
    ``$TMPDIR/shrike-serve/<ts>``; off Bazel use gitignored
    ``scripts/.run/<ts>/``. A timestamp keeps runs side by side for inspection.
    """
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    test_tmp = os.environ.get("TEST_TMPDIR")
    if _runfiles() is not None:
        base = (
            Path(test_tmp) if test_tmp else Path(os.environ.get("TMPDIR", "/tmp")) / "shrike-serve"
        )
    else:
        base = _SCRIPT_RUN_ROOT
    root = base / ts
    root.mkdir(parents=True, exist_ok=True)
    return root


def seed_qa_collection(collection_path: Path) -> None:
    """Generate the ``tests/qa`` synthetic fixture into *collection_path*.

    Reuses ``tests/qa/build_collection.py``'s ``build`` — the same write path
    ``launch-qa-server.sh`` drove, now a launcher seed.
    """
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from tests.qa.build_collection import build

    spec = _REPO_ROOT / "tests" / "qa" / "collection.json"
    logger.info("seeding qa fixture from %s → %s", spec, collection_path)
    rc = build(spec, collection_path)
    if rc != 0:
        raise SystemExit(f"qa fixture build reported errors (rc={rc})")


# -- Argument parsing ----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serve",
        description="Boot a real Shrike server against a fresh collection from a "
        "checked-in capability profile (offline-integration dogfooding, #565).",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Capability profile name → scripts/profiles/<name>.yml (path-free).",
    )
    seed = parser.add_mutually_exclusive_group()
    seed.add_argument(
        "--seed",
        choices=["qa"],
        help="Seed the fresh collection with a named fixture (qa = the tests/qa corpus).",
    )
    seed.add_argument(
        "--import",
        dest="import_path",
        metavar="PATH.apkg",
        help="Seed the fresh collection by importing an .apkg/.colpkg (not yet wired; "
        "stubbed for MVP — tracked for a later wave).",
    )
    fg = parser.add_mutually_exclusive_group()
    fg.add_argument(
        "--foreground",
        dest="foreground",
        action="store_true",
        default=True,
        help="Run the server in the foreground (default; Ctrl+C to stop).",
    )
    fg.add_argument(
        "--daemon",
        dest="foreground",
        action="store_false",
        help="Run the server as a background daemon (stop with `shrike server stop`).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Override the run collection path (default: a fresh one under the run dir).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override the run cache dir (default: <run>/cache).",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Override the run log dir (default: <run>/logs).",
    )
    prov = parser.add_mutually_exclusive_group()
    prov.add_argument(
        "--providers",
        type=_split_providers,
        default=None,
        metavar="A,B",
        help="Override ONNX execution-provider auto-detect with an explicit "
        "comma-separated list in priority order (e.g. "
        "CUDAExecutionProvider,CPUExecutionProvider). The backend still "
        "intersects + falls back to CPU, so a typo degrades, not crashes.",
    )
    prov.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only ONNX execution (sugar for "
        "--providers CPUExecutionProvider); skips auto-detect.",
    )
    return parser


def _split_providers(value: str) -> list[str]:
    """Parse ``--providers A,B`` into a clean list (empty entries dropped)."""
    items = [p.strip() for p in value.split(",")]
    return [p for p in items if p]


def _status_port(profile: Mapping[str, Any]) -> int:
    """The port the server will listen on — a profile's ``server.port`` else the
    default 8372 (the launcher doesn't pass --port, so the CLI uses this same
    resolution). Used to poll /status for the active-provider readback."""
    server = profile.get("server")
    if isinstance(server, Mapping) and isinstance(server.get("port"), int):
        return int(server["port"])
    return 8372


def read_active_providers(port: int, *, timeout: float = 30.0) -> list[str] | None:
    """Poll ``GET /status`` until the daemon responds, then return its embedding
    ``active_providers`` (what onnxruntime ACTUALLY loaded — GPU or CPU).

    Returns ``None`` if the server never responds in *timeout* or carries no
    active providers (e.g. embedding off). Best-effort: a readback failure is a
    visibility gap, never a launch failure."""
    import time

    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/status", timeout=5.0)
            resp.raise_for_status()
        except (httpx.HTTPError, httpx.TransportError):
            time.sleep(0.5)
            continue
        emb = resp.json().get("embedding") or {}
        active = emb.get("active_providers")
        if active:
            return list(active)
        # Server up but no active providers yet (engine still starting) — retry.
        time.sleep(0.5)
    return None


def _server_argv(
    *,
    config_path: Path,
    collection_path: Path,
    cache_dir: Path,
    log_dir: Path,
    foreground: bool,
) -> list[str]:
    """The ``shrike … server start`` argv the launcher invokes.

    The capability config rides as ``--config`` (config-file-only — the launcher
    invents NO flags for embedders/managed); run paths ride as flags.
    """
    argv = [
        "--config",
        str(config_path),
        "server",
        "start",
        "--collection",
        str(collection_path),
        "--cache-dir",
        str(cache_dir),
        "--log-dir",
        str(log_dir),
    ]
    if foreground:
        argv.append("--foreground")
    return argv


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)

    if args.import_path is not None:
        # MVP stub: real .apkg import is a later-wave seam (epic #565). Refuse
        # loudly rather than silently start an empty collection.
        raise SystemExit(
            "--import is not wired yet (the .apkg seed seam lands in a later wave); "
            "use --seed qa or run without a seed for a fresh empty collection"
        )

    profile = load_profile(args.profile)

    root = run_root()
    models_root = root / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else root / "cache"
    log_dir = Path(args.log_dir) if args.log_dir else root / "logs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    collection_path = (
        Path(args.collection) if args.collection else root / "collection" / "working.anki2"
    )
    collection_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("run dir: %s", root)

    resolved_models: dict[str, str] = {}
    for name in _model_names_in_profile(profile):
        if name not in resolved_models:
            resolved_models[name] = str(materialize_model(name, models_root))

    # Resolve ONNX execution providers for this host (auto-detect, or the
    # --providers/--cpu override) and overlay them onto onnx entries (#569). The
    # BEFORE readout: print what we'll request, so the after-readback from
    # /status makes "did the GPU engage?" unambiguous.
    providers = resolve_providers(args)
    if providers is not None and _model_names_in_profile(profile):
        logger.info("ONNX execution providers (resolved, priority order): %s", ", ".join(providers))
    config = compose_effective_config(profile, resolved_models, providers=providers)
    config_path = root / "config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    logger.info("wrote effective config → %s", config_path)

    if args.seed == "qa":
        seed_qa_collection(collection_path)

    server_argv = _server_argv(
        config_path=config_path,
        collection_path=collection_path,
        cache_dir=cache_dir,
        log_dir=log_dir,
        foreground=args.foreground,
    )
    logger.info("launching: shrike %s", " ".join(server_argv))

    has_onnx = bool(_model_names_in_profile(profile))
    if args.foreground and has_onnx:
        # Foreground blocks on the server, so we can't poll /status after — point
        # the user at where the ACTIVE provider lives (the before-readout above is
        # the resolved request; `shrike server status` shows what actually loaded).
        logger.info(
            "foreground mode: the ACTIVE ONNX provider (what onnxruntime loaded — "
            "GPU vs the CPU fallback) is in `shrike server status` once it's up"
        )

    # Invoke the CLI in-process: under `bazel run` this keeps the runfiles
    # interpreter (so foreground main() and a daemon's `-m shrike.server` spawn
    # both resolve the shrike package). standalone_mode=False so Click returns
    # instead of calling sys.exit, letting us return a clean code.
    from shrike.cli import cli

    rc = int(cli.main(args=server_argv, prog_name="shrike", standalone_mode=False) or 0)

    # Daemon mode returns once the child is up — poll /status for the AFTER
    # readback so the before/after provider comparison is visible in one run.
    if not args.foreground and has_onnx and rc == 0:
        active = read_active_providers(_status_port(profile))
        if active:
            logger.info(
                "ONNX execution providers (ACTIVE — what onnxruntime loaded): %s", ", ".join(active)
            )
            if providers and active[0] != providers[0]:
                logger.warning(
                    "active provider %s differs from the resolved first choice %s — "
                    "onnxruntime fell back (see `shrike server status` / the server log)",
                    active[0],
                    providers[0],
                )
        else:
            logger.info(
                "could not read back the active provider from /status — check "
                "`shrike server status` (a visibility gap, not a launch failure)"
            )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
