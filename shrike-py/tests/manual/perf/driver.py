"""Boot a real Harness from a perf profile and time workloads against it.

The runner boots the SAME ``Harness.assemble`` / ``boot`` path the daemon uses,
from a config profile — so a benchmark measures the real serving stack, not a
mock. The only difference between the kernel-isolation and end-to-end runs is
*which profile* (``perf-stub.yml`` → the synthetic embedder; ``perf-real.yml`` →
onnx + CLIP); the runner code is identical.

``time_iterations`` is the single place a profiler attaches (the
attach-a-profiler-to-a-run seam, #866); today it is the clean-timing path only.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

# Allow running from a bare checkout without an editable install.
_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from shrike.harness.collection import _safe_media_name  # noqa: E402
from shrike.harness.engines.embedding.runtime import EmbeddingRuntime  # noqa: E402
from shrike.harness.harness import Harness, KernelIndexView  # noqa: E402
from shrike.harness.profiles import (  # noqa: E402
    parse_capabilities,
    plan_to_runtime_params_set,
    resolve_profile,
)
from tests.manual.perf.result import WorkloadResult  # noqa: E402
from tests.manual.perf.stats import summarize  # noqa: E402


def _load_profile(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def _image_resolver(media_dir: str) -> tuple[Callable[[str], bytes | None], Callable[[str], bool]]:
    """A ``(read, exists)`` pair over the corpus's on-disk media dir — the same
    production wiring the daemon uses (``server._make_image_resolver``), so image
    embedding reads the corpus's PNGs from disk rather than an in-memory map."""

    def _path(name: str) -> str | None:
        safe = _safe_media_name(name)
        return os.path.join(media_dir, safe) if safe else None

    def _read(name: str) -> bytes | None:
        path = _path(name)
        if path is None:
            return None
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            return None

    def _exists(name: str) -> bool:
        path = _path(name)
        return path is not None and os.path.isfile(path)

    return _read, _exists


def _materialize_model(name: str) -> str:
    """Resolve a bare profile model dir-name to an absolute path (perf-real),
    via ``$SHRIKE_TEST_MODEL_DIR`` / the shared model cache, fetching if absent.
    Never reached by perf-stub (synthetic loads no model)."""
    from tests.integration.model_cache import cached_model_path, default_model_cache_base

    return str(cached_model_path(name, default_model_cache_base()))


def _runtime_from_params(params: dict[str, Any]) -> EmbeddingRuntime:
    """One resolved embedder param-dict → an ``EmbeddingRuntime`` (the perf
    subset of the daemon's ``_runtime_from_params``: onnx/clip/synthetic, no
    llama/remote/router shapes — the perf profiles never declare them)."""
    backend = params["backend"]
    model = params.get("model")
    if backend in ("onnx", "clip") and model:
        model = _materialize_model(model)
    kwargs: dict[str, Any] = {
        "backend": backend,
        "model": model,
        "pooling": params.get("pooling"),
        "onnx_providers": params.get("onnx_providers"),
        "batch_size": params.get("batch_size"),
    }
    if params.get("modalities") is not None:
        kwargs["modalities"] = params["modalities"]
    return EmbeddingRuntime(**kwargs)


@dataclass
class Booted:
    """A booted harness + the MCP registry driving the real ``search_notes``."""

    harness: Any
    mcp: Any
    profile_name: str

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        _, structured = await self.mcp.call_tool("search_notes", {"queries": [query], **kwargs})
        return cast("dict[str, Any]", structured)

    async def close(self) -> None:
        await self.harness.close()


async def boot_from_profile(profile_path: Path, collection_path: Path, cache_dir: Path) -> Booted:
    """Boot a harness over ``collection_path`` with the embedders ``profile_path``
    declares, and register the MCP tool surface (so search drives the real
    action). Mirrors the daemon's config→harness sequence."""
    import shrike_native
    from mcp.server.fastmcp import FastMCP

    from shrike.api.tools import register_tools

    plan = resolve_profile(
        parse_capabilities(_load_profile(profile_path)), shrike_native.build_features()
    )
    param_sets = plan_to_runtime_params_set(plan)
    if not param_sets:
        raise ValueError(f"profile {profile_path.name} declares no embedder")
    primary = _runtime_from_params(param_sets[0])
    secondary = [_runtime_from_params(p) for p in param_sets[1:]]

    media_dir = str(collection_path).removesuffix(".anki2") + ".media"
    media_read, media_exists = _image_resolver(media_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    harness = await Harness.assemble(
        collection_path=str(collection_path),
        cache_dir=str(cache_dir),
        runtime=primary,
        cooperative=False,
        hold_seconds=5.0,
        media_read=media_read,
        media_exists=media_exists,
        secondary_runtimes=secondary,
    )
    await harness.boot(start_embedding=True)

    view = KernelIndexView(harness.kernel, harness.runtime)
    mcp = FastMCP("perf")
    register_tools(mcp, harness.wrapper, index=view, kernel=harness.kernel, derived=harness.derived)
    return Booted(harness=harness, mcp=mcp, profile_name=profile_path.stem)


class Workload(Protocol):
    """One timed operation, repeated against a booted harness over a fixed
    corpus. ``setup`` runs once (untimed); ``run_one`` is the timed unit and
    returns how many items it processed.

    A workload MAY also define ``prepare(booted, iteration)`` — an optional
    coroutine run UNTIMED before each timed ``run_one``, for per-iteration setup
    that must stay out of the measurement. ``reconcile`` uses it to introduce fresh
    out-of-band drift; ``setup`` alone can't, since reconcile is idempotent — the
    first reconcile clears all drift, so each iteration needs its own."""

    name: str

    async def setup(self, booted: Booted, iterations: int) -> None: ...

    async def run_one(self, booted: Booted, iteration: int) -> int: ...


async def time_iterations(
    run_one: Callable[[int], Awaitable[int]],
    *,
    repeats: int,
    warmup: int,
    prepare: Callable[[int], Awaitable[None]] | None = None,
) -> tuple[list[float], int]:
    """Run ``run_one(i)`` ``warmup + repeats`` times, returning every iteration's
    wall time (ms) and the last iteration's item count. The warmup samples are
    kept here and discarded by :func:`~tests.manual.perf.stats.summarize`, so the
    raw run is fully recorded. Under ``run.py --instrument`` the whole run executes
    inside a py-spy sampler, so this loop is where the flamegraph's samples land.

    ``prepare(i)``, if given, runs UNTIMED before each iteration's ``run_one`` — the
    per-iteration setup a workload needs out of the timed region (reconcile
    introducing fresh out-of-band drift)."""
    samples: list[float] = []
    items = 0
    for i in range(warmup + repeats):
        if prepare is not None:
            await prepare(i)
        start = time.perf_counter()
        items = await run_one(i)
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples, items


async def measure(
    workload: Workload, booted: Booted, *, repeats: int, warmup: int
) -> WorkloadResult:
    """Run a workload's setup, then time it, then summarize into a result. A
    workload may expose an optional ``prepare(booted, iteration)`` coroutine, run
    untimed before each timed ``run_one`` (see :class:`Workload`)."""
    await workload.setup(booted, repeats + warmup)
    prepare = getattr(workload, "prepare", None)

    async def _prepare(i: int) -> None:
        await prepare(booted, i)

    samples, items = await time_iterations(
        lambda i: workload.run_one(booted, i),
        repeats=repeats,
        warmup=warmup,
        prepare=_prepare if prepare is not None else None,
    )
    return WorkloadResult(
        workload=workload.name,
        distribution=summarize(samples, warmup=warmup),
        items=items,
    )


#: The cold-ingest scenario name. Not in the ``WORKLOADS`` registry — it owns its
#: boot lifecycle (see :func:`measure_ingest`), so the runner dispatches it apart.
INGEST = "ingest"


async def measure_ingest(
    profile_path: Path,
    corpus: Any,
    scratch: Path,
    *,
    repeats: int,
    warmup: int,
    with_media: bool = True,
) -> WorkloadResult:
    """The cold-ingest scenario: import a synthetic package into a FRESH empty
    collection, end-to-end (parse -> write -> derive -> embed -> index, all driven
    by ``import_package`` itself — no follow-up settle).

    It can't run as a ``Workload`` against a shared boot: each sample needs its own
    empty collection, and the driven runtime holds ONE kernel at a time. So it owns
    its boot/close lifecycle here — export the corpus to a package once, then per
    iteration boot a fresh empty harness, import, and close before the next opens.
    Only the ``import_package`` call is timed; the per-iteration cold boot/close sit
    outside the timer (opening an empty collection is setup, not ingest).

    ``corpus`` is a built corpus (needs ``.anki2_path``)."""
    scratch.mkdir(parents=True, exist_ok=True)
    pkg = scratch / "corpus.apkg"

    # Export the corpus to a package — one kernel: boot over the corpus, export, close.
    # Boot over an ISOLATED copy, never the cached corpus: opening a collection can
    # write back to the .anki2, which would corrupt the cached corpus for later runs.
    src_dir = scratch / "export-src"
    src_dir.mkdir(parents=True, exist_ok=True)
    src_anki2 = src_dir / "collection.anki2"
    shutil.copy2(corpus.anki2_path, src_anki2)
    if Path(corpus.media_dir).is_dir():
        # Read-only reuse: export only reads media, so a symlink is enough.
        link = src_dir / "collection.media"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(corpus.media_dir)
    exporter = await boot_from_profile(profile_path, src_anki2, scratch / "export-cache")
    try:
        await exporter.harness.kernel.export_package(
            str(pkg), "apkg", "whole", with_media=with_media
        )
    finally:
        await exporter.close()

    samples: list[float] = []
    items = 0
    for i in range(warmup + repeats):
        iter_dir = scratch / f"iter-{i}"
        booted = await boot_from_profile(
            profile_path, iter_dir / "collection.anki2", iter_dir / "cache"
        )
        try:
            start = time.perf_counter()
            summary_json, _ = await booted.harness.kernel.import_package(
                str(pkg), "if_newer", "if_newer", False, False
            )
            samples.append((time.perf_counter() - start) * 1000.0)
            items = json.loads(summary_json)["found_notes"]
        finally:
            await booted.close()
    return WorkloadResult(
        workload=INGEST, distribution=summarize(samples, warmup=warmup), items=items
    )


def run_async(coro: Awaitable[Any]) -> Any:
    """Drive an async run from the sync entry point."""
    return asyncio.run(coro)  # type: ignore[arg-type]
