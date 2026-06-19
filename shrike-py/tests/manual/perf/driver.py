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
import os
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
    returns how many items it processed."""

    name: str

    async def setup(self, booted: Booted) -> None: ...

    async def run_one(self, booted: Booted, iteration: int) -> int: ...


async def time_iterations(
    run_one: Callable[[int], Awaitable[int]], *, repeats: int, warmup: int
) -> tuple[list[float], int]:
    """Run ``run_one(i)`` ``warmup + repeats`` times, returning every iteration's
    wall time (ms) and the last iteration's item count. The warmup samples are
    kept here and discarded by :func:`~tests.manual.perf.stats.summarize`, so the
    raw run is fully recorded. This is the profiler-attach seam (#866)."""
    samples: list[float] = []
    items = 0
    for i in range(warmup + repeats):
        start = time.perf_counter()
        items = await run_one(i)
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples, items


async def measure(
    workload: Workload, booted: Booted, *, repeats: int, warmup: int
) -> WorkloadResult:
    """Run a workload's setup, then time it, then summarize into a result."""
    await workload.setup(booted)
    samples, items = await time_iterations(
        lambda i: workload.run_one(booted, i), repeats=repeats, warmup=warmup
    )
    return WorkloadResult(
        workload=workload.name,
        distribution=summarize(samples, warmup=warmup),
        items=items,
    )


def run_async(coro: Awaitable[Any]) -> Any:
    """Drive an async run from the sync entry point."""
    return asyncio.run(coro)  # type: ignore[arg-type]
