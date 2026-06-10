"""Backend case registry for the conformance + parity harness (#268).

Each :class:`BackendCase` describes one ``EmbedderBackend`` configuration runnable
through ``test_backend_conformance.py``. **Registering a new backend implementation
is one entry here** — that's the point: when a native (Rust) implementation of the
same protocol lands (#270/#271), its acceptance gate is this suite plus a
``parity_ref`` pointing at the Python implementation it replaces.

Parity semantics (epic #265 convention 7): a case whose ``parity_ref`` is ``None``
is compared against *a fresh instance of itself* (restart parity — fingerprint
stability and vector reproducibility). A case with a ``parity_ref`` is compared
against that reference implementation: byte-equal vectors **and** an identical
fingerprint mean the new runtime may keep the reference's fingerprint namespace;
anything less must namespace itself (never a tolerance-only match).

``*_exact`` flags encode each runtime's determinism claim on CPU (CI runs CPU
providers): the int8/fp32 ONNX paths are bit-exact; llama-server may sit on Metal
locally, so its claims are tolerance-tier (see docs/decisions.md, "Bit-exact is a
CPU property").
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from shrike.embedding_base import EmbedderBackend

# A small, deliberately varied corpus for parity comparisons: plain prose, unicode,
# symbols/numbers, a long entry, and an empty-ish one — the shapes that shake out
# tokenizer and pooling differences between runtimes.
PARITY_TEXTS = [
    "What is a mitochondrion? An organelle that produces ATP",
    "d/dx[f(g(x))] = f'(g(x)) * g'(x)",
    "Die Hauptstadt von Österreich ist Wien — größer als Graz.",
    "脱氧核糖核酸 (DNA) carries genetic information",
    "momentum " * 60,
    "x",
]


@dataclass(frozen=True)
class BackendCase:
    """One backend configuration to run through the conformance + parity suite."""

    id: str
    ndim: int
    # Acceptable model_fingerprint prefixes (a namespace, e.g. ("onnx:",)).
    fingerprint_prefixes: tuple[str, ...]
    # Build a *fresh, unstarted* backend. Takes the pytest request so it can pull
    # the CI-cached model fixtures (session-scoped) lazily.
    make: Callable[[pytest.FixtureRequest], EmbedderBackend]
    # Vectors byte-equal across two instances of the same config (CPU determinism).
    restart_exact: bool
    # When the probe finds the model batch-safe, batched == serial byte-equal.
    batch_exact: bool
    # Build the *reference* implementation for cross-runtime parity, or None for
    # self (restart) parity. A native backend (#270/#271) registers the Python
    # implementation it replaces here.
    parity_ref: Callable[[pytest.FixtureRequest], EmbedderBackend] | None = None
    # True when the case claims the reference's fingerprint namespace — then the
    # parity test requires byte-equal vectors AND an identical fingerprint.
    claims_reference_namespace: bool = False
    marks: tuple[Any, ...] = field(default=())


def _make_onnx(model_fixture: str) -> Callable[[pytest.FixtureRequest], EmbedderBackend]:
    def make(request: pytest.FixtureRequest) -> EmbedderBackend:
        from shrike.embedding_onnx import OnnxBackend

        model: Path = request.getfixturevalue(model_fixture)
        return OnnxBackend(model=str(model))

    return make


def _make_clip(request: pytest.FixtureRequest) -> EmbedderBackend:
    from shrike.embedding_clip import ClipBackend

    model: Path = request.getfixturevalue("clip_model")
    return ClipBackend(model=str(model))


def _make_llama(request: pytest.FixtureRequest) -> EmbedderBackend:
    from shrike.embedding import LlamaServerBackend
    from tests.integration.conftest import _free_port

    model: Path = request.getfixturevalue("embedding_model")
    workdir = Path(tempfile.mkdtemp(prefix="shrike-conformance-llama-"))
    return LlamaServerBackend(
        model=str(model),
        port=_free_port(),
        log_dir=workdir,
        pid_file=workdir / "embedding.pid",
    )


def cases() -> list[BackendCase]:
    # Markers are imported lazily so importing this module never requires the
    # optional extras themselves.
    from tests.integration.conftest import (
        requires_clip,
        requires_llama_server,
        requires_onnxruntime,
    )

    return [
        BackendCase(
            id="onnx-minilm-int8",
            ndim=384,
            fingerprint_prefixes=("onnx:",),
            make=_make_onnx("onnx_model"),
            restart_exact=True,
            batch_exact=True,
            marks=(requires_onnxruntime,),
        ),
        BackendCase(
            id="onnx-minilm-fp32",
            ndim=384,
            fingerprint_prefixes=("onnx:",),
            make=_make_onnx("onnx_fp32_model"),
            restart_exact=True,
            batch_exact=True,
            marks=(requires_onnxruntime,),
        ),
        BackendCase(
            id="onnx-distilroberta-int8",
            ndim=768,
            fingerprint_prefixes=("onnx:",),
            make=_make_onnx("distilroberta_model"),
            restart_exact=True,
            batch_exact=True,
            marks=(requires_onnxruntime,),
        ),
        BackendCase(
            id="clip-vit-b32",
            ndim=512,
            fingerprint_prefixes=("clip:",),
            make=_make_clip,
            restart_exact=True,
            batch_exact=True,
            marks=(requires_clip,),
        ),
        BackendCase(
            id="llama-minilm-gguf",
            ndim=384,
            fingerprint_prefixes=("meta:", "file:"),
            make=_make_llama,
            # llama-server may run on Metal locally; restart/batch claims are
            # tolerance-tier, not byte-equality (CPU-only CI could tighten this).
            restart_exact=False,
            batch_exact=False,
            marks=(requires_llama_server,),
        ),
    ]


def conformance_params() -> list[Any]:
    """The pytest params for the conformance fixture — one per registered case."""
    return [pytest.param(case, id=case.id, marks=case.marks) for case in cases()]
