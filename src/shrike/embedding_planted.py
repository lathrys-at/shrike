"""A deterministic, test-only "planted-vector" embedding backend (#559).

The adversarial search-quality suite needs **exactly controllable cosines** so
it can assert the *exact* RRF-fused order the kernel produces from its known
constants (``RRF_K=60``, the per-signal weights, the exact-override tier) — a
real model's cosines wobble (int8 quant, float drift) and can't pin a golden
rank. This backend maps a text (or an image's bytes) to a vector by exact
lookup in a *plan* JSON the test writes, falling back to a stable token-hash
for anything not planted.

It is wired so the kernel exercises the **real** ``search_notes`` path: it does
NOT expose ``native_embedder()``, so ``Harness._attach`` captures it behind
``PyEmbedder.capture`` (harness.py) and every embed rides the kernel's blocking
pool exactly like a custom backend. The whole RRF / per-modality / activation-
gate / exact-override pipeline runs end-to-end against vectors the test placed,
with no model download.

OFF-PRODUCTION GATE (load-bearing). This kind is *never* a public
``--embedding-backend`` choice and is *never* constructed unless
``SHRIKE_SEARCH_QUALITY=1`` is set in the environment (the same env that gates
the suite away from CI). ``EmbeddingRuntime._make_backend`` refuses to build it
otherwise, so a stray ``planted`` in a config on a production box is an error,
not a silent test-vector backend.

Plan schema (``SHRIKE_PLANTED_VECTORS`` → a JSON file)::

    { "dim": 8,
      "fingerprint": "planted:case-a:v1",   // folded into the index model_id
      "texts":  { "the mitochondria is the powerhouse": [1, 0, 0, ...] },
      "images": { "<sha256-hex of the image bytes>":     [0, 1, 0, ...] } }

Vectors are L2-normalized on load (USearch's ``cos`` is scale-invariant, but
normalizing keeps the planted cosines literally the dot products the test
reasons about). An un-planted text/image falls back to a deterministic
token/byte hash in the same dimension, so a distractor card still gets a
*stable* vector — it just isn't hand-placed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from shrike.embedding_base import IMAGE, TEXT

#: The env var pointing at the plan JSON. Read at construction; absent → the
#: backend is pure fallback-hash (still deterministic, just nothing hand-placed).
PLAN_ENV = "SHRIKE_PLANTED_VECTORS"

#: The suite's off-CI gate — the planted backend only ever constructs when this
#: is set (mirrors ``requires_search_quality`` / ``SHRIKE_SEARCH_QUALITY``).
GATE_ENV = "SHRIKE_SEARCH_QUALITY"

_DEFAULT_DIM = 16


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class PlantedBackend:
    """A deterministic text+image embedder driven by a planted-vector plan.

    Implements the ``EmbedderBackend`` protocol surface the index/harness need
    (``embed_texts`` / ``embed_images`` / ``modalities`` / lifecycle /
    ``model_fingerprint`` / ``embedding_dim`` / ``health``). It is intentionally
    *captured* — no ``native_embedder()`` — so the kernel drives it through the
    same Python dispatch a custom backend uses, and the full fusion pipeline is
    exercised against the test's vectors.
    """

    def __init__(self, plan_path: str | os.PathLike[str] | None = None) -> None:
        self._running = False
        self._plan_path = (
            Path(plan_path) if plan_path is not None else _plan_path_from_env()
        )
        self._dim = _DEFAULT_DIM
        self._fingerprint = "planted:fallback:v1"
        self._texts: dict[str, list[float]] = {}
        self._images: dict[str, list[float]] = {}
        if self._plan_path is not None:
            self._load_plan(self._plan_path)

    # -- plan loading --------------------------------------------------------

    def _load_plan(self, path: Path) -> None:
        plan: dict[str, Any] = json.loads(path.read_text())
        self._dim = int(plan.get("dim", _DEFAULT_DIM))
        self._fingerprint = str(plan.get("fingerprint", f"planted:{path.stem}:v1"))
        self._texts = {
            text: _l2(self._fit(vec)) for text, vec in plan.get("texts", {}).items()
        }
        # Image keys are the sha256 hex of the image's raw bytes — the only
        # stable handle the captured backend gets (it receives list[bytes]).
        self._images = {
            key: _l2(self._fit(vec)) for key, vec in plan.get("images", {}).items()
        }

    def _fit(self, vec: list[float]) -> list[float]:
        """Pad/truncate a planted vector to the plan's dim (forgiving authoring)."""
        out = [float(x) for x in vec[: self._dim]]
        out.extend([0.0] * (self._dim - len(out)))
        return out

    # -- deterministic fallback ----------------------------------------------

    def _hash_vec(self, payload: bytes) -> list[float]:
        """A stable token/byte-hash vector for an un-planted input.

        Distractors that aren't hand-placed still get a deterministic vector so
        the corpus is reproducible run to run; it just isn't a controlled point
        in the space.
        """
        vec = [0.0] * self._dim
        # Hash whole tokens for text (so token overlap → similarity), whole
        # payload for bytes. Either way: deterministic and in-dim.
        for token in payload.split():
            digest = hashlib.blake2b(token, digest_size=2).hexdigest()
            vec[int(digest, 16) % self._dim] += 1.0
        if not any(vec):  # no whitespace tokens (e.g. raw image bytes)
            digest = hashlib.blake2b(payload, digest_size=2).hexdigest()
            vec[int(digest, 16) % self._dim] = 1.0
        return _l2(vec)

    # -- EmbedderBackend protocol --------------------------------------------

    @property
    def modalities(self) -> frozenset[str]:
        return frozenset({TEXT, IMAGE})

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            planted = self._texts.get(text)
            out.append(planted if planted is not None else self._hash_vec(text.encode("utf-8")))
        return out

    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        out: list[list[float]] = []
        for raw in images:
            key = hashlib.sha256(raw).hexdigest()
            planted = self._images.get(key)
            out.append(planted if planted is not None else self._hash_vec(raw))
        return out

    def embedding_dim(self) -> int:
        return self._dim

    def model_fingerprint(self) -> str:
        return self._fingerprint

    def health(self) -> dict[str, Any]:
        return {
            "available": self._running,
            "backend": "planted",
            "dim": self._dim,
            "fingerprint": self._fingerprint,
            "planted_texts": len(self._texts),
            "planted_images": len(self._images),
        }


def _plan_path_from_env() -> Path | None:
    raw = os.environ.get(PLAN_ENV)
    if not raw:
        return None
    return Path(raw).expanduser()


def gate_open() -> bool:
    """True only when the search-quality gate env is set — the single guard the
    runtime checks before ever constructing a planted backend."""
    return os.environ.get(GATE_ENV) == "1"
