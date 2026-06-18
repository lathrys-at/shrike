"""Manual, local-only end-to-end image-embedding harness.

This is the proof that the image path works end to end against a real
multimodal embedding model: a `modalities: [text, image]` managed
``LlamaServerBackend`` embeds a note's text AND its images, and distinct
images produce distinct vectors (the github.com/ggml-org/llama.cpp#13666
failure mode, where every image collapsed to the same vector).

WHY IT'S NOT IN CI
------------------
The only small multimodal embedding model (jina-embeddings-v5-omni) relies on
llama.cpp patches that are NOT upstream as of b9616 (audio chunked attention,
qwen3vl temporal-pair, encoder combined-decode). The official/pinned
llama-server loads the model and serves *text* embeddings, then SEGFAULTS
during image-embedding extraction — verified on b9415 (pinned) and b9616
(latest). CI pins official release binaries, so it cannot run this until the
patches land upstream. Hence: env-gated, skipped everywhere the patched
binary + model aren't present.

A second constraint, found empirically: the text GGUF must be the **F16**
(unquantized) variant. The fork's encoder-combined-decode path reads the
token-embedding tensor element-by-element with ``ggml_get_f32_1d``, which
aborts on a block-quantized type. So a K-quant whose ``token_embd.weight`` is
quantized crashes the server on the first image embed — verified on Q4_K_M
*and* Q5_K_M (the quant jina's GGUF card recommends), both dying identically
at ``server-common.cpp`` / ``ggml-cpu.c:1049``, with or without ``--pooling
last``. F16 keeps the table readable; that's what the fixture pins.

Pooling: jina-v5-omni is a last-token model, so the harness passes
``--pooling last`` (its pooling isn't in the GGUF metadata; the default mean
would give wrong embeddings).

HOW TO RUN IT
-------------
1. Build the patched llama-server::

       git clone --branch feat-v5-omni https://github.com/jina-ai/llama.cpp.git
       cd llama.cpp && cmake -B build && cmake --build build --config Release -j
       # the binary lands at build/bin/llama-server
       # (Hopper GPUs need GGML_CUDA_DISABLE_GRAPHS=1; CPU/Metal/Vulkan are fine)

2. Download a small multimodal embedding model + its vision projector, e.g.
   jinaai/jina-embeddings-v5-omni-nano-classification-GGUF: the
   ``*-F16.gguf`` (text — must be unquantized, see above) and
   ``*-vision-mmproj-F16.gguf`` files. The fixture fetches both on demand; to
   pre-seed the shared test-model cache (no re-spelled URLs), run::

       python -c "from tests.integration.model_cache import \
           cached_multimodal_model_dir, default_model_cache_base; \
         print(cached_multimodal_model_dir(default_model_cache_base()))"

3. Point the harness at them and run::

       export SHRIKE_MULTIMODAL_LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server
       export SHRIKE_MULTIMODAL_MODEL=/path/to/...-Q4_K_M.gguf
       export SHRIKE_MULTIMODAL_VISION_MMPROJ=/path/to/...-vision-mmproj-F16.gguf
       pytest tests/integration/test_multimodal.py -v -m multimodal

When a multimodal embedding model becomes a server default and the patches are
upstream, this graduates into a pinned-fixture CI test.
"""

from __future__ import annotations

import io
import math

import pytest

from tests.integration.conftest import _free_port, requires_multimodal

pytestmark = [pytest.mark.integration, pytest.mark.multimodal, requires_multimodal]

PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from shrike.harness.engines.embedding.base import IMAGE, TEXT  # noqa: E402
from shrike.harness.engines.embedding.runtime import LlamaServerBackend  # noqa: E402


def _png(color: tuple[int, int, int], size: int = 224) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


@pytest.fixture()
def omni_backend(multimodal_paths, tmp_path):
    """A started managed multimodal LlamaServerBackend (text + image), torn
    down after the test. Skipped unless the patched binary + fixture exist."""
    be = LlamaServerBackend(
        model=multimodal_paths["model"],
        llama_server=multimodal_paths["llama_server"],
        mmprojs=[multimodal_paths["vision_mmproj"]],
        modalities=frozenset({TEXT, IMAGE}),
        # jina-v5-omni is a last-token model: its pooling isn't in the GGUF
        # metadata, so without this llama-server defaults to mean and produces
        # WRONG embeddings (jina's own docs require `--pooling last`).
        pooling="last",
        port=_free_port(),
        log_dir=tmp_path / "logs",
        # An image is many vision tokens; give the server room (embeddings need
        # the whole sequence in one ubatch).
        context_size=8192,
    )
    be.start()
    try:
        yield be
    finally:
        be.stop()


def test_advertises_image_and_vision_is_loaded(omni_backend: LlamaServerBackend) -> None:
    assert IMAGE in omni_backend.modalities
    health = omni_backend.health()
    assert health["available"] is True


def test_text_and_image_both_embed_with_consistent_dim(
    omni_backend: LlamaServerBackend,
) -> None:
    text_vecs = omni_backend.embed_texts(["a red square", "the number forty-two"])
    image_vecs = omni_backend.embed_images([_png((220, 20, 20)), _png((20, 20, 220))])
    assert len(text_vecs) == 2
    assert len(image_vecs) == 2
    dims = {len(v) for v in (*text_vecs, *image_vecs)}
    assert len(dims) == 1, f"text + image vectors must share one space, got dims {dims}"
    assert dims.pop() > 0


def test_distinct_images_give_distinct_vectors(omni_backend: LlamaServerBackend) -> None:
    # The #13666 regression: a broken multimodal path collapses every image to
    # the same vector. Two visibly different images must NOT be near-identical.
    red, blue = omni_backend.embed_images([_png((220, 20, 20)), _png((20, 20, 220))])
    assert _cosine(red, blue) < 0.999, "distinct images collapsed to the same vector"


def test_same_image_is_deterministic(omni_backend: LlamaServerBackend) -> None:
    once = omni_backend.embed_images([_png((10, 140, 60))])[0]
    twice = omni_backend.embed_images([_png((10, 140, 60))])[0]
    assert _cosine(once, twice) > 0.999, "the same image must embed deterministically"
