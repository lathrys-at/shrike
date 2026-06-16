# NOTE (#278 cutover): the Python-engine internals these tests used to pin —
# numpy pooling math, pad-token resolution, feed dtype casting/filtering —
# retired with the Python engine (they run crate-side now, pinned by the
# integration model tests). What remains here is the facade's own behaviour.
"""Tests for the shrike.embedding_onnx OnnxBackend facade (fake native engine).

File resolution, fingerprint behaviour, chunking/batch-cap policy, provider
resolution, and the failure modes are covered without a real ONNX model — the
onnxruntime wheel and the shrike_native engine are faked. A real end-to-end
embed against actual models lives in the integration suite.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from shrike.embed_batching import BATCH_PROBE_TEXTS
from shrike.embed_text import EMBED_TEXT_VERSION
from shrike.embedding_base import TEXT
from shrike.embedding_onnx import OnnxBackend

# -- Fakes for onnxruntime (the carrier wheel) / shrike_native (the engine) ----


class _FakeEngine:
    """All-ones engine: 4-dim unit-normalized vectors, batch-safe.

    Records the batch size of each embed_chunk call so tests can assert the
    chunking policy. Subclasses tweak the variance/failure behaviour.
    """

    _unsupported: tuple[str, ...] = ()

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.run_calls: list[int] = []  # batch size of each embed_chunk call
        self._providers = list(kwargs.get("providers") or ["CPUExecutionProvider"])

    def active_providers(self) -> list[str]:
        return self._providers

    def unsupported_inputs(self) -> list[str]:
        return list(self._unsupported)

    def dim(self) -> int | None:
        return 4

    def embed_chunk(self, texts: list[str]) -> list[list[float]]:
        self.run_calls.append(len(texts))
        return [[0.5, 0.5, 0.5, 0.5] for _ in texts]


class _FakeEngineVariant(_FakeEngine):
    """Batch-variant (like int8 dynamic quant): output *direction* depends on batch
    size, so the startup probe detects it and forces serial embedding."""

    def embed_chunk(self, texts: list[str]) -> list[list[float]]:
        self.run_calls.append(len(texts))
        return [[float(len(texts)), 0.5, 0.5, 0.5] for _ in texts]


class _FakeEngineRequiresPositionIds(_FakeEngine):
    """A model with a *required* input the backend doesn't supply: every embed
    raises (mimicking onnxruntime's "Required inputs ... are missing"), and the
    engine names the culprit via unsupported_inputs()."""

    _unsupported = ("position_ids",)

    def embed_chunk(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Required inputs (position_ids) are missing from input feed.")


class _FakeEngineOptionalExtra(_FakeEngine):
    """Declares an extra input the backend doesn't supply but the graph doesn't
    *require* — embeds succeed, so start() must NOT reject the model."""

    _unsupported = ("extra_optional",)


class _FakeEngineDropsProvider(_FakeEngine):
    """Mimics a provider that's available but fails to initialise: the engine
    reports only CPU loaded (what ort reports when an accelerator can't load)."""

    def active_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]


def _model_dir(tmp_path: Path, *, content: bytes = b"onnx-bytes") -> Path:
    """A minimal HF-style ONNX model dir (model.onnx + tokenizer.json)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "model.onnx").write_bytes(content)
    (tmp_path / "tokenizer.json").write_text("{}")
    return tmp_path


# -- Construction -------------------------------------------------------------


class TestConstruction:
    def test_default_pooling_is_mean(self) -> None:
        assert OnnxBackend(model="x")._pooling == "mean"

    def test_rejects_none_pooling(self) -> None:
        with pytest.raises(ValueError, match="pooling"):
            OnnxBackend(model="x", pooling="none")

    def test_modalities_text_only(self) -> None:
        assert OnnxBackend(model="x").modalities == frozenset({TEXT})

    def test_default_provider_is_cpu(self) -> None:
        assert OnnxBackend(model="x")._providers == ["CPUExecutionProvider"]

    def test_max_length_default_and_override(self) -> None:
        # None (e.g. no --embedding-context-size) → the 256 default; else honored.
        assert OnnxBackend(model="x")._max_length == 256
        assert OnnxBackend(model="x", max_length=None)._max_length == 256
        assert OnnxBackend(model="x", max_length=512)._max_length == 512


# -- File resolution ----------------------------------------------------------


class TestResolveFiles:
    def test_dir_with_root_files(self, tmp_path: Path) -> None:
        _model_dir(tmp_path)
        onnx_path, tok_path = OnnxBackend(model=str(tmp_path))._resolve_files()
        assert onnx_path.name == "model.onnx"
        assert tok_path.name == "tokenizer.json"

    def test_dir_with_onnx_subdir(self, tmp_path: Path) -> None:
        (tmp_path / "onnx").mkdir()
        (tmp_path / "onnx" / "model.onnx").write_bytes(b"x")
        (tmp_path / "onnx" / "tokenizer.json").write_text("{}")
        onnx_path, tok_path = OnnxBackend(model=str(tmp_path))._resolve_files()
        assert onnx_path.parent.name == "onnx"
        assert tok_path.parent.name == "onnx"

    def test_direct_onnx_file(self, tmp_path: Path) -> None:
        _model_dir(tmp_path)
        be = OnnxBackend(model=str(tmp_path / "model.onnx"))
        onnx_path, tok_path = be._resolve_files()
        assert onnx_path.name == "model.onnx"
        assert tok_path.name == "tokenizer.json"

    def test_missing_tokenizer_raises(self, tmp_path: Path) -> None:
        (tmp_path / "model.onnx").write_bytes(b"x")
        with pytest.raises(FileNotFoundError, match="tokenizer"):
            OnnxBackend(model=str(tmp_path))._resolve_files()

    def test_missing_model_raises(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="model.onnx"):
            OnnxBackend(model=str(tmp_path))._resolve_files()

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            OnnxBackend(model=str(tmp_path / "nope"))._resolve_files()

    def test_resolves_quantized_variant_when_no_plain_model(self, tmp_path: Path) -> None:
        # A quant-only export (#667): only model_quantized.onnx present, no plain
        # model.onnx — the variant-suffix resolver finds it (no fetch-time rename).
        (tmp_path / "model_quantized.onnx").write_bytes(b"x")
        (tmp_path / "tokenizer.json").write_text("{}")
        onnx_path, tok_path = OnnxBackend(model=str(tmp_path))._resolve_files()
        assert onnx_path.name == "model_quantized.onnx"
        assert tok_path.name == "tokenizer.json"

    def test_plain_model_wins_over_variant(self, tmp_path: Path) -> None:
        # Full precision is preferred: model.onnx (the "" suffix, tried first) wins
        # over a quantized sibling. So the renamed-to-model.onnx fixtures are unchanged.
        (tmp_path / "model.onnx").write_bytes(b"x")
        (tmp_path / "model_quantized.onnx").write_bytes(b"x")
        (tmp_path / "tokenizer.json").write_text("{}")
        onnx_path, _ = OnnxBackend(model=str(tmp_path))._resolve_files()
        assert onnx_path.name == "model.onnx"

    def test_resolves_variant_under_onnx_subdir(self, tmp_path: Path) -> None:
        (tmp_path / "onnx").mkdir()
        (tmp_path / "onnx" / "model_fp16.onnx").write_bytes(b"x")
        (tmp_path / "onnx" / "tokenizer.json").write_text("{}")
        onnx_path, _ = OnnxBackend(model=str(tmp_path))._resolve_files()
        assert onnx_path.name == "model_fp16.onnx"
        assert onnx_path.parent.name == "onnx"

    def test_external_data_companion_is_left_on_disk(self, tmp_path: Path) -> None:
        # The external-data invariant (#667): _resolve_files returns the .onnx graph;
        # the sibling .onnx_data is NOT named in code (onnxruntime loads it relative
        # to the graph dir). The resolver must not trip over the extra file.
        (tmp_path / "model_quantized.onnx").write_bytes(b"x")
        (tmp_path / "model_quantized.onnx_data").write_bytes(b"weights")
        (tmp_path / "tokenizer.json").write_text("{}")
        onnx_path, _ = OnnxBackend(model=str(tmp_path))._resolve_files()
        assert onnx_path.name == "model_quantized.onnx"
        # The companion is co-located (what makes onnxruntime's external-data load work).
        assert (onnx_path.parent / "model_quantized.onnx_data").is_file()

    def test_only_external_data_no_graph_stub_raises(self, tmp_path: Path) -> None:
        # The external-data landmine (#667, joint-review ADV-2): a HALF-materialized
        # dir holding ONLY the `.onnx_data` companion (a forgotten graph-stub data
        # dep) must raise FileNotFoundError — the resolver must NEVER mistake the
        # `.onnx_data` file for a graph (no variant suffix produces a `.onnx_data`
        # name, so the suffix loop finds nothing and the dir is correctly rejected).
        (tmp_path / "model_quantized.onnx_data").write_bytes(b"weights")
        (tmp_path / "tokenizer.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="model"):
            OnnxBackend(model=str(tmp_path))._resolve_files()


# -- Fingerprint --------------------------------------------------------------


class TestFingerprint:
    def test_format_and_namespace(self, tmp_path: Path) -> None:
        _model_dir(tmp_path, content=b"hello")  # 5 bytes
        fp = OnnxBackend(model=str(tmp_path), pooling="mean").model_fingerprint()
        # onnx-rs: is the native engine's vector-space identity, kept verbatim
        # from the dual-engine bake (#270) — indexes built then load unchanged.
        assert fp == f"onnx-rs:model.onnx:5:pool=mean:textprep={EMBED_TEXT_VERSION}"
        # Namespaced distinctly from a llama fingerprint (meta:/file:).
        assert fp.split(":")[0] == "onnx-rs"

    def test_pooling_changes_fingerprint(self, tmp_path: Path) -> None:
        _model_dir(tmp_path)
        mean_fp = OnnxBackend(model=str(tmp_path), pooling="mean").model_fingerprint()
        cls_fp = OnnxBackend(model=str(tmp_path), pooling="cls").model_fingerprint()
        assert mean_fp != cls_fp

    def test_content_change_changes_fingerprint(self, tmp_path: Path) -> None:
        d1 = _model_dir(tmp_path / "a", content=b"aa")
        d2 = _model_dir(tmp_path / "b", content=b"aaaaaa")
        assert (
            OnnxBackend(model=str(d1)).model_fingerprint()
            != OnnxBackend(model=str(d2)).model_fingerprint()
        )

    def test_normalize_not_in_fingerprint(self, tmp_path: Path) -> None:
        # Normalization is scale-only (cos is scale-invariant), so it must NOT
        # change the fingerprint — same reasoning as llama's --embd-normalize.
        _model_dir(tmp_path)
        on = OnnxBackend(model=str(tmp_path), normalize=True).model_fingerprint()
        off = OnnxBackend(model=str(tmp_path), normalize=False).model_fingerprint()
        assert on == off


# -- Lifecycle + embed (fake engine) ------------------------------------------


class TestLifecycleAndEmbed:
    def _start(
        self,
        be: OnnxBackend,
        engine_cls: type = _FakeEngine,
        available_providers: list[str] | None = None,
    ) -> None:
        # Inject a fake onnxruntime (provider discovery only — the wheel is the
        # runtime carrier) and a fake shrike_native engine into sys.modules, so
        # these tests run without the optional 'onnx' extra or a built extension.
        fake_ort = types.ModuleType("onnxruntime")
        _avail = available_providers or ["CPUExecutionProvider"]
        fake_ort.get_available_providers = lambda: list(_avail)  # type: ignore[attr-defined]
        fake_native = types.ModuleType("shrike_native")
        fake_native.init_onnx_runtime = lambda _path: None  # type: ignore[attr-defined]
        fake_native.OnnxTextEmbedder = engine_cls  # type: ignore[attr-defined]
        with (
            patch.dict(sys.modules, {"onnxruntime": fake_ort, "shrike_native": fake_native}),
            patch("shrike.embedding_onnx.locate_ort_dylib", lambda: Path("/fake/libort.so")),
        ):
            be.start()

    def test_running_toggles(self, tmp_path: Path) -> None:
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        assert be.running is False
        self._start(be)
        assert be.running is True
        be.stop()
        assert be.running is False

    def test_embed_shape(self, tmp_path: Path) -> None:
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        self._start(be)
        vecs = be.embed_texts(["a", "b"])
        assert len(vecs) == 2
        assert all(len(v) == 4 for v in vecs)

    def test_embeds_serially_when_variant(self, tmp_path: Path) -> None:
        # A batch-variant model (the probe detects it) is embedded serially: one run per
        # text, each batch-of-1, so a note's vector can't depend on its batch-mates.
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        self._start(be, _FakeEngineVariant)
        assert be._safe_batch == 1
        be._native_engine.run_calls.clear()  # drop the startup-probe calls
        be.embed_texts(["a", "b", "c"])
        assert be._native_engine.run_calls == [1, 1, 1]

    def test_embeds_batched_when_safe(self, tmp_path: Path) -> None:
        # A batch-safe model (the all-ones fake) embeds the whole input in one chunk.
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        self._start(be)
        assert be._safe_batch >= 2
        be._native_engine.run_calls.clear()
        be.embed_texts(["a", "b", "c"])
        assert be._native_engine.run_calls == [3]

    def test_batch_size_cap_chunks(self, tmp_path: Path) -> None:
        # --embedding-batch-size caps the chunk size even on a batch-safe model.
        be = OnnxBackend(model=str(_model_dir(tmp_path)), batch_size=2)
        self._start(be)
        be._native_engine.run_calls.clear()
        be.embed_texts(["a", "b", "c", "d", "e"])
        assert be._native_engine.run_calls == [2, 2, 1]

    def test_variant_model_warns_when_batch_requested(self, tmp_path: Path) -> None:
        be = OnnxBackend(model=str(_model_dir(tmp_path)), batch_size=8)
        with patch("shrike.embedding_onnx.logger.warning") as warn:
            self._start(be, _FakeEngineVariant)
        assert be._safe_batch == 1
        assert any("batch-variant" in str(c.args) for c in warn.call_args_list)

    def test_cap_above_ceiling_clamps_to_probe_size(self, tmp_path: Path) -> None:
        # The probe-set size is the batch ceiling; a cap *above* it is logged once and clamped.
        be = OnnxBackend(model=str(_model_dir(tmp_path)), batch_size=len(BATCH_PROBE_TEXTS) * 2)
        with patch("shrike.embedding_onnx.logger.info") as info:
            self._start(be)
        assert be._safe_batch == len(BATCH_PROBE_TEXTS)
        assert any("exceeds the probe-verified ceiling" in str(c.args) for c in info.call_args_list)
        be._native_engine.run_calls.clear()
        be.embed_texts([f"n{i}" for i in range(be._safe_batch + 8)])
        assert be._native_engine.run_calls == [be._safe_batch, 8]

    def test_required_unsupported_input_fails_loud(self, tmp_path: Path) -> None:
        # A model with a required input we don't supply (position_ids) must fail start() with
        # a named error, not boot fine and silently break embedding on the first real call.
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        with pytest.raises(RuntimeError, match="position_ids"):
            self._start(be, _FakeEngineRequiresPositionIds)

    def test_optional_unsupported_input_does_not_reject(self, tmp_path: Path) -> None:
        # An extra declared input the graph doesn't *require* (embeds succeed without it)
        # must not be rejected — the guard keys on an actual embed failure, not a name.
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        self._start(be, _FakeEngineOptionalExtra)  # must not raise
        assert be.embed_texts(["a", "b"])

    def test_health_batch_label_reflects_cap(self, tmp_path: Path) -> None:
        # Safe model, no cap → batched.
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        self._start(be)
        assert be.health()["batch_safe"] is True
        assert be.health()["batch"] == "batched"
        # Safe model, cap=1 → the model is still capable (batch_safe), but it embeds serially.
        capped = OnnxBackend(model=str(_model_dir(tmp_path)), batch_size=1)
        self._start(capped)
        assert capped.health()["batch_safe"] is True
        assert capped.health()["batch"] == "serial"
        # Variant model → both serial.
        var = OnnxBackend(model=str(_model_dir(tmp_path)))
        self._start(var, _FakeEngineVariant)
        assert var.health()["batch_safe"] is False
        assert var.health()["batch"] == "serial"

    def test_embed_empty_returns_empty(self, tmp_path: Path) -> None:
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        self._start(be)
        assert be.embed_texts([]) == []

    def test_embed_raises_when_not_running(self) -> None:
        with pytest.raises(RuntimeError, match="not running"):
            OnnxBackend(model="x").embed_texts(["a"])

    def test_embedding_dim_from_engine(self, tmp_path: Path) -> None:
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        self._start(be)
        assert be.embedding_dim() == 4

    def test_embedding_dim_none_when_stopped(self) -> None:
        assert OnnxBackend(model="x").embedding_dim() is None

    def test_start_missing_dependency_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import → ImportError
        with pytest.raises(ImportError, match="pip install onnxruntime"):
            be.start()

    # -- Health -------------------------------------------------------------------

    def test_unavailable_provider_falls_back_to_cpu_with_warning(self, tmp_path: Path) -> None:
        be = OnnxBackend(model=str(_model_dir(tmp_path)), providers=["CUDAExecutionProvider"])
        with patch("shrike.embedding_onnx.logger.warning") as warn:
            self._start(be, available_providers=["CPUExecutionProvider"])
        assert be.health()["active_providers"] == ["CPUExecutionProvider"]
        assert be.health()["provider"] == "CPUExecutionProvider"
        assert any("not available" in str(c.args) for c in warn.call_args_list)

    def test_available_provider_is_resolved_and_surfaced(self, tmp_path: Path) -> None:
        be = OnnxBackend(model=str(_model_dir(tmp_path)), providers=["CoreMLExecutionProvider"])
        with patch("shrike.embedding_onnx.logger.warning") as warn:
            self._start(be, available_providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
        h = be.health()
        assert h["requested_providers"] == ["CoreMLExecutionProvider"]
        # CPU is appended as the final fallback; the requested provider stays first.
        assert h["active_providers"] == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        assert h["provider"] == "CoreMLExecutionProvider"
        assert not any("not available" in str(c.args) for c in warn.call_args_list)

    def test_available_but_unloaded_provider_warns(self, tmp_path: Path) -> None:
        # Requested + available, but the engine reports it didn't load (init failed) → warn,
        # and health shows the CPU reality, not the request.
        be = OnnxBackend(model=str(_model_dir(tmp_path)), providers=["CUDAExecutionProvider"])
        with patch("shrike.embedding_onnx.logger.warning") as warn:
            self._start(
                be,
                _FakeEngineDropsProvider,
                available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        assert be.health()["provider"] == "CPUExecutionProvider"
        assert any("did not load" in str(c.args) for c in warn.call_args_list)


class TestHealth:
    def test_health_when_stopped(self) -> None:
        h = OnnxBackend(model="x").health()
        assert h["available"] is False
        assert h["backend"] == "onnx"
        assert h["requested_providers"] == ["CPUExecutionProvider"]
        assert h["active_providers"] == []  # nothing loaded until start()
        assert h["provider"] is None
