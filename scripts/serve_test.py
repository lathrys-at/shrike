"""Unit tests for the //scripts:serve_<profile> launcher's pure logic.

Covers profile resolution, the path-free invariant, arg parsing, the
effective-config composition (dir-name → absolute path rewrite), the run-server
argv, and the --import MVP stub — all WITHOUT any model download, so this is the
non-manual target CI runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import serve
import yaml

# -- profile loading + the path-free invariant ---------------------------------


def test_load_text_onnx_profile_is_path_free() -> None:
    profile = serve.load_profile("text-onnx")
    assert "collection" not in profile
    embedders = profile["embedders"]
    assert len(embedders) == 1
    entry = embedders[0]
    assert entry["runtime"] == "onnx"
    assert entry["modalities"] == ["text"]
    # The model is a bare DIR-NAME, never an absolute path.
    assert entry["model"] == "all-MiniLM-L6-v2-onnx-int8"
    assert not Path(entry["model"]).is_absolute()


def test_load_onnx_multispace_profile_is_path_free() -> None:
    # The pure-ONNX multi-space profile: embeddinggemma text + MobileCLIP2
    # image, both onnx, path-free, distinct image leg.
    profile = serve.load_profile("onnx-multispace")
    assert "collection" not in profile
    embedders = profile["embedders"]
    assert len(embedders) == 2
    # The text leg.
    text = embedders[0]
    assert text["runtime"] == "onnx"
    assert text["modalities"] == ["text"]
    assert text["model"] == "embeddinggemma-300m-onnx-int8"
    assert not Path(text["model"]).is_absolute()
    # The image leg (text + image into one shared space; one image space).
    image = embedders[1]
    assert image["runtime"] == "onnx"
    assert image["modalities"] == ["text", "image"]
    assert image["model"] == "mobileclip2-s2-onnx"
    assert not Path(image["model"]).is_absolute()
    # Exactly one entry declares the image modality (the single-image-space rule).
    image_entries = [e for e in embedders if "image" in e["modalities"]]
    assert len(image_entries) == 1


def test_onnx_multispace_model_names_both_legs() -> None:
    profile = serve.load_profile("onnx-multispace")
    assert serve._model_names_in_profile(profile) == [
        "embeddinggemma-300m-onnx-int8",
        "mobileclip2-s2-onnx",
    ]


# -- jina-text-clip: the manual/local-only hybrid multi-space profile ----------
#
# Download-free, server-free structural checks on the committed profile: it is
# path-free, declares TWO embedding spaces — a dedicated TEXT space on a managed
# (manage: auto) llama-server with `pooling: last`, plus an in-process ONNX CLIP
# (text+image) image leg — and references the three operator-provided paths
# (binary / jina-text GGUF / MobileCLIP2 dir) as ${ENV} placeholders, NEVER as
# machine-absolute paths. Both the patched binary and the model are
# operator-provided (jina-v5-text-nano is a custom arch the stock llama-server
# lacks), so this profile is consumed via `shrike server start --config`, NOT
# `serve --profile`: the launcher (onnx-only materializer) leaves the managed
# entry untouched AND its onnx leg names an operator ${ENV} path, not a bare
# registered dir-name — so the launcher materializes nothing here.

_JINA_TEXT_CLIP_ENV_VARS = (
    "SHRIKE_JINA_TEXT_LLAMA_SERVER",
    "SHRIKE_JINA_TEXT_MODEL",
    "SHRIKE_JINA_TEXT_CLIP_MOBILECLIP2",
)


def test_jina_text_clip_profile_is_path_free_hybrid() -> None:
    profile = serve.load_profile("jina-text-clip")
    assert "collection" not in profile
    embedders = profile["embedders"]
    assert len(embedders) == 2, "jina-text-clip is a TWO-space hybrid"
    # The dedicated TEXT leg: remote with no endpoint = the managed llama-server
    # below; text-only; last-token pooling on the ENTRY (single-managed mode).
    text = embedders[0]
    assert text["runtime"] == "remote"
    assert "endpoint" not in text
    assert text["modalities"] == ["text"]
    assert text["pooling"] == "last"
    # The IMAGE leg: in-process ONNX CLIP (text + image into one shared space) —
    # the single image-embedding space.
    image = embedders[1]
    assert image["runtime"] == "onnx"
    assert image["modalities"] == ["text", "image"]
    assert "pooling" not in image, "a CLIP dual-encoder emits a pre-pooled vector"
    image_entries = [e for e in embedders if "image" in e["modalities"]]
    assert len(image_entries) == 1, "exactly ONE image-embedding space (#580)"
    llama = profile["managed"]["llama_server"]
    assert llama["manage"] == "auto", "Shrike launches the patched binary"
    assert "models_dir" not in llama, "single-managed, NOT #567's router mode"
    assert "mmprojs" not in llama, "no image-embed via the managed server (CLIP leg does it)"


def test_jina_text_clip_operator_paths_are_env_placeholders() -> None:
    # The three operator-provided paths are ${ENV} placeholders, never absolute
    # paths — the path-free invariant for a profile whose binary + models can't
    # be Bazel externals (the patched fork + the operator-local model dirs).
    profile = serve.load_profile("jina-text-clip")
    text = profile["embedders"][0]
    image = profile["embedders"][1]
    llama = profile["managed"]["llama_server"]
    for value in (text["model"], image["model"], llama["binary"]):
        assert value.startswith("${") and value.endswith("}"), (
            f"{value!r} must be an ${{ENV}} placeholder, not a baked path"
        )
        assert not Path(value).is_absolute()
    # Exactly the three documented vars (so the README one-liner stays accurate).
    referenced = {text["model"], image["model"], llama["binary"]}
    assert referenced == {f"${{{name}}}" for name in _JINA_TEXT_CLIP_ENV_VARS}


def test_jina_text_clip_onnx_leg_is_not_a_registered_dir_name() -> None:
    # The CLIP leg's model is an operator ${ENV} path, NOT a bare registered
    # dir-name (e.g. mobileclip2-s2-onnx) — this profile is `--config`-consumed,
    # so the launcher never materializes it. _model_names_in_profile DOES collect
    # any onnx entry's model string (it's onnx), but here that string is the
    # ${ENV} placeholder, which the operator's envsubst replaces with a real dir.
    profile = serve.load_profile("jina-text-clip")
    names = serve._model_names_in_profile(profile)
    assert names == ["${SHRIKE_JINA_TEXT_CLIP_MOBILECLIP2}"]
    # It is NOT a known materializable dir-name, so a stray `serve --profile`
    # would fail loud at resolve_model_dir rather than silently fetch — the
    # placeholder can't collide with a registered model.
    assert "mobileclip2-s2-onnx" not in names


def test_jina_text_clip_compose_leaves_managed_text_entry_untouched() -> None:
    # compose_effective_config rewrites only onnx dir-names it can resolve; the
    # managed text entry (remote/no-endpoint, the GGUF) passes through
    # byte-for-byte (the operator's instantiated config supplies the real path).
    profile = serve.load_profile("jina-text-clip")
    # Resolve the placeholder onnx leg so compose doesn't KeyError on it (the
    # operator's envsubst does this for real; here a stub path stands in).
    onnx_name = profile["embedders"][1]["model"]
    config = serve.compose_effective_config(profile, {onnx_name: "/run/mobileclip2"})
    # The managed text entry is unchanged (still the placeholder).
    assert config["embedders"][0]["model"] == profile["embedders"][0]["model"]
    assert config["managed"] == profile["managed"]


def test_unknown_profile_errors_with_available_list() -> None:
    with pytest.raises(SystemExit) as exc:
        serve.load_profile("does-not-exist")
    msg = str(exc.value)
    assert "does-not-exist" in msg
    # The committed text-onnx profile is named as available.
    assert "text-onnx" in msg


def test_profile_with_collection_key_is_rejected() -> None:
    with pytest.raises(SystemExit, match="path-free"):
        serve.check_path_free("bad", {"collection": "/some/path.anki2", "embedders": []})


def test_path_free_profile_passes() -> None:
    # A profile with no collection: key is accepted (no raise).
    serve.check_path_free("ok", {"embedders": [{"runtime": "onnx", "model": "m"}]})


# -- jina-omni: the manual/local-only single-omni profile ----------------------
#
# Download-free, server-free structural checks on the committed profile: it is
# path-free, declares ONE text+image space on a managed (manage: auto)
# llama-server, sets `pooling: last`, and references the three operator-provided
# paths (binary / model / vision mmproj) as ${ENV} placeholders — NEVER as
# machine-absolute paths (so the file stays portable + can't leak a dev path).
# The model is operator-provided, not a Bazel external, so the launcher leaves it
# untouched: _model_names_in_profile (onnx-only) skips it.

_JINA_OMNI_ENV_VARS = (
    "SHRIKE_JINA_OMNI_LLAMA_SERVER",
    "SHRIKE_JINA_OMNI_MODEL",
    "SHRIKE_JINA_OMNI_VISION_MMPROJ",
)


def test_jina_omni_profile_is_path_free_managed_omni() -> None:
    profile = serve.load_profile("jina-omni")
    assert "collection" not in profile
    embedders = profile["embedders"]
    assert len(embedders) == 1, "jina-omni is ONE shared text+image space"
    entry = embedders[0]
    # remote with no endpoint = the managed llama-server below (the multimodal
    # managed shape); text+image into one space; last-token pooling.
    assert entry["runtime"] == "remote"
    assert "endpoint" not in entry
    assert entry["modalities"] == ["text", "image"]
    assert entry["pooling"] == "last"
    llama = profile["managed"]["llama_server"]
    assert llama["manage"] == "auto", "Shrike launches the patched binary"
    assert llama["mmprojs"], "a vision projector is loaded for image embeds"


def test_jina_omni_operator_paths_are_env_placeholders() -> None:
    # The three operator-provided paths are ${ENV} placeholders, never absolute
    # paths — the path-free invariant for a profile whose binary + model can't be
    # Bazel externals (the patched fork + F16 GGUF are hand-built/operator-local).
    profile = serve.load_profile("jina-omni")
    entry = profile["embedders"][0]
    llama = profile["managed"]["llama_server"]
    for value in (entry["model"], llama["binary"], llama["mmprojs"][0]):
        assert value.startswith("${") and value.endswith("}"), (
            f"{value!r} must be an ${{ENV}} placeholder, not a baked path"
        )
        assert not Path(value).is_absolute()
    # Exactly the three documented vars (so the README one-liner stays accurate).
    referenced = {entry["model"], llama["binary"], llama["mmprojs"][0]}
    assert referenced == {f"${{{name}}}" for name in _JINA_OMNI_ENV_VARS}


def test_jina_omni_has_no_materializable_onnx_model() -> None:
    # The launcher materializes ONLY onnx model dir-names from externals; the
    # operator-provided GGUF is a remote/managed entry it must leave alone (so
    # `serve` never tries to fetch it). _model_names_in_profile is onnx-only.
    profile = serve.load_profile("jina-omni")
    assert serve._model_names_in_profile(profile) == []


def test_jina_omni_compose_leaves_managed_entry_untouched() -> None:
    # compose_effective_config rewrites only onnx dir-names; jina-omni's managed
    # entry (binary/model/mmprojs) passes through byte-for-byte (the operator's
    # instantiated config supplies the real paths, not the launcher).
    profile = serve.load_profile("jina-omni")
    config = serve.compose_effective_config(profile, {})
    assert config["embedders"][0]["model"] == profile["embedders"][0]["model"]
    assert config["managed"] == profile["managed"]


# -- model-name extraction -----------------------------------------------------


def test_model_names_only_onnx_embedders() -> None:
    profile = {
        "embedders": [
            {"runtime": "onnx", "modalities": ["text"], "model": "minilm"},
            {"runtime": "remote", "modalities": ["text"], "endpoint": "http://x"},
            {"runtime": "onnx", "modalities": ["text"], "model": "other"},
        ]
    }
    assert serve._model_names_in_profile(profile) == ["minilm", "other"]


def test_model_names_empty_profile() -> None:
    assert serve._model_names_in_profile({}) == []
    assert serve._model_names_in_profile({"embedders": []}) == []


# -- model resolution (the adversarial-case regression guards) -----------------
#
# Download-free: monkeypatch serve._runfiles / serve._fetchers so no model
# externals or HuggingFace fetch is touched.


class _FakeRunfiles:
    """A stub Bazel runfiles resolver: Rlocation returns a real path only for the
    runfiles keys in *present*; everything else resolves to None (not in this
    binary's runfiles)."""

    def __init__(self, present: dict[str, str]) -> None:
        self._present = present

    def Rlocation(self, key: str) -> str | None:  # noqa: N802 - mirrors the runfiles API
        return self._present.get(key)


def test_resolve_absolute_model_name_points_at_path_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An absolute model: name violates the path-free rule — fail with a pointed
    # message, not the misleading "don't know how to resolve".
    monkeypatch.setattr(serve, "_runfiles", lambda: None)
    monkeypatch.setattr(serve, "_fetchers", lambda: {})
    with pytest.raises(SystemExit, match="path-free"):
        serve.resolve_model_dir("/abs/models/minilm", tmp_path)


def test_resolve_under_bazel_returns_runfiles_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Under Bazel the per-model dir is already assembled in the runfiles by
    # serve.bzl; resolve_model_dir resolves the dir's SENTINEL file and returns its
    # parent IN PLACE (no copy, no fetch — a bare dir has no reliable Rlocation).
    model_dir = tmp_path / "rf" / "all-MiniLM-L6-v2-onnx-int8"
    model_dir.mkdir(parents=True)
    sentinel = model_dir / serve._SENTINEL_NAME
    sentinel.write_text("# marker")
    key = f"{serve._MODEL_RUNFILES_ROOT}/all-MiniLM-L6-v2-onnx-int8/{serve._SENTINEL_NAME}"
    monkeypatch.setattr(serve, "_runfiles", lambda: _FakeRunfiles({key: str(sentinel)}))
    out = serve.resolve_model_dir("all-MiniLM-L6-v2-onnx-int8", tmp_path / "unused")
    assert out == model_dir


def test_resolve_under_bazel_missing_dir_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Under Bazel but the dir isn't in THIS target's runfiles — a forgotten `data`
    # dep / a missing _MODEL_FILES row. Fail loud HERE (naming serve.bzl + the
    # BUILD `models` list), not late + cryptically at the backend's _resolve_files.
    monkeypatch.setattr(serve, "_runfiles", lambda: _FakeRunfiles({}))
    with pytest.raises(SystemExit) as exc:
        serve.resolve_model_dir("mobileclip2-s2-onnx", tmp_path)
    msg = str(exc.value)
    assert "mobileclip2-s2-onnx" in msg
    assert "_MODEL_FILES" in msg
    assert "scripts/BUILD.bazel" in msg


def test_resolve_off_bazel_unknown_model_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Off Bazel, an unknown dir-name must fail loud, naming the model + the fetcher map.
    monkeypatch.setattr(serve, "_runfiles", lambda: None)
    monkeypatch.setattr(serve, "_fetchers", lambda: {"known-model": lambda root: root})
    with pytest.raises(SystemExit) as exc:
        serve.resolve_model_dir("unknown-xyz", tmp_path)
    msg = str(exc.value)
    assert "unknown-xyz" in msg
    assert "fetcher map" in msg


def test_resolve_off_bazel_calls_the_fetcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Off Bazel, a known dir-name delegates to its model_cache fetcher (the single
    # download source) and returns that dir. The fetcher is stubbed — no download.
    fetched = tmp_path / "fetched" / "minilm"
    fetched.mkdir(parents=True)
    monkeypatch.setattr(serve, "_runfiles", lambda: None)
    monkeypatch.setattr(serve, "_fetchers", lambda: {"minilm": lambda root: fetched})
    out = serve.resolve_model_dir("minilm", tmp_path / "models")
    assert out == fetched


# -- the off-Bazel fetcher map -------------------------------------------------
#
# Exercises the real _fetchers() map (imports model_cache) — a registration
# regression guard: every CI/dogfooding profile model dir-name must resolve to a
# callable model_cache fetcher. No download (the fetchers are referenced, never
# called). The bazel runfiles file-map lives in serve.bzl (one source of
# truth); this map is only the off-Bazel download source serve.py owns.


def test_fetchers_cover_the_profile_models() -> None:
    fetchers = serve._fetchers()
    for name in (
        "all-MiniLM-L6-v2-onnx-int8",
        "embeddinggemma-300m-onnx-int8",
        "mobileclip2-s2-onnx",
    ):
        assert name in fetchers, f"{name} missing from _fetchers()"
        assert callable(fetchers[name]), f"{name} has no fetch fn"


def test_fetchers_registers_jina_clip_v2_for_673() -> None:
    # jina-clip-v2 is pre-staged for native fused-graph ClipBackend support;
    # not consumed by any current profile, but kept registered as a download source.
    fetchers = serve._fetchers()
    assert "jina-clip-v2-onnx-int8" in fetchers
    assert callable(fetchers["jina-clip-v2-onnx-int8"])


# -- effective-config composition ----------------------------------------------


def test_compose_rewrites_onnx_model_to_absolute_path() -> None:
    profile = {
        "server": {"port": 8372},
        "embedders": [{"runtime": "onnx", "modalities": ["text"], "model": "minilm"}],
    }
    config = serve.compose_effective_config(profile, {"minilm": "/abs/models/minilm"})
    assert config["server"] == {"port": 8372}  # pass-through unchanged
    assert config["embedders"][0]["model"] == "/abs/models/minilm"
    # Other keys preserved.
    assert config["embedders"][0]["modalities"] == ["text"]


def test_compose_leaves_non_onnx_entries_untouched() -> None:
    profile = {
        "embedders": [
            {"runtime": "remote", "modalities": ["text"], "endpoint": "http://x", "model": "m"},
        ],
        "managed": {"llama_server": {"manage": "attach"}},
    }
    config = serve.compose_effective_config(profile, {})
    # A remote entry's model is NOT a dir-name to rewrite.
    assert config["embedders"][0]["model"] == "m"
    assert config["managed"] == {"llama_server": {"manage": "attach"}}


def test_compose_missing_resolved_path_is_error() -> None:
    profile = {"embedders": [{"runtime": "onnx", "modalities": ["text"], "model": "minilm"}]}
    with pytest.raises(KeyError):
        serve.compose_effective_config(profile, {})


def test_compose_roundtrips_text_onnx_profile() -> None:
    profile = serve.load_profile("text-onnx")
    config = serve.compose_effective_config(
        profile, {"all-MiniLM-L6-v2-onnx-int8": "/run/models/all-MiniLM-L6-v2-onnx-int8"}
    )
    # Serializes cleanly (the launcher writes it to config.yml).
    dumped = yaml.safe_dump(config)
    reloaded = yaml.safe_load(dumped)
    assert reloaded["embedders"][0]["model"] == "/run/models/all-MiniLM-L6-v2-onnx-int8"
    assert reloaded["embedders"][0]["runtime"] == "onnx"


# -- ONNX provider auto-detect + overlay ---------------------------------------
#
# Download-free: stub onnxruntime.get_available_providers / nvidia-smi presence /
# the active-provider readback. No GPU and no real onnxruntime call.


def _stub_ort(monkeypatch: pytest.MonkeyPatch, available: list[str]) -> None:
    """Make serve._available_providers return *available* (stubs the onnxruntime
    query at serve's seam, so no real onnxruntime call)."""
    monkeypatch.setattr(serve, "_available_providers", lambda: list(available))


def test_detect_priority_intersected_cpu_last(monkeypatch: pytest.MonkeyPatch) -> None:
    # A CUDA host: CUDA wins, TensorRT next, CPU always last; CoreML/Dml absent.
    _stub_ort(
        monkeypatch,
        ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(serve, "_nvidia_gpu_present", lambda: True)
    assert serve.detect_providers() == [
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_detect_coreml_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ort(monkeypatch, ["CoreMLExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(serve, "_nvidia_gpu_present", lambda: False)
    assert serve.detect_providers() == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_detect_cpu_only_host(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ort(monkeypatch, ["CPUExecutionProvider"])
    monkeypatch.setattr(serve, "_nvidia_gpu_present", lambda: False)
    assert serve.detect_providers() == ["CPUExecutionProvider"]


def test_detect_onnxruntime_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # onnxruntime not importable → CPU, well-formed (the backend would fail later).
    monkeypatch.setattr(serve, "_available_providers", lambda: None)
    assert serve.detect_providers() == ["CPUExecutionProvider"]


def test_gpu_mismatch_warning_emitted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # NVIDIA GPU present but the base wheel (no CUDA EP) → the carrier remedy.
    _stub_ort(monkeypatch, ["CPUExecutionProvider"])
    monkeypatch.setattr(serve, "_nvidia_gpu_present", lambda: True)
    with caplog.at_level("WARNING", logger="shrike.serve"):
        resolved = serve.detect_providers()
    assert resolved == ["CPUExecutionProvider"]
    assert any("onnxruntime-gpu" in r.message for r in caplog.records)


def test_no_mismatch_warning_when_cuda_present(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # NVIDIA GPU present AND the CUDA EP is there → no carrier warning.
    _stub_ort(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(serve, "_nvidia_gpu_present", lambda: True)
    with caplog.at_level("WARNING", logger="shrike.serve"):
        serve.detect_providers()
    assert not any("onnxruntime-gpu" in r.message for r in caplog.records)


def test_resolve_providers_cpu_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # --cpu wins over detection (would otherwise pick up CUDA).
    _stub_ort(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    args = serve.build_parser().parse_args(["--profile", "text-onnx", "--cpu"])
    assert serve.resolve_providers(args) == ["CPUExecutionProvider"]


def test_resolve_providers_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # --providers takes the user's list verbatim, skipping detection.
    _stub_ort(monkeypatch, ["CPUExecutionProvider"])  # detection would yield CPU
    args = serve.build_parser().parse_args(
        ["--profile", "text-onnx", "--providers", "CUDAExecutionProvider,CPUExecutionProvider"]
    )
    assert serve.resolve_providers(args) == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_resolve_providers_autodetect_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ort(monkeypatch, ["CoreMLExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(serve, "_nvidia_gpu_present", lambda: False)
    args = serve.build_parser().parse_args(["--profile", "text-onnx"])
    assert serve.resolve_providers(args) == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_compose_overlays_providers_onto_onnx() -> None:
    profile = {"embedders": [{"runtime": "onnx", "modalities": ["text"], "model": "minilm"}]}
    config = serve.compose_effective_config(
        profile,
        {"minilm": "/abs/minilm"},
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    assert config["embedders"][0]["providers"] == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_compose_explicit_profile_providers_wins() -> None:
    # A profile that already declares providers: keeps them — detection never
    # overrides an explicit choice.
    profile = {
        "embedders": [
            {
                "runtime": "onnx",
                "modalities": ["text"],
                "model": "minilm",
                "providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
            }
        ]
    }
    config = serve.compose_effective_config(
        profile,
        {"minilm": "/abs/minilm"},
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert config["embedders"][0]["providers"] == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_compose_does_not_overlay_remote_or_platform() -> None:
    # providers: is onnx-only (profiles.py:291) — never injected onto remote or
    # platform entries.
    profile = {
        "embedders": [
            {"runtime": "remote", "modalities": ["text"], "endpoint": "http://x"},
            {"runtime": "platform", "modalities": ["text"]},
        ]
    }
    config = serve.compose_effective_config(
        profile, {}, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert "providers" not in config["embedders"][0]
    assert "providers" not in config["embedders"][1]


def test_compose_no_providers_arg_leaves_entries_unchanged() -> None:
    # providers=None injects nothing.
    profile = {"embedders": [{"runtime": "onnx", "modalities": ["text"], "model": "minilm"}]}
    config = serve.compose_effective_config(profile, {"minilm": "/abs/minilm"}, providers=None)
    assert "providers" not in config["embedders"][0]


def test_read_active_providers_reads_status(monkeypatch: pytest.MonkeyPatch) -> None:
    # The after-readback parses embedding.active_providers from /status.
    import httpx

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"embedding": {"active_providers": ["CoreMLExecutionProvider"]}}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    assert serve.read_active_providers(8372, timeout=1.0) == ["CoreMLExecutionProvider"]


def test_read_active_providers_unreachable_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    def _boom(*a: object, **k: object) -> object:
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(httpx, "get", _boom)
    # A near-zero timeout so the poll loop exits fast (no real sleep needed).
    assert serve.read_active_providers(8372, timeout=0.01) is None


def test_status_port_from_profile_or_default() -> None:
    assert serve._status_port({"server": {"port": 9999}}) == 9999
    assert serve._status_port({}) == 8372
    assert serve._status_port({"server": {}}) == 8372


# -- argument parsing ----------------------------------------------------------


def test_foreground_is_default() -> None:
    args = serve.build_parser().parse_args(["--profile", "text-onnx"])
    assert args.foreground is True
    assert args.profile == "text-onnx"
    assert args.seed is None
    assert args.import_path is None


def test_daemon_flips_foreground() -> None:
    args = serve.build_parser().parse_args(["--profile", "text-onnx", "--daemon"])
    assert args.foreground is False


def test_seed_qa_parses() -> None:
    args = serve.build_parser().parse_args(["--profile", "text-onnx", "--seed", "qa"])
    assert args.seed == "qa"


def test_seed_and_import_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        serve.build_parser().parse_args(
            ["--profile", "text-onnx", "--seed", "qa", "--import", "x.apkg"]
        )


def test_foreground_and_daemon_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        serve.build_parser().parse_args(["--profile", "text-onnx", "--foreground", "--daemon"])


def test_profile_is_required() -> None:
    with pytest.raises(SystemExit):
        serve.build_parser().parse_args([])


def test_path_overrides_parse() -> None:
    args = serve.build_parser().parse_args(
        [
            "--profile",
            "text-onnx",
            "--collection",
            "/c.anki2",
            "--cache-dir",
            "/c",
            "--log-dir",
            "/l",
        ]
    )
    assert args.collection == "/c.anki2"
    assert args.cache_dir == "/c"
    assert args.log_dir == "/l"


# -- the server argv the launcher invokes --------------------------------------


def test_server_argv_rides_config_and_run_paths() -> None:
    argv = serve._server_argv(
        config_path=Path("/run/config.yml"),
        collection_path=Path("/run/collection/working.anki2"),
        cache_dir=Path("/run/cache"),
        log_dir=Path("/run/logs"),
        foreground=True,
    )
    # Capability config rides as --config (config-file-only); run paths as flags.
    assert "--config" in argv
    assert argv[argv.index("--config") + 1] == "/run/config.yml"
    assert argv[:1] == ["--config"]
    assert "server" in argv and "start" in argv
    assert "--collection" in argv
    assert argv[argv.index("--collection") + 1] == "/run/collection/working.anki2"
    assert "--cache-dir" in argv and "--log-dir" in argv
    assert "--foreground" in argv
    # The launcher must NOT invent capability flags (config-file-only invariant).
    for forbidden in ("--embedding-model", "--embedding-backend", "--embedders"):
        assert forbidden not in argv


def test_server_argv_daemon_omits_foreground() -> None:
    argv = serve._server_argv(
        config_path=Path("/c.yml"),
        collection_path=Path("/col.anki2"),
        cache_dir=Path("/cache"),
        log_dir=Path("/logs"),
        foreground=False,
    )
    assert "--foreground" not in argv


# -- the --import MVP stub -----------------------------------------------------


def test_import_is_a_loud_stub() -> None:
    with pytest.raises(SystemExit, match="not wired yet"):
        serve.main(["--profile", "text-onnx", "--import", "deck.apkg"])
