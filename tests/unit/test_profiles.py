"""Config model v2 (#498): parse, validate, migrate, and resolve profiles.

Pure-module tests — build features are passed in, so every build/profile
combination is exercised without needing that build. The serving shapes are
pinned through `plan_to_runtime_params` (the slice-1 bridge onto today's
runtime).
"""

from __future__ import annotations

import pytest

from shrike.profiles import (
    EmbedderEntry,
    ManagedLlama,
    ProfileError,
    parse_capabilities,
    plan_to_runtime_params,
    resolve_profile,
)

SERVER = ("anki-core", "engine-ort", "engine-remote", "manage-llama")
MOBILE = ("anki-core", "engine-remote", "engine-apple")


def _resolve(config: dict, features=SERVER):
    return resolve_profile(parse_capabilities(config), features)


# ── Parsing: every entry shape round-trips ───────────────────────────────────


class TestParse:
    def test_text_onnx_entry(self):
        caps = parse_capabilities(
            {
                "embedders": [
                    {
                        "modalities": ["text"],
                        "runtime": "onnx",
                        "model": "~/models/minilm",
                        "pooling": "mean",
                        "providers": ["CoreMLExecutionProvider"],
                        "batch_size": 16,
                    }
                ]
            }
        )
        assert caps.embedders == (
            EmbedderEntry(
                modalities=("text",),
                runtime="onnx",
                model="~/models/minilm",
                pooling="mean",
                providers=("CoreMLExecutionProvider",),
                batch_size=16,
            ),
        )
        assert not caps.legacy and not caps.warnings

    def test_clip_pair_entry(self):
        caps = parse_capabilities(
            {"embedders": [{"modalities": ["text", "image"], "runtime": "onnx", "model": "m"}]}
        )
        assert caps.embedders[0].modalities == ("text", "image")

    def test_omni_single_entry(self):
        # jina-v5-omni as ONE entry (docs/distribution.md) — parses as a
        # single space spanning three modalities.
        caps = parse_capabilities(
            {
                "embedders": [
                    {
                        "modalities": ["text", "image", "audio"],
                        "runtime": "remote",
                        "model": "jina-embeddings-v5-omni-small",
                    }
                ]
            }
        )
        assert caps.embedders[0].modalities == ("text", "image", "audio")

    def test_remote_entry_with_endpoint_and_key(self):
        caps = parse_capabilities(
            {
                "embedders": [
                    {
                        "modalities": ["text"],
                        "runtime": "remote",
                        "model": "text-embedding-3-small",
                        "endpoint": "https://api.example.com/v1",
                        "api_key_env": "EXAMPLE_API_KEY",
                    }
                ]
            }
        )
        e = caps.embedders[0]
        assert e.endpoint == "https://api.example.com/v1"
        assert e.api_key_env == "EXAMPLE_API_KEY"

    def test_managed_llama_and_sync(self):
        caps = parse_capabilities(
            {
                "embedders": [{"modalities": ["text"], "runtime": "remote", "model": "m.gguf"}],
                "managed": {
                    "llama_server": {
                        "manage": "auto",
                        "binary": "/opt/llama-server",
                        "args": ["--flash-attn"],
                        "port": 8373,
                    },
                    "sync_server": {"manage": "off"},
                },
            }
        )
        assert caps.managed_llama == ManagedLlama(
            manage="auto", binary="/opt/llama-server", args=("--flash-attn",), port=8373
        )
        assert caps.managed_sync is not None and caps.managed_sync.manage == "off"

    def test_recognizer_rows_parse(self):
        caps = parse_capabilities(
            {
                "embedders": [],
                "recognizers": {
                    "ocr": {"runtime": "platform"},
                    "describe": {
                        "runtime": "remote",
                        "endpoint": "http://127.0.0.1:8081/v1",
                        "api_key_env": "K",
                    },
                },
            }
        )
        assert {r.source for r in caps.recognizers} == {"ocr", "describe"}

    # — illegal shapes are structurally inexpressible (loud, named errors) —

    @pytest.mark.parametrize(
        ("config", "match"),
        [
            ({"embedders": [{"modalities": [], "runtime": "onnx", "model": "m"}]}, "modalities"),
            (
                {"embedders": [{"modalities": ["video"], "runtime": "onnx", "model": "m"}]},
                "unknown modality",
            ),
            (
                {"embedders": [{"modalities": ["text"], "runtime": "torch", "model": "m"}]},
                "runtime",
            ),
            ({"embedders": [{"modalities": ["text"], "runtime": "onnx"}]}, "model is required"),
            (
                {
                    "embedders": [
                        {"modalities": ["text"], "runtime": "onnx", "model": "m", "endpoint": "x"}
                    ]
                },
                "only to runtime: remote",
            ),
            (
                {
                    "embedders": [
                        {
                            "modalities": ["text"],
                            "runtime": "onnx",
                            "model": "m",
                            "api_key_env": "K",
                        }
                    ]
                },
                "only to runtime: remote",
            ),
            (
                {
                    "embedders": [
                        {
                            "modalities": ["text"],
                            "runtime": "remote",
                            "model": "m",
                            "providers": ["CPU"],
                        }
                    ]
                },
                "only to runtime: onnx",
            ),
            (
                {
                    "embedders": [
                        {"modalities": ["text"], "runtime": "onnx", "model": "m", "batch_size": 0}
                    ]
                },
                "batch_size",
            ),
            (
                {
                    "embedders": [
                        {"modalities": ["text"], "runtime": "onnx", "model": "m", "extra": 1}
                    ]
                },
                "unknown key",
            ),
            ({"recognizers": {"vlm": {"runtime": "remote"}}}, "unknown source"),
            ({"recognizers": {"ocr": {"runtime": "gpu"}}}, "runtime"),
            (
                {"recognizers": {"ocr": {"runtime": "platform", "endpoint": "x"}}},
                "only to runtime: remote",
            ),
            ({"managed": {"llama_server": {"manage": "maybe"}}}, "manage"),
            ({"managed": {"mystery": {}}}, "unknown key"),
        ],
    )
    def test_invalid_shapes_error(self, config, match):
        with pytest.raises(ProfileError, match=match):
            parse_capabilities(config)

    def test_v2_plus_legacy_is_an_error(self):
        with pytest.raises(ProfileError, match="both the v2 sections"):
            parse_capabilities(
                {
                    "embedders": [{"modalities": ["text"], "runtime": "onnx", "model": "m"}],
                    "embedding": {"model": "old.gguf"},
                }
            )


# ── Legacy migration: warn-and-map, one release ──────────────────────────────


class TestLegacyMigration:
    def test_llama_section_maps_to_remote_plus_manager(self):
        caps = parse_capabilities(
            {
                "embedding": {
                    "model": "~/m.gguf",
                    "llama_server": "/opt/llama-server",
                    "port": 9999,
                    "pooling": "last",
                    "extra_args": ["--flash-attn"],
                }
            }
        )
        assert caps.legacy
        assert any("deprecated" in w for w in caps.warnings)
        (e,) = caps.embedders
        assert e.runtime == "remote" and e.endpoint is None and e.pooling == "last"
        assert caps.managed_llama == ManagedLlama(
            manage="auto", binary="/opt/llama-server", args=("--flash-attn",), port=9999
        )

    def test_onnx_section_maps_to_onnx_entry(self):
        caps = parse_capabilities(
            {
                "embedding": {
                    "backend": "onnx",
                    "model": "~/m",
                    "onnx_providers": ["CUDAExecutionProvider"],
                    "batch_size": 8,
                }
            }
        )
        (e,) = caps.embedders
        assert e.runtime == "onnx" and e.modalities == ("text",)
        assert e.providers == ("CUDAExecutionProvider",) and e.batch_size == 8

    def test_clip_section_maps_to_text_image_entry(self):
        caps = parse_capabilities({"embedding": {"backend": "clip", "model": "~/clip"}})
        (e,) = caps.embedders
        assert e.runtime == "onnx" and e.modalities == ("text", "image")

    def test_legacy_ocr_warns_and_degrades(self):
        # Legacy semantics preserved: warn + absent capability, never a
        # boot-refusing error (the old flag degraded the same way).
        caps = parse_capabilities({"recognition": {"ocr": "apple"}})
        assert caps.recognizers == ()
        assert any("#502" in w for w in caps.warnings)
        plan = resolve_profile(caps, SERVER)
        assert any("#502" in w for w in plan.warnings)

    def test_empty_config_is_empty_capabilities(self):
        caps = parse_capabilities({})
        assert caps.embedders == () and caps.recognizers == ()
        plan = resolve_profile(caps, SERVER)
        assert plan.embedder is None
        assert plan_to_runtime_params(plan)["backend"] is None


# ── Resolution: build-capability intersection, named errors ──────────────────


class TestResolve:
    def test_multi_space_is_a_named_error_until_229(self):
        # text + CLIP as TWO entries = two vector spaces — the #229 substrate.
        config = {
            "embedders": [
                {"modalities": ["text"], "runtime": "onnx", "model": "a"},
                {"modalities": ["text", "image"], "runtime": "onnx", "model": "b"},
            ]
        }
        with pytest.raises(ProfileError, match="#229"):
            _resolve(config)

    def test_platform_embedder_on_server_names_the_profile(self):
        config = {"embedders": [{"modalities": ["text"], "runtime": "platform"}]}
        with pytest.raises(ProfileError, match="server build"):
            _resolve(config)

    def test_onnx_on_a_build_without_ort_names_the_profile(self):
        config = {"embedders": [{"modalities": ["text"], "runtime": "onnx", "model": "m"}]}
        with pytest.raises(ProfileError, match="mobile build"):
            _resolve(config, MOBILE)

    def test_ocr_platform_on_server_names_profile_and_replacement(self):
        config = {"embedders": [], "recognizers": {"ocr": {"runtime": "platform"}}}
        with pytest.raises(ProfileError, match="never in the server build.*#502"):
            _resolve(config)

    def test_ocr_platform_on_mobile_resolves(self):
        config = {"embedders": [], "recognizers": {"ocr": {"runtime": "platform"}}}
        # The build-derived check: the same declaration is valid where the
        # engine is compiled. (Serving it is the kernel's existing OCR path.)
        plan = resolve_profile(parse_capabilities(config), MOBILE)
        assert plan.embedder is None

    @pytest.mark.parametrize(
        ("source", "issue"),
        [("asr", "#485"), ("describe", "#485")],
    )
    def test_unintegrated_recognizers_name_their_issue(self, source, issue):
        config = {
            "embedders": [],
            "recognizers": {source: {"runtime": "remote", "endpoint": "http://x/v1"}},
        }
        with pytest.raises(ProfileError, match=issue):
            _resolve(config)

    def test_remote_ocr_names_502(self):
        config = {
            "embedders": [],
            "recognizers": {"ocr": {"runtime": "remote", "endpoint": "http://x/v1"}},
        }
        with pytest.raises(ProfileError, match="#502"):
            _resolve(config)

    def test_sync_server_auto_names_36(self):
        config = {"embedders": [], "managed": {"sync_server": {"manage": "auto"}}}
        with pytest.raises(ProfileError, match="#36"):
            _resolve(config)

    def test_manage_attach_resolves_without_manage_llama_feature(self):
        # attach uses someone else's server — works on builds without the
        # manager (the mobile set), unlike manage: auto.
        config = {
            "embedders": [{"modalities": ["text"], "runtime": "remote"}],
            "managed": {"llama_server": {"manage": "attach", "port": 9000}},
        }
        plan = resolve_profile(parse_capabilities(config), MOBILE)
        assert plan.managed_llama is not None and plan.managed_llama.manage == "attach"

    def test_attach_rejects_launch_knobs(self):
        config = {
            "embedders": [{"modalities": ["text"], "runtime": "remote"}],
            "managed": {"llama_server": {"manage": "attach", "binary": "/opt/llama-server"}},
        }
        with pytest.raises(ProfileError, match="launch knobs"):
            _resolve(config)
        config["managed"]["llama_server"] = {"manage": "attach", "args": ["--mlock"]}
        with pytest.raises(ProfileError, match="args don't apply"):
            _resolve(config)

    @pytest.mark.parametrize(
        "managed",
        [
            {"llama_server": {"manage": "attach"}},
            None,  # explicit endpoint, no manager at all
        ],
    )
    def test_pooling_rejected_when_shrike_does_not_launch_the_server(self, managed):
        entry = {"modalities": ["text"], "runtime": "remote", "pooling": "last"}
        if managed is None:
            entry["endpoint"] = "https://api.example.com/v1"
        config = {"embedders": [entry]}
        if managed is not None:
            config["managed"] = managed
        with pytest.raises(ProfileError, match="owns its own pooling"):
            _resolve(config)

    def test_remote_without_endpoint_and_manager_off_is_an_error(self):
        config = {
            "embedders": [{"modalities": ["text"], "runtime": "remote", "model": "m.gguf"}],
            "managed": {"llama_server": {"manage": "off"}},
        }
        with pytest.raises(ProfileError, match="manage is off"):
            _resolve(config)

    def test_remote_without_endpoint_needs_manage_llama_feature(self):
        config = {"embedders": [{"modalities": ["text"], "runtime": "remote", "model": "m.gguf"}]}
        with pytest.raises(ProfileError, match="manage-llama"):
            _resolve(config, MOBILE)


# ── The slice-1 bridge onto today's runtime ──────────────────────────────────


class TestLegacyBridge:
    def test_onnx_text_entry_drives_onnx_backend(self):
        plan = _resolve(
            {
                "embedders": [
                    {
                        "modalities": ["text"],
                        "runtime": "onnx",
                        "model": "~/m",
                        "pooling": "cls",
                        "providers": ["CPUExecutionProvider"],
                    }
                ]
            }
        )
        legacy = plan_to_runtime_params(plan)
        assert legacy["backend"] == "onnx" and legacy["pooling"] == "cls"
        assert legacy["onnx_providers"] == ["CPUExecutionProvider"]

    def test_text_image_entry_drives_clip_backend(self):
        plan = _resolve(
            {"embedders": [{"modalities": ["text", "image"], "runtime": "onnx", "model": "~/clip"}]}
        )
        assert plan_to_runtime_params(plan)["backend"] == "clip"

    def test_managed_remote_entry_drives_llama_backend(self):
        plan = _resolve(
            {
                "embedders": [
                    {
                        "modalities": ["text"],
                        "runtime": "remote",
                        "model": "~/m.gguf",
                        "pooling": "last",
                    }
                ],
                "managed": {
                    "llama_server": {
                        "binary": "/opt/llama-server",
                        "port": 8474,
                        "args": ["--mlock"],
                    }
                },
            }
        )
        legacy = plan_to_runtime_params(plan)
        assert legacy["backend"] == "llama"
        assert legacy["llama_server"] == "/opt/llama-server"
        assert legacy["port"] == 8474 and legacy["extra_args"] == ["--mlock"]
        assert legacy["pooling"] == "last"

    def test_external_endpoint_drives_remote_backend(self):
        plan = _resolve(
            {
                "embedders": [
                    {
                        "modalities": ["text"],
                        "runtime": "remote",
                        "model": "text-embedding-3-small",
                        "endpoint": "https://api.example.com/v1",
                        "api_key_env": "EXAMPLE_API_KEY",
                    }
                ]
            }
        )
        params = plan_to_runtime_params(plan)
        assert params["backend"] == "remote"
        assert params["endpoint"] == "https://api.example.com/v1"
        assert params["api_key_env"] == "EXAMPLE_API_KEY"
        assert params["model"] == "text-embedding-3-small"

    def test_attach_drives_remote_backend_at_the_managed_port(self):
        plan = _resolve(
            {
                "embedders": [{"modalities": ["text"], "runtime": "remote"}],
                "managed": {"llama_server": {"manage": "attach", "port": 9000}},
            }
        )
        params = plan_to_runtime_params(plan)
        assert params["backend"] == "remote"
        assert params["endpoint"] == "http://127.0.0.1:9000"
        assert params["api_key_env"] is None

    def test_attach_without_port_uses_the_manager_default(self):
        plan = _resolve(
            {
                "embedders": [{"modalities": ["text"], "runtime": "remote"}],
                "managed": {"llama_server": {"manage": "attach"}},
            }
        )
        assert plan_to_runtime_params(plan)["endpoint"] == "http://127.0.0.1:8373"

    def test_legacy_round_trip_matches_original_section(self):
        # A legacy llama config migrated to v2 and bridged back yields the
        # original section's params — the migration mapping is lossless for
        # the shapes it maps. (NOTE: production's legacy path short-circuits
        # to resolve_embedding and never bridges — that byte-equivalence is
        # pinned in test_config.py::test_legacy_config_runs_the_old_cascade.
        # This test pins the MAPPING itself, which becomes the live path when
        # the slice-2 facade rework consumes the plan natively.)
        legacy_section = {
            "embedding": {
                "model": "~/m.gguf",
                "llama_server": "/opt/llama-server",
                "port": 8373,
                "pooling": "last",
                "extra_args": ["--flash-attn"],
                "context_size": 4096,
                "threads": 8,
                "gpu_layers": 99,
            }
        }
        plan = _resolve(legacy_section)
        bridged = plan_to_runtime_params(plan)
        assert bridged["backend"] == "llama"
        for key in ("llama_server", "port", "pooling", "context_size", "threads", "gpu_layers"):
            expected = legacy_section["embedding"][key if key != "llama_server" else "llama_server"]
            assert bridged[key] == expected
        assert bridged["extra_args"] == ["--flash-attn"]
        assert bridged["model"] == "~/m.gguf"


class TestManagedConsumption:
    """managed.llama_server must be consumed by a remote entry (the
    no-silent-noop rule) — manage: off is a valid explicit declaration."""

    def test_unconsumed_managed_auto_is_an_error(self):
        config = {
            "embedders": [{"modalities": ["text"], "runtime": "onnx", "model": "m"}],
            "managed": {"llama_server": {"manage": "auto"}},
        }
        with pytest.raises(ProfileError, match="nothing consumes it"):
            _resolve(config)

    def test_attach_plus_explicit_endpoint_is_an_error(self):
        config = {
            "embedders": [{"modalities": ["text"], "runtime": "remote", "endpoint": "http://e/v1"}],
            "managed": {"llama_server": {"manage": "attach"}},
        }
        with pytest.raises(ProfileError, match="nothing consumes it"):
            _resolve(config)

    def test_manage_off_is_fine_alongside_any_entry(self):
        config = {
            "embedders": [{"modalities": ["text"], "runtime": "onnx", "model": "m"}],
            "managed": {"llama_server": {"manage": "off"}},
        }
        plan = _resolve(config)
        assert plan.embedder is not None and plan.embedder.runtime == "onnx"
