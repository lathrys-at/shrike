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
    plan_to_runtime_params_set,
    recognizer_plans,
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

    def test_text_inclusive_modalities_accepted(self):
        # #603 control: the floor only rejects entries MISSING text — every
        # text-inclusive shape still parses (text, text+image, text+audio).
        for mods in (["text"], ["text", "image"], ["text", "audio"]):
            caps = parse_capabilities(
                {"embedders": [{"modalities": mods, "runtime": "remote", "model": "m"}]}
            )
            assert caps.embedders[0].modalities == tuple(mods)

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
            # #603: every space must embed text (the EmbedderBackend contract is
            # modalities ⊇ {text}). An image-only entry is currently a valid
            # shape per the unknown-modality check but violates the protocol —
            # reject it at parse time. (Same for an audio-only entry.)
            (
                {"embedders": [{"modalities": ["image"], "runtime": "remote", "model": "m"}]},
                "must include 'text'",
            ),
            (
                {"embedders": [{"modalities": ["audio"], "runtime": "remote", "model": "m"}]},
                "must include 'text'",
            ),
            (
                {
                    "embedders": [
                        {"modalities": ["image", "audio"], "runtime": "remote", "model": "m"}
                    ]
                },
                "must include 'text'",
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
    def test_multi_space_resolves_to_an_ordered_set_with_primary_roles(self):
        # text + CLIP as TWO entries = two vector spaces (#233 — the multi-space
        # substrate, no longer a #229 reject). Both resolve; the per-modality
        # PRIMARY is the FIRST declaring entry (text → entry 0; image → entry 1,
        # the first to carry it).
        config = {
            "embedders": [
                {"modalities": ["text"], "runtime": "onnx", "model": "a"},
                {"modalities": ["text", "image"], "runtime": "onnx", "model": "b"},
            ]
        }
        plan = _resolve(config)
        assert len(plan.embedders) == 2
        # The back-compat .embedder accessor returns the PRIMARY (first) entry.
        assert plan.embedder is not None and plan.embedder.model == "a"
        text_space, clip_space = plan.embedders
        assert text_space.primary_modalities == frozenset({"text"})
        assert text_space.text_capable
        # entry 1 is primary only for image — text was already claimed by entry 0.
        assert clip_space.primary_modalities == frozenset({"image"})
        assert clip_space.text_capable
        # plan_to_runtime_params (the N=1 accessor) keys off the primary.
        assert plan_to_runtime_params(plan)["backend"] == "onnx"
        # The N-dict set emits one dict per space, in declaration order.
        dicts = plan_to_runtime_params_set(plan)
        assert [d["backend"] for d in dicts] == ["onnx", "clip"]

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

    def test_asr_recognizer_names_its_issue(self):
        # asr is still PR2 — declared but not wired.
        config = {
            "embedders": [],
            "recognizers": {"asr": {"runtime": "remote", "endpoint": "http://x/v1"}},
        }
        with pytest.raises(ProfileError, match="asr"):
            _resolve(config)

    def test_describe_remote_resolves_and_plans(self):
        # #485 PR1: describe over a remote vision endpoint is attachable. It
        # resolves (no reject) and maps onto a describe-remote RecognizerPlan.
        config = {
            "embedders": [],
            "recognizers": {
                "describe": {
                    "runtime": "remote",
                    "endpoint": "http://vlm.local/v1",
                    "model": "smolvlm",
                    "api_key_env": "VLM_KEY",
                }
            },
        }
        plan = _resolve(config)
        assert {r.source for r in plan.recognizers} == {"describe"}
        plans = recognizer_plans(plan)
        assert len(plans) == 1
        (p,) = plans
        assert p.purpose == "describe"
        assert p.kind == "describe-remote"
        assert p.endpoint == "http://vlm.local/v1"
        assert p.model == "smolvlm"
        assert p.api_key_env == "VLM_KEY"

    def test_describe_remote_needs_an_endpoint(self):
        config = {
            "embedders": [],
            "recognizers": {"describe": {"runtime": "remote"}},
        }
        with pytest.raises(ProfileError, match="endpoint"):
            _resolve(config)

    def test_describe_non_remote_runtime_is_rejected(self):
        # platform/onnx describe engines don't exist — only remote is wired.
        config = {
            "embedders": [],
            "recognizers": {"describe": {"runtime": "platform"}},
        }
        with pytest.raises(ProfileError, match="remote"):
            _resolve(config)

    def test_describe_remote_needs_engine_remote_feature(self):
        # A build without engine-remote can't serve a describe engine.
        config = {
            "embedders": [],
            "recognizers": {"describe": {"runtime": "remote", "endpoint": "http://x/v1"}},
        }
        with pytest.raises(ProfileError, match="engine-remote"):
            resolve_profile(parse_capabilities(config), ("anki-core",))

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


class TestMultimodalRemote:
    """#501: a [text, image] remote entry routes images; mmprojs ride the
    managed server and fold into the fingerprint."""

    def test_managed_omni_carries_modalities_and_mmprojs(self):
        config = {
            "embedders": [
                {"modalities": ["text", "image"], "runtime": "remote", "model": "~/omni.gguf"}
            ],
            "managed": {
                "llama_server": {
                    "manage": "auto",
                    "mmprojs": ["~/vision.mmproj.gguf", "~/audio.mmproj.gguf"],
                }
            },
        }
        params = plan_to_runtime_params(_resolve(config))
        assert params["backend"] == "llama"
        assert params["modalities"] == frozenset({"text", "image"})
        assert params["mmprojs"] == ["~/vision.mmproj.gguf", "~/audio.mmproj.gguf"]

    def test_attached_multimodal_carries_modalities_no_mmprojs(self):
        config = {
            "embedders": [{"modalities": ["text", "image"], "runtime": "remote"}],
            "managed": {"llama_server": {"manage": "attach", "port": 9000}},
        }
        params = plan_to_runtime_params(_resolve(config))
        assert params["backend"] == "remote"
        assert params["modalities"] == frozenset({"text", "image"})
        assert "mmprojs" not in params  # attach loads its own

    def test_mmprojs_without_image_modality_is_an_error(self):
        config = {
            "embedders": [{"modalities": ["text"], "runtime": "remote", "model": "m.gguf"}],
            "managed": {"llama_server": {"manage": "auto", "mmprojs": ["~/v.mmproj"]}},
        }
        with pytest.raises(ProfileError, match="does not declare an image modality"):
            _resolve(config)

    def test_mmprojs_rejected_when_image_is_on_a_separate_endpoint(self):
        # #609: the managed llama-server's CONSUMER is the text-only
        # remote/no-endpoint entry; image lives on a SEPARATE remote endpoint.
        # The projectors would load onto the text-only server that never embeds
        # images (silent no-op + a needless TEXT rebuild) — so reject, even
        # though *some* space declares image. Was wrongly ACCEPTED before #609.
        config = {
            "embedders": [
                # The managed-llama consumer: remote, no endpoint, text-only.
                {"modalities": ["text"], "runtime": "remote", "model": "~/text.gguf"},
                # Image lives elsewhere — a separate remote endpoint.
                {
                    "modalities": ["text", "image"],
                    "runtime": "remote",
                    "endpoint": "https://img.example.com/v1",
                    "model": "clip",
                },
            ],
            "managed": {"llama_server": {"manage": "auto", "mmprojs": ["~/v.mmproj"]}},
        }
        with pytest.raises(ProfileError, match="does not declare an image modality"):
            _resolve(config)

    def test_mmprojs_accepted_when_the_consumer_declares_image(self):
        # #609 control: the SEPARATE-endpoint shape is fine when the managed
        # server's own consumer (remote, no endpoint) is the image-capable one.
        config = {
            "embedders": [
                # The managed-llama consumer is image-capable — projectors apply.
                {"modalities": ["text", "image"], "runtime": "remote", "model": "~/omni.gguf"},
                # A separate text-only remote endpoint coexists fine.
                {
                    "modalities": ["text"],
                    "runtime": "remote",
                    "endpoint": "https://text.example.com/v1",
                    "model": "minilm",
                },
            ],
            "managed": {"llama_server": {"manage": "auto", "mmprojs": ["~/v.mmproj"]}},
        }
        params = plan_to_runtime_params(_resolve(config))
        assert params["mmprojs"] == ["~/v.mmproj"]
        assert "image" in params["modalities"]

    def test_attach_rejects_mmprojs(self):
        config = {
            "embedders": [{"modalities": ["text", "image"], "runtime": "remote"}],
            "managed": {"llama_server": {"manage": "attach", "mmprojs": ["~/v.mmproj"]}},
        }
        with pytest.raises(ProfileError, match="mmprojs don't"):
            _resolve(config)

    def test_onnx_image_entry_carries_modalities(self):
        config = {
            "embedders": [{"modalities": ["text", "image"], "runtime": "onnx", "model": "~/clip"}]
        }
        params = plan_to_runtime_params(_resolve(config))
        assert params["backend"] == "clip"
        assert params["modalities"] == frozenset({"text", "image"})


# ── Multi-space embedding (#233 — the substrate's config half) ────────────────


class TestMultiSpace:
    def test_text_onnx_plus_clip_resolves_to_two_spaces(self):
        # The eval's canonical config (MiniLM text + a CLIP fixture) — a
        # dedicated text space PLUS a separate joint text↔image space. It
        # resolves WITHOUT a ProfileError (the #229 reject is gone), and
        # plan_to_runtime_params_set emits the two backend dicts.
        config = {
            "embedders": [
                {"modalities": ["text"], "runtime": "onnx", "model": "~/minilm"},
                {"modalities": ["text", "image"], "runtime": "onnx", "model": "~/clip"},
            ]
        }
        plan = _resolve(config)
        assert len(plan.embedders) == 2
        dicts = plan_to_runtime_params_set(plan)
        assert len(dicts) == 2
        assert [d["backend"] for d in dicts] == ["onnx", "clip"]
        assert dicts[1]["modalities"] == frozenset({"text", "image"})
        # The set's first dict equals the primary accessor's dict — N=1 stays
        # byte-identical because the primary is just the first space.
        assert dicts[0] == plan_to_runtime_params(plan)

    def test_single_space_set_is_a_one_tuple_equal_to_the_primary(self):
        # N=1: the set is a 1-tuple whose sole element equals the primary dict,
        # so the single-space runtime is unchanged.
        config = {"embedders": [{"modalities": ["text"], "runtime": "onnx", "model": "~/m"}]}
        plan = _resolve(config)
        dicts = plan_to_runtime_params_set(plan)
        assert len(dicts) == 1
        assert dicts[0] == plan_to_runtime_params(plan)

    def test_no_embedder_set_is_empty(self):
        plan = _resolve({"embedders": []})
        assert plan.embedders == ()
        assert plan_to_runtime_params_set(plan) == ()

    def test_per_modality_primary_is_the_first_declaring_space(self):
        # The first space claims text+image; a second TEXT space declares text
        # too but is primary for NOTHING (its modality is already claimed). The
        # role mirrors the kernel's insertion-order primary. (The duplicate is
        # text, not image: at most one image space is allowed since #580 — the
        # primary-role logic is modality-agnostic, so text exercises it.)
        config = {
            "embedders": [
                {"modalities": ["text", "image"], "runtime": "onnx", "model": "~/a"},
                {"modalities": ["text"], "runtime": "onnx", "model": "~/b"},
            ]
        }
        plan = _resolve(config)
        a, b = plan.embedders
        assert a.primary_modalities == frozenset({"text", "image"})
        assert b.primary_modalities == frozenset()

    def test_two_image_spaces_are_rejected(self):
        # #580: at most ONE image-embedding space. Floor-admission (the cross-
        # space mechanism since #580) admits a single image space on its own
        # calibrated floor; two would reintroduce the N≥2 flood the retired
        # relative gate guarded against, with no mechanism left to bound it → a
        # named config error, not a silent degrade.
        # Both spaces are valid text+image entries (each satisfies the #603
        # modalities ⊇ {text} floor); the ceiling rejects the SECOND image space.
        config = {
            "embedders": [
                {"modalities": ["text", "image"], "runtime": "onnx", "model": "~/a"},
                {"modalities": ["text", "image"], "runtime": "onnx", "model": "~/b"},
            ]
        }
        with pytest.raises(ProfileError, match="at most ONE image-embedding space"):
            _resolve(config)

    def test_one_image_plus_text_only_space_resolves(self):
        # The valid multi-space shape: one image-capable space + a text-only
        # space. The single-image rule allows it (only the image modality is
        # capped at one).
        config = {
            "embedders": [
                {"modalities": ["text", "image"], "runtime": "onnx", "model": "~/clip"},
                {"modalities": ["text"], "runtime": "onnx", "model": "~/minilm"},
            ]
        }
        plan = _resolve(config)
        assert len(plan.embedders) == 2

    def test_two_managed_remote_no_endpoint_entries_are_rejected(self):
        # In SINGLE-model mode (no managed.llama_server.models_dir) at most one
        # entry may bind the single managed llama-server (a remote entry with no
        # endpoint). Two is ambiguous → a named error that now points at router
        # mode (models_dir) as the way to share ONE server (#567).
        config = {
            "embedders": [
                {"modalities": ["text"], "runtime": "remote", "model": "a.gguf"},
                {"modalities": ["text"], "runtime": "remote", "model": "b.gguf"},
            ],
            "managed": {"llama_server": {"manage": "auto"}},
        }
        with pytest.raises(ProfileError, match="bind the single managed llama-server"):
            _resolve(config)
        with pytest.raises(ProfileError, match="models_dir"):
            _resolve(config)

    def test_text_onnx_plus_remote_endpoint_space_resolves(self):
        # A dedicated local text space + a separate remote (endpoint) space —
        # the no-omni deployment shape the epic targets. Both resolve; the set
        # carries both backend dicts.
        config = {
            "embedders": [
                {"modalities": ["text"], "runtime": "onnx", "model": "~/minilm"},
                {
                    "modalities": ["text", "image"],
                    "runtime": "remote",
                    "model": "clip-1",
                    "endpoint": "https://api.example.com/v1",
                    "api_key_env": "EXAMPLE_API_KEY",
                },
            ]
        }
        plan = _resolve(config)
        dicts = plan_to_runtime_params_set(plan)
        assert [d["backend"] for d in dicts] == ["onnx", "remote"]
        assert dicts[1]["endpoint"] == "https://api.example.com/v1"


class TestRouterMode:
    """Shared managed llama-server router (#567): N remote/no-endpoint spaces
    collapse onto ONE spawn + N model-pinned clients, driven by
    managed.llama_server.models_dir. The N=1 single-managed path is unchanged
    (its own dedicated test below proves byte-equality)."""

    def test_two_router_consumers_collapse_onto_one_remote_set(self):
        # The payoff: two remote/no-endpoint spaces under models_dir resolve to
        # TWO `remote` backends, each pinned to its own model, both pointed at
        # ONE loopback router endpoint — never two `llama` spawns.
        config = {
            "embedders": [
                {"modalities": ["text"], "runtime": "remote", "model": "text-a.gguf"},
                {"modalities": ["text"], "runtime": "remote", "model": "text-b.gguf"},
            ],
            "managed": {
                "llama_server": {"models_dir": "/models/router", "port": 8500, "models_max": 2}
            },
        }
        plan = _resolve(config)
        dicts = plan_to_runtime_params_set(plan)
        assert [d["backend"] for d in dicts] == ["remote", "remote"]
        # Both hit the SAME loopback endpoint (one shared router on the port).
        assert {d["endpoint"] for d in dicts} == {"http://127.0.0.1:8500"}
        # Each pins its OWN model (the request-routing key + space discriminator).
        assert [d["model"] for d in dicts] == ["text-a.gguf", "text-b.gguf"]
        assert [d["router_model"] for d in dicts] == ["text-a.gguf", "text-b.gguf"]
        # Each carries the identical shared-spawn router sub-map.
        assert dicts[0]["router"] == dicts[1]["router"]
        assert dicts[0]["router"]["models_dir"] == "/models/router"
        assert dicts[0]["router"]["models_max"] == 2
        assert dicts[0]["router"]["port"] == 8500
        # Never a `llama` (spawn-my-own) backend in router mode.
        assert all(d["backend"] != "llama" for d in dicts)

    def test_single_router_consumer_is_valid(self):
        # models_dir with ONE consumer is fine — a router serving one model.
        config = {
            "embedders": [{"modalities": ["text"], "runtime": "remote", "model": "only.gguf"}],
            "managed": {"llama_server": {"models_dir": "/models/router", "port": 8500}},
        }
        params = plan_to_runtime_params(_resolve(config))
        assert params["backend"] == "remote"
        assert params["endpoint"] == "http://127.0.0.1:8500"
        assert params["model"] == "only.gguf"
        assert params["router"]["models_dir"] == "/models/router"
        # models_max unset → None in the sub-map (server default).
        assert params["router"]["models_max"] is None

    def test_n1_single_managed_is_byte_unchanged_no_router(self):
        # BOUNDARY #1: a single managed-server config (no models_dir) still maps
        # to exactly ONE `llama` backend with NO router sub-map — today's
        # behavior, untouched. This is the N=1 proof.
        config = {
            "embedders": [
                {
                    "modalities": ["text"],
                    "runtime": "remote",
                    "model": "~/m.gguf",
                    "pooling": "last",
                }
            ],
            "managed": {"llama_server": {"binary": "/opt/llama-server", "port": 8474}},
        }
        params = plan_to_runtime_params(_resolve(config))
        assert params["backend"] == "llama"
        assert params["llama_server"] == "/opt/llama-server"
        assert params["port"] == 8474
        assert params["pooling"] == "last"
        # The new keys are ABSENT on the single-managed shape (not None) — the
        # dict is the pre-#567 shape verbatim.
        assert "router" not in params
        assert "router_model" not in params

    def test_router_consumer_without_a_model_is_rejected(self):
        # The model is the routing key into the directory — a router consumer
        # without one can't be routed. Named error.
        config = {
            "embedders": [{"modalities": ["text"], "runtime": "remote"}],
            "managed": {"llama_server": {"models_dir": "/models/router"}},
        }
        with pytest.raises(ProfileError, match="declare no model"):
            _resolve(config)

    def test_two_router_consumers_naming_the_same_model_are_rejected(self):
        # BOUNDARY #3: the model disambiguates BOTH the request routing and the
        # vector space, so two consumers naming the same model is one
        # indistinguishable space → a config error, not a silent collapse.
        config = {
            "embedders": [
                {"modalities": ["text"], "runtime": "remote", "model": "dup.gguf"},
                {"modalities": ["text"], "runtime": "remote", "model": "dup.gguf"},
            ],
            "managed": {"llama_server": {"models_dir": "/models/router"}},
        }
        with pytest.raises(ProfileError, match="name the SAME model"):
            _resolve(config)

    def test_router_with_attach_is_rejected(self):
        # Router mode is Shrike-launched — meaningless on an attached server.
        config = {
            "embedders": [{"modalities": ["text"], "runtime": "remote", "model": "m.gguf"}],
            "managed": {"llama_server": {"manage": "attach", "models_dir": "/models/router"}},
        }
        with pytest.raises(ProfileError, match="cannot apply to manage: attach"):
            _resolve(config)

    def test_router_with_manage_off_is_rejected(self):
        # manage: off means no managed server at all, so a remote/no-endpoint
        # entry is rejected by the earlier per-entry check ("manage is off")
        # before the router-shape check even runs — a clearer error (the entry
        # can't consume an off server, router or not). Either way: rejected.
        config = {
            "embedders": [{"modalities": ["text"], "runtime": "remote", "model": "m.gguf"}],
            "managed": {"llama_server": {"manage": "off", "models_dir": "/models/router"}},
        }
        with pytest.raises(ProfileError, match="manage is off"):
            _resolve(config)

    def test_models_max_without_models_dir_is_rejected(self):
        # models_max is a router-only knob — set without models_dir it's a
        # silent no-op, which the no-cross-talk rule forbids.
        config = {
            "embedders": [{"modalities": ["text"], "runtime": "remote", "model": "m.gguf"}],
            "managed": {"llama_server": {"models_max": 4}},
        }
        with pytest.raises(ProfileError, match="models_max is a router knob"):
            _resolve(config)

    def test_router_parse_round_trips_the_fields(self):
        caps = parse_capabilities(
            {
                "embedders": [{"modalities": ["text"], "runtime": "remote", "model": "m.gguf"}],
                "managed": {"llama_server": {"models_dir": "/d", "models_max": 3, "port": 9001}},
            }
        )
        assert caps.managed_llama == ManagedLlama(
            manage="auto", models_dir="/d", models_max=3, port=9001
        )

    def test_router_plus_separate_endpoint_space_coexist(self):
        # Two router-shared spaces AND a separate cloud-endpoint space — the
        # router collapses only its own no-endpoint consumers; the endpoint
        # space is untouched.
        config = {
            "embedders": [
                {"modalities": ["text"], "runtime": "remote", "model": "r-a.gguf"},
                {"modalities": ["text"], "runtime": "remote", "model": "r-b.gguf"},
                {
                    "modalities": ["text"],
                    "runtime": "remote",
                    "model": "cloud-1",
                    "endpoint": "https://api.example.com/v1",
                },
            ],
            "managed": {"llama_server": {"models_dir": "/models/router", "port": 8500}},
        }
        dicts = plan_to_runtime_params_set(_resolve(config))
        assert [d["backend"] for d in dicts] == ["remote", "remote", "remote"]
        # The two router consumers share the loopback router; the endpoint space
        # keeps its own external endpoint and carries NO router sub-map.
        assert dicts[0]["endpoint"] == dicts[1]["endpoint"] == "http://127.0.0.1:8500"
        assert dicts[2]["endpoint"] == "https://api.example.com/v1"
        assert "router" in dicts[0] and "router" in dicts[1]
        assert "router" not in dicts[2]
