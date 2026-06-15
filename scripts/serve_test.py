"""Unit tests for the //scripts:serve launcher's pure logic (#565/#656).

Covers profile resolution, the path-free invariant, arg parsing, the
effective-config composition (dir-name → absolute path rewrite), the run-server
argv, and the --import MVP stub — all WITHOUT any model download, so this is the
non-manual target CI runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import serve


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


def test_unknown_profile_errors_with_available_list() -> None:
    with pytest.raises(SystemExit) as exc:
        serve.load_profile("does-not-exist")
    msg = str(exc.value)
    assert "does-not-exist" in msg
    # The committed text-onnx profile is named as available.
    assert "text-onnx" in msg


def test_profile_with_collection_key_is_rejected(tmp_path: Path) -> None:
    bad = serve._PROFILES_DIR / "_test_bad_collection.yml"
    bad.write_text("collection: /some/path.anki2\nembedders: []\n")
    try:
        with pytest.raises(SystemExit, match="path-free"):
            serve.load_profile("_test_bad_collection")
    finally:
        bad.unlink()


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
        ["--profile", "text-onnx", "--collection", "/c.anki2", "--cache-dir", "/c", "--log-dir", "/l"]
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
