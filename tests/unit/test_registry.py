"""Registry model + config round-trip (#66, slice 1)."""

from __future__ import annotations

import dataclasses
import os

import pytest

from shrike.cli.config import load_config, save_config
from shrike.registry import CollectionProfile, Registry, RegistryError


class TestRegistryModel:
    def test_add_normalizes_path_and_sets_first_as_default(self):
        reg = Registry()
        p = reg.add("work", "~/decks/work.anki2")
        assert p.name == "work"
        assert p.path == os.path.abspath(os.path.expanduser("~/decks/work.anki2"))
        # First profile registered becomes the default implicitly.
        assert reg.default == "work"
        assert reg.resolve_default() is p

    def test_add_second_not_default_unless_requested(self):
        reg = Registry()
        reg.add("work", "/a/work.anki2")
        reg.add("home", "/a/home.anki2")
        assert reg.default == "work"
        reg.add("study", "/a/study.anki2", make_default=True)
        assert reg.default == "study"

    def test_add_duplicate_name_errors(self):
        reg = Registry()
        reg.add("work", "/a/work.anki2")
        with pytest.raises(RegistryError, match="already registered"):
            reg.add("work", "/other/path.anki2")

    def test_add_empty_name_or_path_errors(self):
        reg = Registry()
        with pytest.raises(RegistryError, match="name must not be empty"):
            reg.add("  ", "/a/x.anki2")
        with pytest.raises(RegistryError, match="path must not be empty"):
            reg.add("x", "   ")

    def test_remove_unknown_errors(self):
        reg = Registry()
        with pytest.raises(RegistryError, match="not registered"):
            reg.remove("ghost")

    def test_remove_clears_default_when_multiple_remain(self):
        reg = Registry()
        reg.add("a", "/a.anki2")
        reg.add("b", "/b.anki2")
        reg.add("c", "/c.anki2")  # default still "a"
        reg.remove("a")
        # Two remain, the default was removed → no dangling default.
        assert reg.default is None
        assert reg.resolve_default() is None

    def test_remove_promotes_sole_survivor_to_default(self):
        reg = Registry()
        reg.add("a", "/a.anki2")
        reg.add("b", "/b.anki2")  # default "a"
        reg.remove("a")
        assert reg.default == "b"

    def test_remove_non_default_keeps_default(self):
        reg = Registry()
        reg.add("a", "/a.anki2")
        reg.add("b", "/b.anki2")
        reg.remove("b")
        assert reg.default == "a"

    def test_set_default_unknown_errors(self):
        reg = Registry()
        reg.add("a", "/a.anki2")
        with pytest.raises(RegistryError, match="not registered"):
            reg.set_default("ghost")

    def test_set_default_switches(self):
        reg = Registry()
        reg.add("a", "/a.anki2")
        reg.add("b", "/b.anki2")
        reg.set_default("b")
        assert reg.default == "b"
        assert reg.resolve_default().name == "b"

    def test_per_profile_fields_modeled(self):
        reg = Registry()
        p = reg.add(
            "a",
            "/a.anki2",
            embedding={"model": "/m.gguf"},
            cache_dir="~/cache",
        )
        assert p.embedding == {"model": "/m.gguf"}
        assert p.cache_dir == os.path.abspath(os.path.expanduser("~/cache"))


class TestRegistrySerialization:
    def test_empty_registry_section_is_empty(self):
        assert Registry().to_config_section() == {}

    def test_round_trip_through_config_section(self):
        reg = Registry()
        reg.add("work", "/a/work.anki2")
        reg.add("home", "/a/home.anki2", embedding={"model": "/m.gguf"}, cache_dir="/c")
        reg.set_default("home")

        section = reg.to_config_section()
        # Insertion order preserved.
        assert [e["name"] for e in section["entries"]] == ["work", "home"]
        assert section["default"] == "home"
        # Optional fields only on the entry that set them.
        assert "embedding" not in section["entries"][0]
        assert section["entries"][1]["embedding"] == {"model": "/m.gguf"}
        assert section["entries"][1]["cache_dir"] == "/c"

        back = Registry.from_config({"profiles": section})
        assert back.names() == ["work", "home"]
        assert back.default == "home"
        assert back.get("home").embedding == {"model": "/m.gguf"}
        assert back.get("home").cache_dir == "/c"

    def test_from_config_drops_dangling_default(self):
        section = {"entries": [{"name": "a", "path": "/a.anki2"}], "default": "ghost"}
        reg = Registry.from_config({"profiles": section})
        assert reg.default is None

    def test_from_config_skips_malformed_entries(self):
        section = {
            "entries": [
                {"name": "", "path": "/a"},
                {"path": "/b"},
                {"name": "ok", "path": "/c"},
            ]
        }
        reg = Registry.from_config({"profiles": section})
        assert reg.names() == ["ok"]

    def test_from_config_tolerates_missing_section(self):
        reg = Registry.from_config({})
        assert reg.profiles == []
        assert reg.default is None

    def test_apply_to_config_clears_when_empty(self):
        reg = Registry()
        reg.add("a", "/a.anki2")
        reg.remove("a")
        config = {"profiles": {"entries": [{"name": "a", "path": "/a.anki2"}], "default": "a"}}
        reg.apply_to_config(config)
        assert config["profiles"] == {"entries": [], "default": None}


class TestRegistryConfigFile:
    """End-to-end through load_config / save_config (the persistence contract)."""

    def test_save_and_reload_registry(self, tmp_path):
        path = tmp_path / "config.yml"
        config = load_config(path)
        reg = Registry.from_config(config)
        reg.add("work", "/decks/work.anki2")
        reg.add("home", "/decks/home.anki2", make_default=True)
        reg.apply_to_config(config)
        save_config(config, path)

        reloaded = load_config(path)
        back = Registry.from_config(reloaded)
        assert back.names() == ["work", "home"]
        assert back.default == "home"

    def test_empty_registry_not_written(self, tmp_path):
        path = tmp_path / "config.yml"
        config = load_config(path)
        config["collection"] = "/some/collection.anki2"
        save_config(config, path)
        # No profiles touched → no profiles section persisted.
        assert "profiles:" not in path.read_text()

    def test_default_config_has_empty_registry(self, tmp_path):
        config = load_config(tmp_path / "nonexistent.yml")
        assert config["profiles"] == {"entries": [], "default": None}
        # Mutating the loaded copy must not poison the module default.
        config["profiles"]["entries"].append({"name": "x", "path": "/x"})
        again = load_config(tmp_path / "nonexistent.yml")
        assert again["profiles"]["entries"] == []


def test_collection_profile_is_frozen():
    p = CollectionProfile(name="a", path="/a.anki2")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.name = "b"  # type: ignore[misc]
