"""Multi-collection routing — the CollectionManager (#68, slice 1).

Each registered collection routes to its own lazily-assembled Harness (own
AsyncKernel + namespaced index + per-collection derived store), sharing one
base cache dir and one embedding runtime. These open real kernels, so they
live in tests/native.
"""

from __future__ import annotations

import asyncio

import pytest

shrike_native = pytest.importorskip("shrike_native")

from shrike.harness import cache_layout  # noqa: E402
from shrike.cli.config import load_config, save_config  # noqa: E402
from shrike.harness.derived import DerivedTextStore, NativeDerivedEngine  # noqa: E402
from shrike.harness.engines.embedding.runtime import EmbeddingRuntime  # noqa: E402
from shrike.harness.harness import (  # noqa: E402
    CollectionManager,
    Harness,
    HarnessParams,
    RoutingError,
)
from shrike.harness.registry import Registry  # noqa: E402


async def _default_harness(cache_dir, default_path, runtime):
    derived = DerivedTextStore(path=cache_dir / "shrike.db", engine_factory=NativeDerivedEngine)
    harness = await Harness.assemble(
        collection_path=str(default_path),
        cache_dir=str(cache_dir),
        runtime=runtime,
        derived=derived,
        cooperative=False,
        hold_seconds=5.0,
        media_read=None,
        media_exists=None,
    )
    await harness.boot(start_embedding=False)
    return harness


def _write_registry(config_path, entries, default=None):
    reg = Registry()
    for name, path in entries:
        reg.add(name, str(path), make_default=(name == default))
    config = load_config(config_path)
    reg.apply_to_config(config)
    save_config(config, config_path)


def _manager(tmp_path, config_path, default_harness, runtime):
    return CollectionManager(
        params=HarnessParams(
            cache_dir=str(tmp_path / "cache"),
            runtime=runtime,
            media_read=None,
            media_exists=None,
            cooperative=False,
            hold_seconds=5.0,
        ),
        default_harness=default_harness,
        default_collection_path=str(tmp_path / "default.anki2"),
        config_path=config_path,
    )


class TestSelectorResolution:
    def test_no_selector_no_registry_routes_to_boot_collection(self, tmp_path) -> None:
        async def flow():
            runtime = EmbeddingRuntime(model=None)
            default = await _default_harness(
                tmp_path / "cache", tmp_path / "default.anki2", runtime
            )
            mgr = _manager(tmp_path, tmp_path / "config.yml", default, runtime)
            # No registry, no selector → the daemon's boot collection.
            h = await mgr.harness_for(None)
            assert h is default
            await mgr.close()

        asyncio.run(flow())

    def test_no_selector_uses_registry_default(self, tmp_path) -> None:
        async def flow():
            cfg = tmp_path / "config.yml"
            # The boot collection IS the registry default → routes to it.
            _write_registry(cfg, [("primary", tmp_path / "default.anki2")], default="primary")
            runtime = EmbeddingRuntime(model=None)
            default = await _default_harness(
                tmp_path / "cache", tmp_path / "default.anki2", runtime
            )
            mgr = _manager(tmp_path, cfg, default, runtime)
            key, path = mgr.resolve(None)
            assert key == "primary"
            assert path == str((tmp_path / "default.anki2").resolve()) or path.endswith(
                "default.anki2"
            )
            await mgr.close()

        asyncio.run(flow())

    def test_unknown_selector_is_routing_error(self, tmp_path) -> None:
        async def flow():
            runtime = EmbeddingRuntime(model=None)
            default = await _default_harness(
                tmp_path / "cache", tmp_path / "default.anki2", runtime
            )
            mgr = _manager(tmp_path, tmp_path / "config.yml", default, runtime)
            with pytest.raises(RoutingError, match="unknown collection"):
                mgr.resolve("ghost")
            await mgr.close()

        asyncio.run(flow())

    def test_no_default_set_among_several_resolves_to_boot_collection(self, tmp_path) -> None:
        # The #66-review case: with profiles registered but NO default set, a
        # bare (no-selector) call falls back to the daemon's boot collection
        # rather than erroring — the boot collection is the implicit default.
        async def flow():
            cfg = tmp_path / "config.yml"
            # Two registered profiles, neither the boot collection, no default.
            reg = Registry()
            reg.add("a", str(tmp_path / "a.anki2"))
            reg.add("b", str(tmp_path / "b.anki2"))
            reg.default = None
            config = load_config(cfg)
            reg.apply_to_config(config)
            save_config(config, cfg)

            runtime = EmbeddingRuntime(model=None)
            default = await _default_harness(
                tmp_path / "cache", tmp_path / "default.anki2", runtime
            )
            mgr = _manager(tmp_path, cfg, default, runtime)
            _, path = mgr.resolve(None)
            assert path.endswith("default.anki2")
            await mgr.close()

        asyncio.run(flow())


class TestLazyAssemblyAndIsolation:
    def test_routes_to_a_registered_collection_and_isolates_it(self, tmp_path) -> None:
        async def flow():
            cfg = tmp_path / "config.yml"
            other = tmp_path / "other.anki2"
            _write_registry(cfg, [("other", other)])
            runtime = EmbeddingRuntime(model=None)
            default = await _default_harness(
                tmp_path / "cache", tmp_path / "default.anki2", runtime
            )
            mgr = _manager(tmp_path, cfg, default, runtime)

            # Write a note into the DEFAULT collection.
            await default.wrapper.upsert_notes(
                [
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {"Front": "in default", "Back": "x"},
                    }
                ]
            )
            # Route to "other" — lazily assembled, a DISTINCT harness/kernel.
            h_other = await mgr.harness_for("other")
            assert h_other is not default
            await h_other.wrapper.upsert_notes(
                [
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {"Front": "in other", "Back": "y"},
                    }
                ]
            )

            # Isolation: each collection sees only its own note.
            default_notes = await default.wrapper.run(lambda c: c.find_notes(""))
            other_notes = await h_other.wrapper.run(lambda c: c.find_notes(""))
            assert len(default_notes) == 1
            assert len(other_notes) == 1

            # The routed collection's index + derived store live in DISTINCT
            # per-collection namespaces under the shared cache dir (#67/#547).
            base = str(tmp_path / "cache")
            assert cache_layout.collection_index_dir(
                base, str(tmp_path / "default.anki2")
            ) != cache_layout.collection_index_dir(base, str(other))
            await mgr.close()

        asyncio.run(flow())

    def test_routing_is_idempotent_same_harness(self, tmp_path) -> None:
        async def flow():
            cfg = tmp_path / "config.yml"
            _write_registry(cfg, [("other", tmp_path / "other.anki2")])
            runtime = EmbeddingRuntime(model=None)
            default = await _default_harness(
                tmp_path / "cache", tmp_path / "default.anki2", runtime
            )
            mgr = _manager(tmp_path, cfg, default, runtime)
            h1 = await mgr.harness_for("other")
            h2 = await mgr.harness_for("other")
            assert h1 is h2  # assembled once, cached
            await mgr.close()

        asyncio.run(flow())

    def test_concurrent_first_routes_assemble_once(self, tmp_path) -> None:
        async def flow():
            cfg = tmp_path / "config.yml"
            _write_registry(cfg, [("other", tmp_path / "other.anki2")])
            runtime = EmbeddingRuntime(model=None)
            default = await _default_harness(
                tmp_path / "cache", tmp_path / "default.anki2", runtime
            )
            mgr = _manager(tmp_path, cfg, default, runtime)
            # Two concurrent first-routes to the same collection must yield the
            # SAME harness (the per-key assembly lock prevents double-open).
            h1, h2 = await asyncio.gather(mgr.harness_for("other"), mgr.harness_for("other"))
            assert h1 is h2
            await mgr.close()

        asyncio.run(flow())


class TestStatusRows:
    def test_default_only_single_row(self, tmp_path) -> None:
        async def flow():
            runtime = EmbeddingRuntime(model=None)
            default = await _default_harness(
                tmp_path / "cache", tmp_path / "default.anki2", runtime
            )
            mgr = _manager(tmp_path, tmp_path / "config.yml", default, runtime)
            rows = mgr.status_rows()
            assert len(rows) == 1
            row = rows[0]
            assert row["is_default"] is True
            assert row["active"] is True  # the boot harness is assembled
            assert row["registered"] is False  # not a registered profile
            assert row["index_state"] is not None  # an assembled harness reports it
            await mgr.close()

        asyncio.run(flow())

    def test_registered_unrouted_collection_is_inactive_row(self, tmp_path) -> None:
        async def flow():
            cfg = tmp_path / "config.yml"
            _write_registry(cfg, [("other", tmp_path / "other.anki2")])
            runtime = EmbeddingRuntime(model=None)
            default = await _default_harness(
                tmp_path / "cache", tmp_path / "default.anki2", runtime
            )
            mgr = _manager(tmp_path, cfg, default, runtime)
            rows = {r["name"]: r for r in mgr.status_rows()}
            # The boot collection + the registered (but never-routed) "other".
            assert set(rows) == {mgr.DEFAULT_KEY, "other"}
            assert rows["other"]["registered"] is True
            assert rows["other"]["active"] is False  # never routed → not assembled
            assert rows["other"]["held"] is None
            assert rows["other"]["index_state"] is None
            await mgr.close()

        asyncio.run(flow())

    def test_routed_collection_becomes_active_row(self, tmp_path) -> None:
        async def flow():
            cfg = tmp_path / "config.yml"
            _write_registry(cfg, [("other", tmp_path / "other.anki2")])
            runtime = EmbeddingRuntime(model=None)
            default = await _default_harness(
                tmp_path / "cache", tmp_path / "default.anki2", runtime
            )
            mgr = _manager(tmp_path, cfg, default, runtime)
            await mgr.harness_for("other")  # route → assemble
            rows = {r["name"]: r for r in mgr.status_rows()}
            assert rows["other"]["active"] is True
            assert rows["other"]["index_state"] is not None
            await mgr.close()

        asyncio.run(flow())

    def test_boot_collection_that_is_a_registered_profile_dedupes(self, tmp_path) -> None:
        async def flow():
            cfg = tmp_path / "config.yml"
            # The boot collection IS registered as "primary" and is the default.
            _write_registry(cfg, [("primary", tmp_path / "default.anki2")], default="primary")
            runtime = EmbeddingRuntime(model=None)
            default = await _default_harness(
                tmp_path / "cache", tmp_path / "default.anki2", runtime
            )
            mgr = _manager(tmp_path, cfg, default, runtime)
            rows = mgr.status_rows()
            # One row (deduped by namespace), named by the registry, default+active.
            assert len(rows) == 1
            assert rows[0]["name"] == "primary"
            assert rows[0]["registered"] is True
            assert rows[0]["is_default"] is True
            await mgr.close()

        asyncio.run(flow())


class TestLiveRegistryView:
    def test_register_then_route_in_one_session(self, tmp_path) -> None:
        # Contract #2: the registry is a live view — a profile added to the
        # config AFTER the manager was built routes in the same session.
        async def flow():
            cfg = tmp_path / "config.yml"
            runtime = EmbeddingRuntime(model=None)
            default = await _default_harness(
                tmp_path / "cache", tmp_path / "default.anki2", runtime
            )
            mgr = _manager(tmp_path, cfg, default, runtime)
            # Initially unknown.
            with pytest.raises(RoutingError):
                mgr.resolve("late")
            # Register it now (as the CLI would), then route — no restart.
            _write_registry(cfg, [("late", tmp_path / "late.anki2")])
            key, _ = mgr.resolve("late")
            assert key == "late"
            h = await mgr.harness_for("late")
            assert h is not default
            await mgr.close()

        asyncio.run(flow())
