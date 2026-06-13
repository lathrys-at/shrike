"""Per-call collection routing through the action layer (#68, slice 2).

The selector resolves to a per-call CollectionBundle via the ActionContext
resolver; a single-collection server (no resolver) rejects a selector; an
unknown selector is a clean ToolInputError; and the selector lands in the
single completion log line.
"""

from __future__ import annotations

import logging

import pytest
from mcp.server.fastmcp import FastMCP

from shrike.actions import ActionContext, CollectionBundle, ToolInputError, build_actions


def _bundle(kharness) -> CollectionBundle:
    return CollectionBundle(
        wrapper=kharness.wrapper,
        index=None,
        derived=None,
        kernel=kharness.kernel,
        dedup_stats=None,
    )


class TestResolverPlumbing:
    def test_selector_is_passed_to_the_resolver(self, kharness):
        seen: list[str | None] = []

        async def resolver(selector: str | None) -> CollectionBundle:
            seen.append(selector)
            return _bundle(kharness)

        mcp = FastMCP("test")
        from shrike.tools import register_tools

        register_tools(mcp, kharness.wrapper, kernel=kharness.kernel, resolver=resolver)
        # A routed call carries the selector through to the resolver.
        kharness.call_tool(mcp, "collection_info", {"collection": "work"})
        assert seen == ["work"]

    def test_no_selector_resolves_to_default(self, kharness):
        seen: list[str | None] = []

        async def resolver(selector: str | None) -> CollectionBundle:
            seen.append(selector)
            return _bundle(kharness)

        mcp = FastMCP("test")
        from shrike.tools import register_tools

        register_tools(mcp, kharness.wrapper, kernel=kharness.kernel, resolver=resolver)
        kharness.call_tool(mcp, "collection_info", {})
        # None selector → the resolver is asked for the default.
        assert seen == [None]

    def test_routed_call_operates_on_the_resolved_collection(self, kharness):
        # The resolved bundle's wrapper/kernel ARE what the action uses: a note
        # created through a routed upsert lands in the resolved collection.
        async def resolver(selector: str | None) -> CollectionBundle:
            return _bundle(kharness)

        mcp = FastMCP("test")
        from shrike.tools import register_tools

        register_tools(mcp, kharness.wrapper, kernel=kharness.kernel, resolver=resolver)
        res = kharness.call_tool(
            mcp,
            "upsert_notes",
            {
                "notes": [
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {"Front": "routed", "Back": "b"},
                    }
                ],
                "on_duplicate": "allow",
                "collection": "work",
            },
        )
        assert res["results"][0]["status"] == "created"
        notes = kharness.run(_find_all(kharness))
        assert len(notes) == 1


def _find_all(kharness):
    async def _go():
        return await kharness.wrapper.run(lambda c: c.find_notes(""))

    return _go()


class TestSingleCollectionRejectsSelector:
    def test_selector_without_resolver_is_input_error(self, kharness):
        # No resolver (single-collection / standalone): a selector has nothing
        # to route to → a clean ToolInputError (surfaced as an MCP isError).
        mcp = FastMCP("test")
        from shrike.tools import register_tools

        register_tools(mcp, kharness.wrapper, kernel=kharness.kernel)  # resolver=None
        with pytest.raises(Exception) as ei:
            kharness.call_tool(mcp, "collection_info", {"collection": "work"})
        assert "routing is not enabled" in str(ei.value)

    def test_no_selector_without_resolver_works(self, kharness):
        # The default path is unchanged: no resolver + no selector → the fixed
        # bundle, exactly as before #68.
        mcp = FastMCP("test")
        from shrike.tools import register_tools

        register_tools(mcp, kharness.wrapper, kernel=kharness.kernel)
        info = kharness.call_tool(mcp, "collection_info", {})
        assert "summary" in info


class TestUnknownSelector:
    def test_resolver_routing_error_becomes_input_error(self, kharness):
        async def resolver(selector: str | None) -> CollectionBundle:
            raise RuntimeError(f"unknown collection {selector!r}")

        ctx = ActionContext(wrapper=kharness.wrapper, kernel=kharness.kernel, resolver=resolver)
        actions = {a.name: a for a in build_actions(ctx)}
        # collection_check takes only the selector — a clean target for the
        # resolver-failure path.
        check = actions["collection_check"].impl

        async def _go():
            with pytest.raises(ToolInputError, match="unknown collection"):
                await check(collection="ghost")

        kharness.run(_go())


class TestSelectorInLogLine:
    def test_completion_log_line_includes_the_selector(self, kharness, caplog):
        async def resolver(selector: str | None) -> CollectionBundle:
            return _bundle(kharness)

        mcp = FastMCP("test")
        from shrike.tools import register_tools

        register_tools(mcp, kharness.wrapper, kernel=kharness.kernel, resolver=resolver)
        with caplog.at_level(logging.INFO, logger="shrike.tools"):
            kharness.call_tool(mcp, "collection_info", {"collection": "work"})
        # _safe_tool logs the call params; the selector must appear so the
        # per-call log line records which collection the op ran against.
        line = "\n".join(r.getMessage() for r in caplog.records)
        assert "collection_info" in line
        assert "work" in line
