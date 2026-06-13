"""The list_profiles enumeration action (#66, slice 3).

Read-only, host-side: it reflects the registry snapshot in the ActionContext,
opens no collection. Invoked via the action impl directly (no MCP/HTTP needed
for an enumeration that touches neither the kernel nor the wrapper).
"""

from __future__ import annotations

from shrike.actions import ActionContext, build_actions
from shrike.registry import Registry


def _list_profiles_action(kharness, registry):
    ctx = ActionContext(wrapper=kharness.wrapper, kernel=kharness.kernel, registry=registry)
    actions = {a.name: a for a in build_actions(ctx)}
    return actions["list_profiles"]


async def test_lists_registered_profiles_and_default(kharness):
    reg = Registry()
    reg.add("work", "/decks/work.anki2")
    reg.add("home", "/decks/home.anki2", make_default=True)
    action = _list_profiles_action(kharness, reg)

    resp = await action.impl()
    assert [p.name for p in resp.profiles] == ["work", "home"]
    assert resp.default == "home"
    by_name = {p.name: p for p in resp.profiles}
    assert by_name["home"].is_default is True
    assert by_name["work"].is_default is False
    assert by_name["work"].path == "/decks/work.anki2"


async def test_empty_registry(kharness):
    action = _list_profiles_action(kharness, Registry())
    resp = await action.impl()
    assert resp.profiles == []
    assert resp.default is None


async def test_none_registry_reports_empty(kharness):
    # An absent registry snapshot (server with no config) → empty, not an error.
    action = _list_profiles_action(kharness, None)
    resp = await action.impl()
    assert resp.profiles == []
    assert resp.default is None


async def test_no_default_set(kharness):
    # Default removed among several profiles → default None, none flagged.
    reg = Registry()
    reg.add("a", "/a.anki2")
    reg.add("b", "/b.anki2")
    reg.add("c", "/c.anki2")
    reg.remove("a")  # was the default; 2 remain → no default
    action = _list_profiles_action(kharness, reg)
    resp = await action.impl()
    assert resp.default is None
    assert all(not p.is_default for p in resp.profiles)
