"""``shrike profile`` — the collection/profile registry.

Register collections by a friendly name, set an active default, and list them.
The registry lives in ``config.yml`` (a ``profiles:`` section) and is managed
entirely client-side — these commands never talk to the server. A registered
name is a handle for humans and the per-call routing selector; the active
default is the config-level fallback that selector resolves to when no selector
is passed. Neither is a server runtime switch.

Distinct from the capability/build profiles (``shrike.profiles``): those decide
*how vectors are produced*; these decide *where the notes live*.
"""

from __future__ import annotations

import click

from shrike.cli import output
from shrike.cli.config import save_config
from shrike.cli.groups import OrderedGroup
from shrike.cli.output import output_options
from shrike.harness.registry import Registry, RegistryError


def _save(ctx: click.Context, registry: Registry) -> None:
    """Persist a mutated registry back to the config file the CLI was given."""
    config = ctx.obj["config"]
    config_path = ctx.obj["config_path"]
    registry.apply_to_config(config)
    save_config(config, config_path)


@click.group("profile", cls=OrderedGroup, short_help="Manage the collection/profile registry")
def profile() -> None:
    """Register collections by name and pick an active default.

    A profile maps a friendly name to a collection (.anki2) path — Shrike's
    superset of Anki's profiles, so any collection path qualifies, not only
    ones under Anki's base directory. The active default is the collection
    used when no profile is selected.

    \b
    Examples:
      shrike profile create work ~/Anki2/Work/collection.anki2 --default
      shrike profile list
      shrike profile default personal
      shrike profile rename work job
      shrike profile delete old
    """


@profile.command("create", short_help="Register a collection under a name")
@output_options
@click.argument("name")
@click.argument("path", type=click.Path())
@click.option("--default", "make_default", is_flag=True, help="Make this the active default.")
@click.pass_context
def profile_create(ctx: click.Context, name: str, path: str, make_default: bool) -> None:
    """Register the collection at PATH under NAME.

    The path is expanded and absolutized but is not required to exist yet (a
    collection can be created later). The first profile registered becomes the
    default automatically; pass --default to make a later one the default.

    \b
    Examples:
      shrike profile create work ~/Anki2/Work/collection.anki2
      shrike profile create personal /data/anki/personal.anki2 --default
    """
    config = ctx.obj["config"]
    registry = Registry.from_config(config)
    try:
        added = registry.add(name, path, make_default=make_default)
    except RegistryError as err:
        raise click.ClickException(str(err)) from err
    _save(ctx, registry)

    if ctx.obj["json"]:
        output.emit_json({"name": added.name, "path": added.path, "default": registry.default})
        return
    suffix = " [dim](default)[/dim]" if registry.default == added.name else ""
    output.console.print(
        f"[green]+[/green] Registered profile [cyan]{added.name}[/cyan] "
        f"-> [cyan]{added.path}[/cyan]{suffix}"
    )


@profile.command("rename", short_help="Rename a registered profile")
@output_options
@click.argument("old")
@click.argument("new")
@click.pass_context
def profile_rename(ctx: click.Context, old: str, new: str) -> None:
    """Rename the registered profile OLD to NEW.

    The collection path and any per-profile overrides are kept; only the handle
    changes. The active default follows the rename if it named OLD. This is a
    pure config edit — the collection file and its search index are untouched
    (the index keys on the collection path, never the profile name).

    \b
    Examples:
      shrike profile rename work job
    """
    config = ctx.obj["config"]
    registry = Registry.from_config(config)
    try:
        renamed = registry.rename(old, new)
    except RegistryError as err:
        raise click.ClickException(str(err)) from err
    _save(ctx, registry)

    if ctx.obj["json"]:
        output.emit_json({"renamed": renamed.name, "from": old, "default": registry.default})
        return
    output.console.print(
        f"[yellow]~[/yellow] Renamed profile [cyan]{old}[/cyan] -> [cyan]{renamed.name}[/cyan]"
    )


@profile.command("delete", short_help="Unregister a profile")
@output_options
@click.argument("name")
@click.pass_context
def profile_delete(ctx: click.Context, name: str) -> None:
    """Unregister the profile NAME.

    Deleting the current default clears it (unless one profile remains, which
    then becomes the default). The collection file itself is never touched.

    \b
    Examples:
      shrike profile delete old
    """
    config = ctx.obj["config"]
    registry = Registry.from_config(config)
    try:
        removed = registry.remove(name)
    except RegistryError as err:
        raise click.ClickException(str(err)) from err
    _save(ctx, registry)

    if ctx.obj["json"]:
        output.emit_json({"removed": removed.name, "default": registry.default})
        return
    output.console.print(f"[red]-[/red] Removed profile [cyan]{removed.name}[/cyan]")


@profile.command("default", short_help="Set the active default profile")
@output_options
@click.argument("name")
@click.pass_context
def profile_default(ctx: click.Context, name: str) -> None:
    """Make the already-registered profile NAME the active default.

    This writes the config-level default — the persistent fallback the per-call
    selector resolves to when no profile is selected. It is not a server
    runtime switch.

    \b
    Examples:
      shrike profile default personal
    """
    config = ctx.obj["config"]
    registry = Registry.from_config(config)
    try:
        chosen = registry.set_default(name)
    except RegistryError as err:
        raise click.ClickException(str(err)) from err
    _save(ctx, registry)

    if ctx.obj["json"]:
        output.emit_json({"default": chosen.name})
        return
    output.console.print(f"Default profile is now [cyan]{chosen.name}[/cyan]")


@profile.command("list", short_help="List registered profiles")
@output_options
@click.option(
    "--discover",
    is_flag=True,
    help="Scan Anki's base directory for profiles instead of listing the registry.",
)
@click.pass_context
def profile_list(ctx: click.Context, discover: bool) -> None:
    """List registered profiles, marking the active default.

    With --discover, scan Anki's base directory (its prefs21.db) for the
    profiles Anki knows about, annotating which are already registered with
    Shrike — so you can register them without hunting for paths. Discovery is
    read-only and never touches Anki's files.

    \b
    Examples:
      shrike profile list
      shrike profile list --json
      shrike profile list --discover
    """
    config = ctx.obj["config"]
    registry = Registry.from_config(config)

    if discover:
        _list_discovered(ctx, registry)
        return

    if ctx.obj["json"]:
        output.emit_json(
            {
                "profiles": [{"name": p.name, "path": p.path} for p in registry.profiles],
                "default": registry.default,
            }
        )
        return

    if not registry.profiles:
        output.console.print(
            "[dim]No profiles registered. Add one with "
            "[/dim][cyan]shrike profile create <name> <path>[/cyan][dim].[/dim]"
        )
        return

    rows = [
        [
            "[green]*[/green]" if p.name == registry.default else " ",
            f"[cyan]{p.name}[/cyan]",
            f"[cyan]{p.path}[/cyan]",
        ]
        for p in registry.profiles
    ]
    output.table(["", "Name", "Collection"], rows)
    if registry.default:
        output.console.print(f"\n[dim]* active default: [/dim][cyan]{registry.default}[/cyan]")


def _list_discovered(ctx: click.Context, registry: Registry) -> None:
    """Render Anki-base-dir discovery (`profile list --discover`)."""
    import os

    from shrike.platform.paths import anki_base_dir, discover_anki_profiles

    base = anki_base_dir()
    discovered = discover_anki_profiles(base)
    # Membership is path-based, not name-based: a collection registered under a
    # different friendly name still reads as known. Compare on the same
    # normalized (abspath + expanduser) form the registry stores.
    registered = {p.path for p in registry.profiles}

    def _is_registered(coll_path: str) -> bool:
        return os.path.abspath(os.path.expanduser(coll_path)) in registered

    if ctx.obj["json"]:
        output.emit_json(
            {
                "base_dir": str(base),
                "profiles": [
                    {
                        "name": p.name,
                        "path": p.collection_path,
                        "exists": p.exists,
                        "registered": _is_registered(p.collection_path),
                    }
                    for p in discovered
                ],
            }
        )
        return

    if not discovered:
        output.console.print(
            f"[dim]No Anki profiles found under [/dim][cyan]{base}[/cyan][dim] "
            "(no prefs21.db, or Anki's base directory is elsewhere — set ANKI_BASE, "
            "or register collections manually with [/dim]"
            "[cyan]shrike profile create[/cyan][dim]).[/dim]"
        )
        return

    output.console.print(f"[dim]Anki profiles under [/dim][cyan]{base}[/cyan]\n")
    rows = []
    for p in discovered:
        marks = []
        if _is_registered(p.collection_path):
            marks.append("[green]registered[/green]")
        if not p.exists:
            marks.append("[yellow]missing[/yellow]")
        rows.append(
            [
                f"[cyan]{p.name}[/cyan]",
                f"[cyan]{p.collection_path}[/cyan]",
                " ".join(marks),
            ]
        )
    output.table(["Name", "Collection", ""], rows)
    output.console.print(
        "\n[dim]Register one with [/dim]"
        "[cyan]shrike profile create <name> <path>[/cyan][dim].[/dim]"
    )
