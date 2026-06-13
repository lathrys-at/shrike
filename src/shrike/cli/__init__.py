from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import click

from shrike import __version__
from shrike.cli.config import DEFAULT_CONFIG_PATH, build_server_spec, load_config, resolve_url
from shrike.errors import ShrikeError

# Subcommand name -> "module:attribute", imported only when the command is
# actually invoked. Loading every command module up front pulled in httpx +
# Pydantic on every `shrike` call (tab-completion, --help, --version included);
# this defers each module's import to the one command that needs it. Each group
# (note, server, type, …) lives in a single module, so lazy-loading at this top
# level already defers a whole group's subcommands — there's nothing finer to
# split below it.
_LAZY_COMMANDS: dict[str, str] = {
    "collection": "shrike.cli.collection_cmd:collection",
    "completion": "shrike.cli.completion_cmd:completion",
    "deck": "shrike.cli.deck_cmd:deck",
    "embedding": "shrike.cli.embedding_cmd:embedding",
    "export": "shrike.cli.export_cmd:export",
    "import": "shrike.cli.import_cmd:import_cmd",
    "index": "shrike.cli.index_cmd:index",
    "info": "shrike.cli.info_cmd:info",
    "media": "shrike.cli.media_cmd:media",
    "note": "shrike.cli.note_cmd:note",
    "profile": "shrike.cli.profile_cmd:profile",
    "server": "shrike.cli.server_cmd:server",
    "tag": "shrike.cli.tag_cmd:tag",
    "type": "shrike.cli.type_cmd:type_group",
}


class ShrikeGroup(click.Group):
    """Root group: lazy-loads subcommands and turns library ``ShrikeError``s into
    clean CLI errors.

    Subcommand modules are imported on demand (see ``_LAZY_COMMANDS``) so a bare
    `shrike` invocation stays cheap. ``ShrikeError`` is caught here — from the
    dependency-light ``shrike.errors`` — to render server/connection failures as
    clean messages instead of tracebacks, without the standalone client needing
    ``click``.
    """

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(_LAZY_COMMANDS)

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        target = _LAZY_COMMANDS.get(name)
        if target is None:
            return None
        module_name, attr = target.rsplit(":", 1)
        command: click.Command = getattr(importlib.import_module(module_name), attr)
        return command

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except ShrikeError as err:
            raise click.ClickException(str(err)) from err


@click.group(cls=ShrikeGroup)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_CONFIG_PATH,
    envvar="SHRIKE_CONFIG",
    help="Path to config file.",
    show_default=True,
)
@click.option(
    "--url",
    envvar="SHRIKE_URL",
    default=None,
    help="Server URL (overrides config). [env: SHRIKE_URL]",
)
@click.option(
    "--profile",
    "--collection",
    "profile",
    envvar="SHRIKE_PROFILE",
    default=None,
    help=(
        "Route commands to this registered collection profile (see "
        "'shrike profile list'). Defaults to the active profile. "
        "[env: SHRIKE_PROFILE]"
    ),
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output raw JSON instead of formatted text.",
)
@click.option(
    "--pretty/--no-pretty",
    default=True,
    help="Styled output (default: --pretty).",
)
@click.version_option(version=__version__, prog_name="shrike")
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: Path,
    url: str | None,
    profile: str | None,
    json_output: bool,
    pretty: bool,
) -> None:
    """Shrike — manage your Anki collection from the command line.

    \b
    Quick start:
      shrike server start --collection ~/path/to/collection.anki2
      shrike info
      shrike note list --deck Default
      shrike server stop

    \b
    Configuration file location is platform-dependent (use --help to see
    the default for your system). Override with -c/--config or SHRIKE_CONFIG.
    """
    ctx.ensure_object(dict)

    config = load_config(config_path)
    server_url = resolve_url(config, url)

    # The client auto-starts a local daemon from this spec on connection
    # failure. None (no collection configured) disables auto-start — e.g. when
    # targeting a remote server. A capability config error (#498) disables
    # auto-start with a warning rather than breaking every command — commands
    # against an already-running or remote server still work; `shrike server
    # start` surfaces the same error loudly.
    from shrike.profiles import ProfileError

    try:
        spec = build_server_spec(config, config_path=config_path)
    except ProfileError as e:
        click.echo(f"warning: {e} (auto-start disabled)", err=True)
        spec = None

    # Imported here, not at module top, so commands that never reach this
    # callback (tab-completion, --help, --version) don't pull in httpx/Pydantic.
    from shrike.client import ShrikeClient

    ctx.obj["config"] = config
    ctx.obj["config_path"] = config_path
    ctx.obj["url"] = server_url
    ctx.obj["json"] = json_output
    ctx.obj["profile"] = profile
    # The client injects --profile/--collection into every routed tool call (#68);
    # None uses the server's active default.
    ctx.obj["client"] = ShrikeClient(server_url, spec=spec, collection=profile)

    from shrike.cli import output

    if json_output:
        pretty = False
    ctx.obj["pretty"] = pretty
    output.set_pretty(pretty)


# Subcommands are registered lazily — see ``ShrikeGroup`` / ``_LAZY_COMMANDS``.
