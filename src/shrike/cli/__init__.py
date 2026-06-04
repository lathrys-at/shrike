from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from shrike import __version__
from shrike.cli.config import DEFAULT_CONFIG_PATH, build_server_spec, load_config, resolve_url
from shrike.client import ShrikeClient, ShrikeError


class ShrikeGroup(click.Group):
    """Root group that turns library ``ShrikeError``s into clean CLI errors.

    Keeps the standalone client free of ``click`` while giving the CLI a single
    place to render server/connection failures (instead of tracebacks).
    """

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
    # targeting a remote server.
    spec = build_server_spec(config)

    ctx.obj["config"] = config
    ctx.obj["config_path"] = config_path
    ctx.obj["url"] = server_url
    ctx.obj["json"] = json_output
    ctx.obj["client"] = ShrikeClient(server_url, spec=spec)

    from shrike.cli import output

    if json_output:
        pretty = False
    ctx.obj["pretty"] = pretty
    output.set_pretty(pretty)


# Register subcommands
from shrike.cli.completion_cmd import completion  # noqa: E402
from shrike.cli.embedding_cmd import embedding  # noqa: E402
from shrike.cli.index_cmd import index  # noqa: E402
from shrike.cli.info_cmd import info  # noqa: E402
from shrike.cli.note_cmd import note  # noqa: E402
from shrike.cli.server_cmd import server  # noqa: E402
from shrike.cli.tag_cmd import tag  # noqa: E402
from shrike.cli.type_cmd import type_group  # noqa: E402

cli.add_command(completion)
cli.add_command(embedding)
cli.add_command(index)
cli.add_command(server)
cli.add_command(info)
cli.add_command(note)
cli.add_command(tag)
cli.add_command(type_group, name="type")
