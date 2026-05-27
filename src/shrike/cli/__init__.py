from __future__ import annotations

from pathlib import Path

import click

from shrike.cli.client import ShrikeClient
from shrike.cli.config import DEFAULT_CONFIG_PATH, load_config, resolve_url


@click.group()
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
@click.version_option(package_name="shrike")
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

    ctx.obj["config"] = config
    ctx.obj["config_path"] = config_path
    ctx.obj["url"] = server_url
    ctx.obj["json"] = json_output
    ctx.obj["client"] = ShrikeClient(server_url, config=config)

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
from shrike.cli.type_cmd import type_group  # noqa: E402

cli.add_command(completion)
cli.add_command(embedding)
cli.add_command(index)
cli.add_command(server)
cli.add_command(info)
cli.add_command(note)
cli.add_command(type_group, name="type")
