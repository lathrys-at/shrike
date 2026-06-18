from __future__ import annotations

import click
from click.shell_completion import get_completion_class

SHELLS = ("bash", "zsh", "fish")


@click.command("completion", short_help="Generate shell completion script")
@click.argument("shell", type=click.Choice(SHELLS))
def completion(shell: str) -> None:
    """Generate a shell completion script for shrike.

    \b
    Supported shells: bash, zsh, fish

    \b
    Quick setup:
      eval "$(shrike completion zsh)"     # zsh (add to ~/.zshrc)
      eval "$(shrike completion bash)"    # bash (add to ~/.bashrc)
      shrike completion fish > ~/.config/fish/completions/shrike.fish
    """
    from shrike.cli import cli as cli_group

    comp_cls = get_completion_class(shell)
    if comp_cls is None:
        raise click.ClickException(f"Unsupported shell: {shell}")

    comp = comp_cls(cli_group, {}, "shrike", "_SHRIKE_COMPLETE")
    click.echo(comp.source())
