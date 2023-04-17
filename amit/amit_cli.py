import click

from debug.__main__ import debug_cli_group
from profile.__main__ import profile_cli_group

cli = click.Group(commands=[debug_cli_group, profile_cli_group])

if __name__ == "__main__":
    cli()