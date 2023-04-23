import click

from amit.debug.debug import debug_cli_group
from amit.profile.profile import profile_cli_group

cli = click.Group(commands=[debug_cli_group, profile_cli_group])

if __name__ == "__main__":
    cli()