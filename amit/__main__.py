import click

from amit.debug.debug import debug_cli_group
cli = click.Group(commands=[debug_cli_group])

if __name__ == "__main__":
    cli()