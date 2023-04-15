import click

from .compare.main_cli import compare_cli_enter
from .surgeon.main_cli import surgeon_cmd_group

debug_cli_group = click.Group(name="debug", commands=[compare_cli_enter, surgeon_cmd_group])

if __name__ == "__main__":
    debug_cli_group()