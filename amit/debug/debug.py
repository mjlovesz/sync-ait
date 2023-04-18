import click

from debug.compare.main_cli import compare_cli_enter
from debug.surgeon.main_cli import surgeon_cmd_group

debug_cli_group = click.Group(name="debug", commands=[compare_cli_enter, surgeon_cmd_group])
