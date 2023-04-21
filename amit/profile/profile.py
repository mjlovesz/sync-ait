import click
import pkg_resources
from main_cli import benchmark_cli_enter

profile_sub_task = {}
for entry_point in pkg_resources.iter_entry_points('debug_sub_task'):
    debug_sub_task[entry_point.name] = entry_point.load()

profile_cli_group = click.Group(name="profile", commands=profile_sub_task)