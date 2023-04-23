import click
import pkg_resources


debug_sub_task = {}
for entry_point in pkg_resources.iter_entry_points('debug_sub_task'):
    debug_sub_task[entry_point.name] = entry_point.load()


debug_cli_group = click.Group(name="debug", commands=debug_sub_task)