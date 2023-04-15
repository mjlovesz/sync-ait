import click

from .benchmark.main_cli import benchmark_cli_enter

profile_cli_group = click.Group(name="profile", commands=[benchmark_cli_enter])

if __name__ == "__main__":
    profile_cli_group()