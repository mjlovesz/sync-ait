# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pkg_resources
import argparse


class BaseCommand:
    def __init__(self, name = "", help = "", children = None):
        self.name = name
        self.help = help
        if not children:
            self.children = []
        else:
            self.children = children

    def add_arguments(self, parser, **kwargs):
        pass

    def handle(self, args, **kwargs):
        pass


def register_parser(parser, commands):
    if commands is None or (isinstance(commands, list) and len(commands) * [None] == commands):
        return
    subparsers = parser.add_subparsers(title="Command")
    for command in commands:
        if command is None:
            continue
        subparser = subparsers.add_parser(
            command.name, formatter_class=argparse.ArgumentDefaultsHelpFormatter, help=command.help
        )
        command.add_arguments(subparser)
        subparser.set_defaults(handle=command.handle)
        register_parser(subparser, command.children)


def load_command_instance(entry_points : str, name = None, help_info = None, CommandClass = None):
    cmd_instances = []
    for entry_point in pkg_resources.iter_entry_points(entry_points):
        cmd_instances.append(entry_point.load()())

    if len(cmd_instances) == 1:
        return cmd_instances[0]
    elif len(cmd_instances) > 1:
        if not isinstance(name, str) or not isinstance(help_info, str) or CommandClass is None:
            print(f"load subcommands from entry point {entry_points} failed, \
                  lack of name or help_info or subcommand class")
        else:
            return CommandClass(name, help_info, cmd_instances)

