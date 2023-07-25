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

class BaseCommand:
    def __init__(self, name = "", help = "", children = []):
        self.name = name
        self.help = help
        self.children = children

    def add_arguments(self, parser, **kwargs):
        pass

    def handle(self, args, **kwargs):
        pass

# class CommandInfo:
#     # CASE 1. Not the ending component, e.g. debug, surgeon. cmd_instance = None, children cannot be empty
#     # CASE 2. The ending component, e.g. benchmark, evalute. cmd_instance cannot be None, children = []
#     def __init__(self, cmd_name : str, cmd_instance, children = None):
#         self.cmd_name = cmd_name
#         self.cmd_instance = cmd_instance
#         self.children = children
#         if (cmd_instance is None and children is None) or (cmd_instance is not None and children is not None):
#             print(f"subcommand {cmd_name} is set incorrectly.")

def register_parser(parser, commands):
    if commands is None or (isinstance(commands, list) and len(commands) * [None] == commands):
        return
    subparsers = parser.add_subparsers(title="Command")
    for command in commands:
        if command is None:
            continue
        subparser = subparsers.add_parser(command.name, help=command.help)
        command.add_arguments(subparser)
        subparser.set_defaults(handle=command.handle)
        register_parser(subparser, command.children)


def load_command_info(entry_points : str):
    cmd_infos = []
    for entry_point in pkg_resources.iter_entry_points(entry_points):
        cmd_infos.append(entry_point.load()())

    if len(cmd_infos) == 1:
        return cmd_infos[0]
    elif len(cmd_infos) > 1:
        return cmd_infos

