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

# class Command:
#     def add_arguments(self, parser):
#         pass

#     def handle(self, args):
#         pass

class CommandInfo:
    # CASE 1. Not the ending component, e.g. debug, surgeon. cmd_instance = None, children cannot be empty
    # CASE 2. The ending component, e.g. benchmark, evalute. cmd_instance cannot be None, children = []
    def __init__(self, cmd_name : str, cmd_instance, children = None):
        self.cmd_name = cmd_name
        self.cmd_instance = cmd_instance
        self.children = children
        if (cmd_instance is None and children is None) or (cmd_instance is not None and children is not None):
            print(f"subcommand {cmd_name} is set incorrectly.")

def register_parser(parser, command_infos):
    if command_infos is None or (isinstance(command_infos, list) and len(command_infos) * [None] == command_infos):
        return
    subparsers = parser.add_subparsers(title="Command", help="general help")
    for cmd_info in command_infos:
        if cmd_info is None:
            continue
        subparser = subparsers.add_parser(cmd_info.cmd_name)
        if cmd_info.cmd_instance is not None:
            cmd_info.cmd_instance.add_arguments(subparser)
            subparser.set_defaults(handle=cmd_info.cmd_instance.handle)
        register_parser(subparser, cmd_info.children)


def load_command_info(entry_points : str, name : str):
    cmd_infos = []
    for entry_point in pkg_resources.iter_entry_points(entry_points):
        cmd_infos.append(entry_point.load()())
    
    if len(cmd_infos) == 1:
        return cmd_infos[0]
    elif len(cmd_infos) > 1:
        return CommandInfo(name, None, cmd_infos)

