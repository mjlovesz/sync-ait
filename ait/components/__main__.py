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

import click
import pkg_resources
import argparse

from components.debug import debug_cmd_info
from components.profile import profile_cmd_info
from components.transplt import transplt_cmd_info
from components.benchmark import benchmark_cmd_info
from components.analyze import analyze_cmd_info
from components.convert import convert_cmd_info

def register_parser(parser, command_infos):
    if command_infos is []:
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

if __name__ == "__main__":
    subcommand_infos = [debug_cmd_info, profile_cmd_info, transplt_cmd_info,
                        benchmark_cmd_info, analyze_cmd_info, convert_cmd_info]
    parser = argparse.ArgumentParser()
    register_parser(parser, subcommand_infos)
    args = parser.parse_args()

    if hasattr(args, 'handle'):
        args.handle(args)
