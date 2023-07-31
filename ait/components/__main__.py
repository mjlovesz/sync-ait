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

import argparse

from components.debug import debug_cmd
from components.profile import profile_cmd
from components.transplt import transplt_cmd
from components.benchmark import benchmark_cmd
from components.analyze import analyze_cmd
from components.convert import convert_cmd
from components.parser.parser import register_parser


def main():
    subcommands = [debug_cmd, profile_cmd, transplt_cmd, benchmark_cmd, analyze_cmd, convert_cmd]
    parser = argparse.ArgumentParser(description="ait(Ascend Inference Tools), provides one-site debugging "
                                     "and optimization toolkit for inference use Ascend Devices")
    register_parser(parser, subcommands)
    parser.set_defaults(print_help=parser.print_help)
    args = parser.parse_args()

    if hasattr(args, 'handle'):
        args.handle(args)
    elif hasattr(args, "print_help"):
        args.print_help()

if __name__ == "__main__":
    main()