# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import click

from components.parser.parser import CommandInfo
from app_analyze.utils import log_util
from app_analyze.porting.app import start_scan_kit, opt_source, opt_tools, opt_log_level, opt_report_type


class Args:
    def __init__(self, source, report_type, log_level, tools):
        self.source = source
        self.report_type = report_type
        self.log_level = log_level
        self.tools = tools


@click.command(short_help='Transplant tool to analyze inference applications', no_args_is_help=True)
@opt_source
@opt_report_type
@opt_log_level
@opt_tools
def cli(source, report_type, log_level, tools):
    args = Args(source, report_type, log_level, tools)
    log_util.set_logger_level(args.log_level)
    log_util.init_file_logger()
    start_scan_kit(args)


if __name__ == '__main__':
    cli()

class TranspltCommand:
    def add_arguments(self, parser):
        parser.add_argument("-om", "--om-model", required=True, default=None, help="the path of the om model")
        parser.add_argument("-i", "--input", default=None, help="the path of the input file or dir")
        parser.add_argument("-o", "--output", default=None, help="the path of the output dir")

    def handle(self, args):
        print(vars(args))
        print("hello from transplt")

def get_cmd_info():
    cmd_instance = TranspltCommand()
    return CommandInfo("transplt", cmd_instance)