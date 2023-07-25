# Copyright 2023 Huawei Technologies Co., Ltd

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

from components.parser.parser import load_command_info, BaseCommand

debug_sub_task = {}
for entry_point in pkg_resources.iter_entry_points('debug_sub_task'):
    debug_sub_task[entry_point.name] = entry_point.load()

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
# debug_cli_group = click.Group(context_settings=CONTEXT_SETTINGS, name="debug", 
#                               commands=debug_sub_task, no_args_is_help=True,
#                               short_help="Debug a wide variety of model issues")

help_info = "debug a wide variety of model issues"
debug_cmd_info = load_command_info('debug_sub_task', "debug", help_info)