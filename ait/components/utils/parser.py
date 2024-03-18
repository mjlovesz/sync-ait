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

import sys
import logging
import argparse
import pkg_resources

AIT_FAQ_HOME = "https://gitee.com/ascend/ait/wikis/Home"
MIND_STUDIO_LOGO = "[Powered by MindStudio]"

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class BaseCommand:
    '''
    for a ending command, a derived class need to be created inherienting this BaseCommand,
    then modify the add_arguments and handle method
    '''
    def __init__(self, name="", help_info="", children=None, has_handle=True, alias_name="", **kwargs):
        '''
        parameters:
            name: (string) name of the command
            help_info: (string) help infomation of the command
            children: (list[BaseCommand]) list of BaseCommand instance
            has_handle: (bool) boolean indicating whether this command has handle function
            kwargs: for extension in the future
        return:
            None
        '''
        self.name = name
        self.help_info = help_info
        if not children:
            self.children = []
        else:
            self.children = children
        self.alias_name = alias_name
        self.has_handle = has_handle

    def add_arguments(self, parser, **kwargs):
        '''
        parameters:
            parser: (argparse.ArgumentParser) parser to be added parameters
            kwargs: for extension in the future
        return:
            None
        '''
        pass

    def handle(self, args, **kwargs):
        '''
        parameters:
            args: (argparse.Namespace) argument aggregation
            kwargs: for extension in the future
        return:
            None
        '''
        pass


def register_parser(parser, commands):
    if commands is None or (isinstance(commands, list) and len(commands) * [None] == commands):
        return
    subparsers = parser.add_subparsers(title="Command")
    for command in commands:
        if command is None:
            continue
        cmd_alias_name = [command.alias_name] if command.alias_name else []
        subparser = subparsers.add_parser(
            command.name, formatter_class=argparse.ArgumentDefaultsHelpFormatter, help=command.help_info,
            aliases=cmd_alias_name,
            description=command.help_info + " " + MIND_STUDIO_LOGO
        )
        command.add_arguments(subparser)
        if command.has_handle:
            subparser.set_defaults(handle=command.handle)
        else:
            subparser.set_defaults(print_help=subparser.print_help)
        register_parser(subparser, command.children)


def load_command_instance(entry_points : str, name=None, help_info=None, derived_command=None):
    cmd_instances = []
    for entry_point in pkg_resources.iter_entry_points(entry_points):
        cmd_instances.append(entry_point.load()())

    if len(cmd_instances) == 1:
        return cmd_instances[0]
    elif len(cmd_instances) > 1:
        if not isinstance(name, str) or not isinstance(help_info, str) or derived_command is None:
            logger.warning("load subcommands from entry point %s failed, \
                           lack of name or help_info or subcommand class", entry_points)
        else:
            return derived_command(name, help_info, cmd_instances)
    return None