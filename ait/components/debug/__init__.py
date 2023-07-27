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

from components.parser.parser import load_command_instance, BaseCommand

class DebugCommand(BaseCommand):
    def __init__(self, name="", help="", children=None):
        super().__init__(name, help, children)

    def add_arguments(self, parser, **kwargs):
        return super().add_arguments(parser, **kwargs)

    def handle(self, args, **kwargs):
        return super().handle(args, **kwargs)

help_info = "debug a wide variety of model issues"
debug_cmd = load_command_instance('debug_sub_task', "debug", help_info, DebugCommand)