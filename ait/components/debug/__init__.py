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
import pkg_resources

from components.utils.parser import BaseCommand


class DebugCommand(BaseCommand):
    def __init__(self, name="", help_info="", children=None, has_handle=False, **kwargs):
        super().__init__(name, help_info, children, has_handle, **kwargs)

    def add_arguments(self, parser, **kwargs):
        return super().add_arguments(parser, **kwargs)

    def handle(self, args, **kwargs):
        return super().handle(args, **kwargs)

help_info = "debug a wide variety of model issues"
cmd_instances = []
for entry_point in pkg_resources.iter_entry_points('debug_sub_task'):
    cmd_instances.append(entry_point.load()())
debug_cmd = DebugCommand("debug", help_info, cmd_instances) if cmd_instances else None