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
import pkg_resources

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def load_analyze_sub_task():
    sub_tasks = {}
    for entry_point in pkg_resources.iter_entry_points('analyze_sub_task'):
        sub_tasks[entry_point.name] = entry_point.load()

    if len(sub_tasks) > 1:
        return click.Group(name='analyze',
            context_settings=CONTEXT_SETTINGS,
            commands=sub_tasks
        )
    elif len(sub_tasks) == 1:
        sub_task = list(sub_tasks.values())[0]
        sub_task.name = 'analyze'
        return sub_task
    else:
        return click.Group(name='analyze',
            context_settings=CONTEXT_SETTINGS
        )

analyze_cli = load_analyze_sub_task()