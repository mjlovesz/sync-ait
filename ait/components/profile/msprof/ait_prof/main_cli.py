# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
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
from ait_prof.msprof_process import msprof_process
from ait_prof.args_adapter import MsProfArgsAdapter
from ait_prof.options import (
    opt_application,
    opt_output,
    opt_model_execution,
    opt_sys_hardware_mem,
    opt_sys_cpu_profiling,
    opt_sys_profiling,
    opt_sys_pid_profiling,
    opt_dvpp_profiling,
    opt_runtime_api,
    opt_task_time,
    opt_aicpu
)


@click.command(name="profile",
               short_help = "profile tool to get performance datProfiling, as a professional performance analysis tool "
                            "for Ascension AI tasks, covers the collection of key data and analysis of performance"
                            " indicators during AI task execution.",
               no_args_is_help=True)
@opt_application
@opt_output
@opt_model_execution
@opt_sys_hardware_mem
@opt_sys_cpu_profiling
@opt_sys_profiling
@opt_sys_pid_profiling
@opt_dvpp_profiling
@opt_runtime_api
@opt_task_time
@opt_aicpu
def msprof_cli(application, output, model_execution, sys_hardware_mem, sys_cpu_profiling, sys_profiling,
               sys_pid_profiling, dvpp_profiling, runtime_api, task_time, aicpu):
    args = MsProfArgsAdapter(application, output.as_posix() if output else None,
                                model_execution, sys_hardware_mem, sys_cpu_profiling, sys_profiling,
                 sys_pid_profiling, dvpp_profiling, runtime_api, task_time, aicpu)
    ret = msprof_process(args)
    exit(ret)


class ProfileCommand:
    def add_arguments(self, parser):
        # parser.add_argument("-om", "--om-model", required=True, default=None, help="the path of the om model")
        # parser.add_argument("-i", "--input", default=None, help="the path of the input file or dir")
        # parser.add_argument("-o", "--output", default=None, help="the path of the output dir")
        parser.add_argument(
            "--application",
            required=True,
            help="Configure to run AI task files on the environment"
        )
        parser.add_argument(
            "--output",
            default=None,
            help="The storage path for the collected profiling data,"
                " which defaults to the directory where the app is located"
        )
        parser.add_argument(
            "--model-execution",
            default="on",
            choices=["on", "off"],
            help="Control ge model execution performance data collection switch"
        )
        parser.add_argument(
            "--sys-hardware-mem",
            default="on",
            choices=["on", "off"],
            help="Control the read/write bandwidth data acquisition switch for ddr and llc"
        )
        parser.add_argument(
            "--sys-cpu-profiling",
            default="off",
            choices=["on", "off"],
            help="CPU acquisition switch"
        )
        parser.add_argument(
            "--sys-profiling",
            default="off",
            choices=["on", "off"],
            help="System CPU usage and system memory acquisition switch"
        )
        parser.add_argument(
            "--sys-pid-profiling",
            default="off",
            choices=["on", "off"],
            help="The CPU usage of the process and the memory collection switch of the process"
        )
        parser.add_argument(
            "--dvpp-profiling",
            default="on",
            choices=["on", "off"],
            help="DVPP acquisition switch"
        )
        parser.add_argument(
            "--runtime-api",
            default="on",
            choices=["on", "off"],
            help="Control runtime api performance data collection switch"
        )
        parser.add_argument(
            "--task-time",
            default="on",
            choices=["on", "off"],
            help="Control ts timeline performance data collection switch"
        )
        parser.add_argument(
            "--aicpu",
            default="on",
            choices=["on", "off"],
            help="Control aicpu performance data collection switch"
        )

    def handle(self, args):
        print(vars(args))
        print("hello from profile")

def get_cmd_info():
    cmd_instance = ProfileCommand()
    return CommandInfo("profile", cmd_instance)