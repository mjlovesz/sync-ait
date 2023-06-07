#
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
import pathlib
import argparse

import click


def check_positive_integer(ctx, param, value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def check_batchsize_valid(ctx, param, value):
    # default value is None
    if value is None:
        return value
    # input value no None
    else:
        return check_positive_integer(ctx, param, value)


def check_nonnegative_integer(ctx, param, value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid nonnegative int value" % value)
    return ivalue


def check_device_range_valid(ctx, param, value):
    # if contain , split to int list
    min_value = 0
    max_value = 255
    if ',' in value:
        ilist = [ int(v) for v in value.split(',') ]
        for ivalue in ilist:
            if ivalue < min_value or ivalue > max_value:
                raise argparse.ArgumentTypeError("{} of device:{} is invalid. valid value range is [{}, {}]".format(
                    ivalue, value, min_value, max_value))
        return ilist
    else:
		# default as single int value
        ivalue = int(value)
        if ivalue < min_value or ivalue > max_value:
            raise argparse.ArgumentTypeError("device:{} is invalid. valid value range is [{}, {}]".format(
                ivalue, min_value, max_value))
        return ivalue


opt_application = click.option(
    "--application",
    default=None,
    type=str,
    help="Configure to run AI task files on the environment"
)


opt_output = click.option(
    '-o',
    '--output',
    'output',
    default="output",
    type=click.Path(
        path_type=pathlib.Path
    ),
    help='Inference data output path. The inference results are output to '
        'the subdirectory named current date under given output path'
)

opt_model_execution = click.option(
    '--model-execution',
    default="on",
    type=str,
    help='Control ge model execution performance data collection switch'
)
opt_sys_hardware_mem = click.option(
    "--sys-hardware-mem",
    default="on",
    type=str,
    help="Control the read/write bandwidth data acquisition switch for ddr and llc"
)
opt_sys_cpu_profiling = click.option(
    "--sys-cpu-profiling",
    default="off",
    type=str,
    help="CPU acquisition switch"
)
opt_sys_profiling = click.option(
    "--sys-profiling",
    default="off",
    type=str,
    help="System CPU usage and system memory acquisition switch"
)
opt_sys_pid_profiling = click.option(
    "--sys-pid-profiling",
    default="off",
    type=str,
    help="The CPU usage of the process and the memory collection switch of the process"
)
opt_dvpp_profiling = click.option(
    "--dvpp-profiling",
    default="on",
    type=str,
    help="DVPP acquisition switch"
)

opt_runtime_api = click.option(
    "--runtime-api",
    default="on",
    type=str,
    help="Control runtime api performance data collection switch"
)
opt_task_time = click.option(
    "--task-time",
    default="on",
    type=str,
    help="Control ts timeline performance data collection switch"
)
opt_aicpu = click.option(
    "--aicpu",
    default="on",
    type=str,
    help="Control aicpu performance data collection switch"
)