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
import logging
import math
import os
import sys
import time
import shutil
import copy

from ait_prof.utils import logger
from ait_prof.args_adapter import MsProfArgsAdapter


def msprof_run_profiling(args, msprof_bin):
    cmd = sys.executable + " " + ' '.join(sys.argv) + " --profiler=0 --warmup-count=0"
    msprof_cmd = "{} --output={}/profiler --application=\"{}\" --model-execution={}" \
               " --sys-hardware-mem={} --sys-cpu-profiling={}" \
               " --sys-profiling={} --sys-pid-profiling={} --dvpp-profiling={} " \
               "--runtime-api={} --task-time={} --aicpu={}".format(msprof_bin, args.output, args.application,
                                                                   args.model_execution, args.sys_hardware_mem,
                                                                   args.sys_cpu_profiling, args.sys_profiling,
                                                                   args.sys_pid_profiling, args.dvpp_profiling,
                                                                   args.runtime_api, args.task_time, args.aicpu)
    logger.info("msprof cmd:{} begin run".format(msprof_cmd))
    ret = os.system(msprof_cmd)
    logger.info("msprof cmd:{} end run ret:{}".format(msprof_cmd, ret))


def args_rules(args):
    if args.output is None and args.output_dirname is not None:
        logger.error("parameter --output_dirname cann't be used alone. "
                     "Please use it together with the parameter --output!\n")
        raise RuntimeError('error bad parameters --output_dirname')
    return args


def msprof_process(args:MsProfArgsAdapter):
    msprof_bin = shutil.which('msprof')
    if msprof_bin is None or os.getenv('GE_PROFILIGN_TO_STD_OUT') == '1':
        logger.info("find no msprof continue use acl.json mode")
    else:
        msprof_run_profiling(args, msprof_bin)
        return 0

    return 0
