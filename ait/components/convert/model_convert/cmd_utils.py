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

import argparse
import logging
import subprocess
import sys
import os

from model_convert.aoe.aoe_args_map import aoe_args
from model_convert.atc.atc_args_map import atc_args
from model_convert.security_check import input_file_args, input_dir_args, output_file_args, \
    get_valid_read_path, get_valid_write_path


def get_logger(name=__name__):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger(name)
    return logger


logger = get_logger()

CUR_PATH = os.path.dirname(os.path.relpath(__file__))

BACKEND_ARGS_MAPPING = {
    "atc": atc_args,
    "aoe": aoe_args
}
BACKEND_CMD_MAPPING = {
    "atc": ["atc"],
    "aoe": ["aoe"]
}


def add_arguments(parser, backend="atc"):
    args = BACKEND_ARGS_MAPPING.get(backend)
    if not args:
        raise ValueError("Backend must be atc or aoe!")

    for arg in args:
        abbr_name = arg.get('abbr_name') if arg.get('abbr_name') else ""
        is_required = arg.get('is_required') if arg.get('is_required') else False

        if abbr_name:
            parser.add_argument(abbr_name, arg.get('name'), required=is_required, help=arg.get('desc'))
        else:
            parser.add_argument(arg.get('name'), required=is_required, help=arg.get('desc'))

    return args


def gen_convert_cmd(conf_args: list, parse_args: argparse.Namespace, backend: str = "atc"):
    cmds = BACKEND_CMD_MAPPING.get(backend)
    if not cmds:
        raise ValueError("Backend must be atc or aoe!")

    for arg in conf_args:
        arg_name = arg.get("name")[2:]
        if hasattr(parse_args, arg_name) and getattr(parse_args, arg_name):
            arg_value = str(getattr(parse_args, arg_name))
            if arg_name in input_file_args:
                arg_value = get_valid_read_path(arg_value)
            if arg_name in input_dir_args:
                arg_value = get_valid_read_path(arg_value, is_dir=True)
            if arg_name in output_file_args:
                arg_value = get_valid_write_path(arg_value)

            cmds.append(arg.get("name") + "=" + arg_value)

    return cmds


def execute_cmd(cmd: list):
    result = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while result.poll() is None:
        line = result.stdout.readline()
        if line:
            line = line.strip()
            print(line.decode('utf-8'))

    return result.returncode
