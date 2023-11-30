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


import argparse
import re

STR_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9\"'><=\[\])(,}{: /.~-]")
INVALID_CHARS = ['|', ';', '&', '&&', '||', '>', '>>', '<', '`', '\\', '!', '\n']


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected true, 1, false, 0 with case insensitive.')


def check_positive_integer(value):
    ivalue = int(value)
    if ivalue < 0 or ivalue > 2:
        raise argparse.ArgumentTypeError("%s is an invalid int value" % value)
    return ivalue


def safe_string(value):
    if not value:
        return value
    if re.search(STR_WHITE_LIST_REGEX, value):
        raise ValueError("String parameter contains invalid characters.")
    return value


def check_number_list(value):
    # just like "1241414,124141,124424"
    if not value:
        return value
    outsize_list = value.split(',')
    for outsize in outsize_list:
        regex = re.compile(r"[^0-9]")
        if regex.search(outsize):
            raise argparse.ArgumentTypeError(f"output size \"{outsize}\" is not a legal string")
    return value


def check_ids_string(value):
    if not value:
        return value
    dym_string = value
    regex = re.compile(r"[^_A-Za-z0-9,;:/.\-~]")
    if regex.search(dym_string):
        raise argparse.ArgumentTypeError(f"dym range string \"{dym_string}\" is not a legal string")
    return dym_string

def check_exec_script_file(script_path: str):
    if not os.path.exists(script_path):
        raise argparse.ArgumentTypeError(f"Script Path is not valid : {script_path}")

    if not os.access(script_path, os.X_OK):
        raise argparse.ArgumentTypeError(f"Script Path is not valid : {script_path}")


def check_input_args(args: list):
    for arg in args:
        if arg in INVALID_CHARS:
            raise argparse.ArgumentTypeError(f"Args has invalid chars.Please check")


def check_exec_cmd(command: str):
    if command.startswith("bash") or command.startswith("python"):
        cmds = command.split()
        if len(cmds) < 2:
            raise argparse.ArgumentTypeError(f"Run cmd is not valid: \"{command}\" ")
        elif len(cmds) == 2:
            script_file = cmds[1]
            check_exec_script_file(script_file)
        else:
            script_file = cmds[1]
            check_exec_script_file(script_file)
            args = cmds[2:]
            check_input_args(args)
        return True

    else:
        raise argparse.ArgumentTypeError(f"Run cmd is not valid: \"{command}\" ")