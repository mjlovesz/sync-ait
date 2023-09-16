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
import stat
import subprocess
import sys
import os
import re

from model_convert.aoe.aoe_args_map import aoe_args
from model_convert.atc.atc_args_map import atc_args


PATH_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.-]")
MAX_READ_FILE_SIZE_4G = 4294967296  # 4G, 4 * 1024 * 1024 * 1024
MAX_READ_FILE_SIZE_32G = 34359738368  # 32G, 32 * 1024 * 1024 * 1024
READ_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH

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
            cmds.append(arg.get("name") + "=" + str(getattr(parse_args, arg_name)))

    return cmds


def execute_cmd(cmd: list):
    result = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while result.poll() is None:
        line = result.stdout.readline()
        if line:
            line = line.strip()
            print(line.decode('utf-8'))

    return result.returncode


def is_belong_to_user_or_group(file_stat):
    return file_stat.st_uid == os.getuid() or file_stat.st_gid in os.getgroups()


def is_endswith_extensions(path, extensions):
    result = False
    if isinstance(extensions, (list, tuple)):
        for extension in extensions:
            if path.endswith(extension):
                result = True
                break
    elif isinstance(extensions, str):
        result = path.endswith(extensions)
    return result


def get_valid_path(path, extensions=None):
    if not path or len(path) == 0:
        raise ValueError("The value of the path cannot be empty.")

    if PATH_WHITE_LIST_REGEX.search(path):  # Check special char
        raise ValueError("Input path contains invalid characters.")  # Not printing out the path value for invalid char
    if os.path.islink(os.path.abspath(path)):  # when checking link, get rid of the "/" at the path tail if any
        raise ValueError("The value of the path cannot be soft link: {}.".format(path))

    real_path = os.path.realpath(path)

    file_name = os.path.split(real_path)[1]
    if len(file_name) > 255:
        raise ValueError("The length of filename should be less than 256.")
    if len(real_path) > 4096:
        raise ValueError("The length of file path should be less than 4096.")

    if real_path != path and PATH_WHITE_LIST_REGEX.search(real_path):  # Check special char again
        raise ValueError("Input path contains invalid characters.")  # Not printing out the path value for invalid char
    if extensions and not is_endswith_extensions(path, extensions):  # Check whether the file name endswith extension
        raise ValueError("The filename {} doesn't endswith \"{}\".".format(path, extensions))

    return real_path


def get_valid_read_path(path, extensions=None, size_max=MAX_READ_FILE_SIZE_4G, check_user_stat=True, is_dir=False):
    real_path = get_valid_path(path, extensions)
    if not is_dir and not os.path.isfile(real_path):
        raise ValueError("The path {} doesn't exists or not a file.".format(path))
    if is_dir and not os.path.isdir(real_path):
        raise ValueError("The path {} doesn't exists or not a directory.".format(path))

    file_stat = os.stat(real_path)
    if check_user_stat and not sys.platform.startswith("win") and not is_belong_to_user_or_group(file_stat):
        raise ValueError("The file {} doesn't belong to the current user or group.".format(path))
    if check_user_stat and os.stat(path).st_mode & READ_FILE_NOT_PERMITTED_STAT > 0:
        raise ValueError("The file {} is group writable, or is others writable.".format(path))
    if not os.access(real_path, os.R_OK) or file_stat.st_mode & stat.S_IRUSR == 0:  # At least been 400
        raise ValueError("Current user doesn't have read permission to the file {}.".format(path))
    if not is_dir and size_max > 0 and file_stat.st_size > size_max:
        raise ValueError("The file {} exceeds size limitation of {}.".format(path, size_max))
    return real_path
