import argparse
import logging
import subprocess
import sys
import os

import yaml

from model_convert.aoe.aoe_args_map import aoe_args
from model_convert.atc.atc_args_map import atc_args


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
