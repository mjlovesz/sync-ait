import argparse
import logging
import subprocess
import sys

import yaml


def get_logger(name=__name__):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger(name)
    return logger


BACKEND_CONF_MAPPING = {
    "atc": "atc/atc_args_map.yml",
    "aoe": "aoe/aoe_args_map.yml"
}
BACKEND_CMD_MAPPING = {
    "atc": ["atc"],
    "aoe": ["aoe"]
}


def add_arguments(parser, backend="atc"):
    conf_file = BACKEND_CONF_MAPPING.get(backend)
    if not conf_file:
        raise ValueError("Backend must be atc or aoe!")

    with open(conf_file, 'r', encoding='utf-8') as f:
        args_conf = yaml.load(f, yaml.Loader)

    args = args_conf.get('args')
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
        arg_name = arg[2:]
        if hasattr(parse_args, arg_name) and getattr(parse_args, arg_name):
            cmds.append(arg + "=" + str(getattr(parse_args, arg_name)))

    return cmds


def execute_cmd(cmd: list):
    result = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    logger.info(result.stdout.decode('utf-8'))
