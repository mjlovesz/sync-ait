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
import os
import sys

import click

from parser.parser import CommandInfo
from aie_runtime.bean import ConvertConfig
from aie_runtime.core import Convert
from aie_runtime.options import (
    opt_model,
    opt_out_path,
    opt_soc
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_input_param(model: str,
                      output: str,
                      soc_version: str) -> ConvertConfig:

    return ConvertConfig(
        model=model,
        output=output,
        soc_version=soc_version
    )


@click.command(short_help='Model convert tool to convert offline model', no_args_is_help=True)
@opt_model
@opt_soc
@opt_out_path
def cli(
        model: str,
        output: str,
        soc_version: str
) -> None:
    if not os.path.isfile(model):
        logger.error('Input model is not a file.')
        return

    try:
        config = parse_input_param(
            model, output, soc_version
        )
    except ValueError as e:
        logger.error(f'{e}')
        return

    converter = Convert(config)
    if converter is None:
        logger.error('The object of \'convert\' create failed.')
        return

    converter.convert_model()
    logger.info('convert model finished.')


if __name__ == "__main__":
    cli()


class ConvertCommand:
    def add_arguments(self, parser):
        parser.add_argument("-om", "--om-model", required=True, default=None, help="the path of the om model")
        parser.add_argument("-i", "--input", default=None, help="the path of the input file or dir")
        parser.add_argument("-o", "--output", default=None, help="the path of the output dir")

    def handle(self, args):
        print(vars(args))
        print("hello from convert")

def get_cmd_info():
    cmd_instance = ConvertCommand()
    return CommandInfo("convert", cmd_instance)