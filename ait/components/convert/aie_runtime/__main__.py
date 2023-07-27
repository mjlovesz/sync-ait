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

from components.parser.parser import BaseCommand
from aie_runtime.bean import ConvertConfig
from aie_runtime.core import Convert

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


class ConvertCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument("-gm",
                            "--golden-model",
                            dest="model",
                            required=True,
                            default=None,
                            help="the path of the onnx model")
        parser.add_argument("-of",
                            "--output-file",
                            dest="output",
                            required=True,
                            default=None,
                            help="Output file path&name(needn\'t .om suffix for ATC, need .om suffix for AIE)")
        parser.add_argument("-soc",
                            "--soc-version",
                            dest='soc_version',
                            required=True,
                            default=None,
                            help="The soc version.")

    def handle(self, args):
        if not os.path.isfile(args.model):
            logger.error('Input model is not a file.')
            return

        try:
            config = parse_input_param(
                args.model, args.output, args.soc_version
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


def get_cmd_instance():
    help_info = "convert tool converts the model from ONNX to OM."
    cmd_instance = ConvertCommand("convert", help_info)
    return cmd_instance