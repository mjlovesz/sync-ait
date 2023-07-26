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

import os
import click

from components.parser.parser import BaseCommand
from model_evaluation.common import utils, logger
from model_evaluation.common.enum import Framework
from model_evaluation.bean import ConvertConfig
from model_evaluation.core import Analyze
from model_evaluation.options import (
    opt_model,
    opt_out_path,
    opt_soc,
    opt_weight,
    opt_framework
)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def parse_input_param(model: str,
    framework: str, weight: str, soc: str
) -> ConvertConfig:
    if framework is None:
        framework = utils.get_framework(model)
        if framework == Framework.UNKNOWN:
            raise ValueError(
                'parse framework failed, use --framework.'
            )
    else:
        if not framework.isdigit():
            raise ValueError('framework is illegal, use --help.')
        framework = Framework(int(framework))

    if soc is None:
        soc = utils.get_soc_type()

    return ConvertConfig(
        framework=framework,
        weight=weight,
        soc_type=soc
    )


@click.command(short_help='Analyze tool to analyze model support', no_args_is_help=True,
               context_settings=CONTEXT_SETTINGS)
@opt_model
@opt_framework
@opt_weight
@opt_soc
@opt_out_path
def cli(
    input_model: str, framework: str, weight: str,
    soc: str, output: str
) -> None:
    if not os.path.isfile(input_model):
        logger.error('input model is not file.')
        return

    try:
        config = parse_input_param(
            input_model, framework, weight, soc
        )
    except ValueError as e:
        logger.error(f'{e}')
        return

    analyzer = Analyze(input_model, output, config)
    if analyzer is None:
        logger.error('the object of \'Analyze\' create failed.')
        return

    analyzer.analyze_model()
    logger.info('analyze model finished.')


class AnalyzeCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            "-gm", "--golden-model", type=str,
            required=True, default=None,
            help="model path, support caffe, onnx, tensorflow."
        )
        parser.add_argument(
            "--framework", type=str,
            choices=['0', '3', '5'],
            default=None, help="Framework type: 0:Caffe; 3:Tensorflow; 5:Onnx."
        )
        parser.add_argument(
            "-w", "--weight", type=str,
            required=False, default='',
            help="Weight file. Required when framework is Caffe."
        )
        parser.add_argument(
            "-soc", "--soc-version", type=str,
            required=False, default='',
            help="The soc version."
        )
        parser.add_argument(
            "-o", "--output", type=str,
            required=True, default='',
            help="Output path."
        )

    def handle(self, args):
        input_model = args.golden_model
        framework = args.framework
        weight = args.weight
        soc_version = args.soc_version
        output = args.output

        if not os.path.isfile(input_model):
            logger.error('input model is not file.')
            return

        try:
            config = parse_input_param(
                input_model, framework, weight, soc_version
            )
        except ValueError as e:
            logger.error(f'{e}')
            return

        analyzer = Analyze(input_model, output, config)
        if analyzer is None:
            logger.error('the object of \'Analyze\' create failed.')
            return

        analyzer.analyze_model()
        logger.info('analyze model finished.')


def get_cmd_instance():
    help_info = "Analyze tool to evaluate compatibility of model conversion"
    cmd_instance = AnalyzeCommand("analyze", help_info)
    return cmd_instance


if __name__ == '__main__':
    cli()
