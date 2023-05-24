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

from model_eval.common import utils, logger
from model_eval.common.enum import Framework
from model_eval.bean import ConvertConfig
from model_eval.core import Analyze
from model_eval.options import (
    opt_model,
    opt_out_path,
    opt_soc,
    opt_weight,
    opt_framework
)


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


@click.command(short_help='Analyze tool to analyze model support', no_args_is_help=True)
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


if __name__ == '__main__':
    cli()
