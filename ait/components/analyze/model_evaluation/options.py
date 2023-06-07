# Copyright (c) 2023 Huawei Technologies Co., Ltd.
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

import click


def check_args(ctx: click.Context, params: click.Option, value: str):
    """
    check whether the param is provided
    """
    args = [
        opt
        for param in ctx.command.params
        for opt in param.opts
    ]
    if value in args:
        raise click.MissingParameter()
    return value


opt_model = click.option(
    '-gm',
    '--golden_model',
    'input_model',
    type=str,
    required=True,
    callback=check_args,
    help='model path, support caffe, onnx, tensorflow.'
)

opt_framework = click.option(
    '--framework',
    'framework',
    type=click.Choice(['0', '3', '5']),
    callback=check_args,
    help='Framework type: 0:Caffe; 3:Tensorflow; 5:Onnx.'
)

opt_weight = click.option(
    '--weight',
    'weight',
    type=str,
    default='',
    callback=check_args,
    help='Weight file. Required when framework is Caffe.'
)

opt_soc = click.option(
    '-soc',
    '--soc_version',
    'soc',
    type=str,
    callback=check_args,
    help='The soc version.'
)

opt_out_path = click.option(
    '-o',
    '--output',
    'output',
    type=str,
    required=True,
    callback=check_args,
    help='Output path.'
)
