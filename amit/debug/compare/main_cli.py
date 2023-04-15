# Copyright 2022 Huawei Technologies Co., Ltd
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

from .options import MyArgs
from .options import (
    opt_model_path,
    opt_input_path,
    opt_cann_path,
    opt_out_path,
    opt_input_shape,
    opt_device,
    opt_output_size,
    opt_output_nodes,
    opt_advisor
)


@click.command(name="compare", short_help='one-click network-wide accuracy analysis of TensorFlow and ONNX models.')
@opt_model_path
@opt_input_path
@opt_cann_path
@opt_out_path
@opt_input_shape
@opt_device
@opt_output_size
@opt_output_nodes
@opt_advisor
def compare_cli_enter(
    model_path,
    input_path,
    cann_path,
    out_path,
    input_shape,
    device,
    output_size,
    output_nodes,
    advisor
) -> None:
    my_agrs = MyArgs(model_path, input_path, cann_path, out_path, input_shape, device,
                     output_size, output_nodes, advisor)
    return cil_enter(my_agrs)

## todo: for test
def cil_enter(my_args:MyArgs):
    click.echo("cil_enter, input:MyArgs")
    return
