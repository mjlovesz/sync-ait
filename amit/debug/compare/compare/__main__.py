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

from compare.adapter_cli.args_adapter import MyArgs
from compare.adapter_cli.options import (
    opt_gold_model,
    opt_om_model,
    opt_input,
    opt_cann_path,
    opt_out_path,
    opt_input_shape,
    opt_device,
    opt_output_size,
    opt_output_nodes,
    opt_advisor
)
from compare.main import main


@click.command(name="compare", short_help='one-click network-wide accuracy analysis of TensorFlow and ONNX models.')
@opt_gold_model
@opt_om_model
@opt_input
@opt_cann_path
@opt_out_path
@opt_input_shape
@opt_device
@opt_output_size
@opt_output_nodes
@opt_advisor
def compare_cli_enter(
    gold_model,
    om_model,
    input,
    cann_path,
    out_path,
    input_shape,
    device,
    output_size,
    output_nodes,
    advisor
) -> None:
    my_agrs = MyArgs(gold_model, om_model, input, cann_path, out_path, input_shape, device,
                     output_size, output_nodes, advisor)
    return main(my_agrs)

if __name__ == '__main__':
    compare_cli_enter()
