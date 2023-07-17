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
import sys

import click

from components.parser.parser import CommandInfo
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.adapter_cli.options import (
    opt_golden_model,
    opt_om_model,
    opt_weight_path,
    opt_input,
    opt_cann_path,
    opt_out_path,
    opt_input_shape,
    opt_device,
    opt_output_size,
    opt_output_nodes,
    opt_advisor,
    opt_dym_shape_range,
    opt_dump,
    opt_bin2npy,
    opt_custom_op,
    opt_locat,
    opt_onnx_fusion_switch
)
from msquickcmp.cmp_process import cmp_process
from msquickcmp.common import utils


@click.command(name="compare", short_help='one-click network-wide accuracy analysis of golden models.',
               no_args_is_help=True)
@opt_golden_model
@opt_om_model
@opt_weight_path
@opt_input
@opt_cann_path
@opt_out_path
@opt_input_shape
@opt_device
@opt_output_size
@opt_output_nodes
@opt_advisor
@opt_dym_shape_range
@opt_dump
@opt_bin2npy
@opt_custom_op
@opt_locat
@opt_onnx_fusion_switch
def compare_cli(
    golden_model,
    om_model,
    weight_path,
    input_data_path,
    cann_path,
    out_path,
    input_shape,
    device,
    output_size,
    output_nodes,
    advisor,
    dym_shape_range,
    dump,
    bin2npy,
    custom_op,
    locat,
    onnx_fusion_switch
) -> None:
    cmp_args = CmpArgsAdapter(golden_model, om_model, weight_path, input_data_path, cann_path, out_path,
                              input_shape, device, output_size, output_nodes, advisor, dym_shape_range,
                              dump, bin2npy, custom_op, locat, onnx_fusion_switch)
    return cmp_process(cmp_args, True)

if __name__ == '__main__':
    try:
        compare_cli()
    except utils.AccuracyCompareException as error:
        sys.exit(error.error_info)


class CompareCommand:
    def add_arguments(self, parser):
        parser.add_argument("-om", "--om-model", required=True, default=None, help="the path of the om model")
        parser.add_argument("-i", "--input", default=None, help="the path of the input file or dir")
        parser.add_argument("-o", "--output", default=None, help="the path of the output dir")

    def handle(self, args):
        print(vars(args))
        print("hello from compare")

def get_cmd_info():
    cmd_instance = CompareCommand()
    return CommandInfo("compare", cmd_instance)