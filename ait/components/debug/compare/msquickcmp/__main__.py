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
import os
import argparse

from components.parser.parser import BaseCommand
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
    opt_onnx_fusion_switch,
    opt_fusion_switch_file,
    opt_single_op
)
from msquickcmp.cmp_process import cmp_process
from msquickcmp.common import utils


CANN_PATH = os.environ.get('ASCEND_TOOLKIT_HOME', "/usr/local/Ascend/ascend-toolkit/latest")


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
@opt_fusion_switch_file
@opt_single_op
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
    onnx_fusion_switch,
    single_op,
    fusion_switch_file
) -> None:
    cmp_args = CmpArgsAdapter(golden_model, om_model, weight_path, input_data_path, cann_path, out_path,
                              input_shape, device, output_size, output_nodes, advisor, dym_shape_range,
                              dump, bin2npy, custom_op, locat, onnx_fusion_switch, single_op, fusion_switch_file)
    return cmp_process(cmp_args, True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected true, 1, false, 0 with case insensitive.')

if __name__ == '__main__':
    try:
        compare_cli()
    except utils.AccuracyCompareException as error:
        sys.exit(error.error_info)


class CompareCommand(BaseCommand):
    def __init__(self, name="", help="", children=[]):
        super().__init__(name, help, children)

    def add_arguments(self, parser):
        parser.add_argument(
            '-gm',
            '--golden-model',
            required=True,
            dest="golden_model",
            help='The original model (.onnx or .pb or .prototxt) file path')
        parser.add_argument(
            '-om',
            '--om-model',
            dest="om_model",
            help='The offline model (.om) file path')
        parser.add_argument(
            '-w',
            '--weight',
            dest="weight_path",
            help='Required when framework is Caffe (.cafemodel)')
        parser.add_argument(
            '-i',
            '--input',
            default='',
            dest="input_data_path",
            help='The input data path of the model. Separate multiple inputs with commas(,).' 
                 ' E.g: input_0.bin,input_1.bin')
        parser.add_argument(
            '-c',
            '--cann-path',
            default=CANN_PATH,
            dest="cann_path",
            help='The CANN installation path')
        parser.add_argument(
            '-o',
            '--output',
            dest="out_path",
            default='',
            help='The output path')
        parser.add_argument(
            '-is',
            '--input-shape',
            type=str,
            dest="input_shape",
            default='',
            help="Shape of input shape. Separate multiple nodes with semicolons(;)."
                 " E.g: \"input_name1:1,224,224,3;input_name2:3,300\"")
        parser.add_argument(
            '-d',
            '--device',
            dest="device",
            default='0',
            help='Input device ID [0, 255], default is 0.')
        parser.add_argument(
            '-outsize',
            '--output-size',
            dest="output_size",
            default='',
            help='The size of output. Separate multiple sizes with commas(,). E.g: 10200,34000')
        parser.add_argument(
            '-n',
            '--output-nodes',
            type=str,
            dest="output_nodes",
            default='',
            help="Output nodes designated by user. Separate multiple nodes with semicolons(;)."
                 " E.g: \"node_name1:0;node_name2:1;node_name3:0\"")
        parser.add_argument(
            '--advisor',
            action='store_true',
            dest="advisor",
            help='Enable advisor after compare.')
        parser.add_argument(
            '-dr',
            '--dym-shape-range',
            type=str,
            dest="dym_shape_range",
            default='',
            help="Dynamic shape range using in dynamic model, "
                 "using this means ignore input_shape"
                 " E.g: \"input_name1:1,3,200\~224,224-230;input_name2:1,300\"")
        parser.add_argument(
            '--dump',
            dest="dump",
            default=True,
            type=str2bool,
            help="Whether to dump all the operations' ouput. Default True.")
        parser.add_argument(
            '--convert',
            dest="bin2npy",
            default=False,
            type=str2bool,
            help='Enable npu dump data conversion from bin to npy after compare.Usage: --convert True.')
        parser.add_argument(
            '--locat',
            default=False,
            dest="locat",
            type=str2bool,
            help='Enable accuracy interval location when needed.E.g: --locat True.')
        parser.add_argument(
            '-cp',
            '--custom-op',
            type=str,
            dest="custom_op",
            default='',
            help='Op name witch is not registered in onnxruntime, only supported by Ascend.')
        parser.add_argument(
            '-ofs',
            '--onnx-fusion-switch',
            dest="onnx_fusion_switch",
            default=True,
            type=str2bool,
            help='Onnxruntime fusion switch, set False for dump complete onnx data when '
                 'necessary.Usage: -ofs False.')
        parser.add_argument(
            '--fusion-switch-file',
            dest="fusion_switch_file",
            help='You can disable selected fusion patterns in the configuration file')
        parser.add_argument(
            "-single",
            "--single-op",
            dest="single_op",
            type=str2bool,
            help='Comparision mode:single operator compare, default false.Usage: -single True')

    def handle(self, args):
        print(vars(args))
        print("hello from compare")
        cmp_args = CmpArgsAdapter(args.golden_model, args.om_model, args.weight_path, args.input_data_path,
                                  args.cann_path, args.out_path,
                                  args.input_shape, args.device, args.output_size, args.output_nodes, args.advisor,
                                  args.dym_shape_range,
                                  args.dump, args.bin2npy, args.custom_op, args.locat,
                                  args.onnx_fusion_switch, args.fusion_switch_file, args.single_op)
        cmp_process(cmp_args, True)

def get_cmd_info():
    help_info = "one-click network-wide accuracy analysis of golden models."
    cmd_instance = CompareCommand("compare", help_info)
    return cmd_instance