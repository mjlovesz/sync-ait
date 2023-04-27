#!/usr/bin/env python
# coding=utf-8
"""
Function:
This class mainly involves the main function.
Copyright Information:
HuaWei Technologies Co.,Ltd. All Rights Reserved © 2021
"""

import argparse
import sys

from msquickcmp.cmp_process import cmp_process
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.common.utils import str2bool


def _accuracy_compare_parser(parser):
    parser.add_argument("-m", "--model-path", dest="model_path", default="",
                        help="<Required> The original model (.onnx or .pb) file path", required=True)
    parser.add_argument("-om", "--offline-model-path", dest="offline_model_path", default="",
                        help="<Required> The offline model (.om) file path", required=True)
    parser.add_argument("-i", "--input-path", dest="input_path", default="",
                        help="<Optional> The input data path of the model. Separate multiple inputs with commas(,)."
                             " E.g: input_0.bin,input_1.bin")
    parser.add_argument("-c", "--cann-path", dest="cann_path", default="/usr/local/Ascend/ascend-toolkit/latest/",
                        help="<Optional> The CANN installation path")
    parser.add_argument("-o", "--out-path", dest="out_path", default="", help="<Optional> The output path")
    parser.add_argument("-s", "--input-shape", dest="input_shape", default="",
                        help="<Optional> Shape of input shape. Separate multiple nodes with semicolons(;)."
                             " E.g: input_name1:1,224,224,3;input_name2:3,300")
    parser.add_argument("-d", "--device", dest="device", default="0",
                        help="<Optional> Input device ID [0, 255], default is 0.")
    parser.add_argument("--output-size", dest="output_size", default="",
                        help="<Optional> The size of output. Separate multiple sizes with commas(,)."
                             " E.g: 10200,34000")
    parser.add_argument("--output-nodes", dest="output_nodes", default="",
                        help="<Optional> Output nodes designated by user. Separate multiple nodes with semicolons(;)."
                             " E.g: node_name1:0;node_name2:1;node_name3:0")
    parser.add_argument("--advisor", dest="advisor", action="store_true",
                        help="<Optional> Enable advisor after compare.")
    parser.add_argument("-dr", "--dymShape-range", dest="dym_shape_range", default="",
                        help="<Optional> Dynamic shape range using in dynamic model, "
                             "using this means ignore input_shape")
    parser.add_argument("--dump", dest="dump", default=True, type=str2bool,
                        help="<Optional> Whether to dump all the operations' ouput. Default True.")
    parser.add_argument("--convert", dest = "bin2npy", action="store_true",
                        help="<Optional> Enable npu dump data conversion from bin to npy after compare.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _accuracy_compare_parser(parser)
    args = parser.parse_args(sys.argv[1:])

    cmp_args = CmpArgsAdapter(args.model_path, args.offline_model_path, args.input_path, 
                              args.cann_path, args.out_path, args.input_shape, 
                              args.device, args.output_size, args.output_nodes, args.advisor, 
                              args.dym_shape_range, args.dump, args.bin2npy)
    cmp_process(cmp_args)
