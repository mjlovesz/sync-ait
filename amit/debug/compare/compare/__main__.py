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
import os
import sys
import time

from compare.atc.atc_utils import AtcUtils
from compare.common import utils
from compare.common.utils import AccuracyCompareException, get_shape_to_directory_name
from compare.net_compare.net_compare import NetCompare
from compare.analyser import analyser
from compare.npu.npu_dump_data import NpuDumpData
from compare.adapter_cli.args_adapter import MyArgs
from compare.npu.npu_dump_data_bin2npy import data_convert

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
from compare.adapter_cli.args_adapter import MyArgs


def _generate_golden_data_model(args):
    model_name, extension = utils.get_model_name_and_extension(args.model_path)
    if ".pb" == extension:
        from tf.tf_dump_data import TfDumpData
        return TfDumpData(args)
    elif ".onnx" == extension:
        from compare.onnx_model.onnx_dump_data import OnnxDumpData
        return OnnxDumpData(args)
    else:
        utils.print_error_log("Only model files whose names end with .pb or .onnx are supported")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_MODEL_TYPE_ERROR)


def _correct_the_wrong_order(left_index, right_index, golden_net_output_info):
    if left_index not in golden_net_output_info.keys() or right_index not in golden_net_output_info.keys():
        return
    if left_index != right_index:
        tmp = golden_net_output_info[left_index]
        golden_net_output_info[left_index] = golden_net_output_info[right_index]
        golden_net_output_info[right_index] = tmp
        utils.print_info_log("swap the {} and {} item in golden_net_output_info!"
                             .format(left_index, right_index))


def _check_output_node_name_mapping(original_net_output_node, golden_net_output_info):
    for left_index, node_name in original_net_output_node.items():
        match = False
        for right_index, dump_file_path in golden_net_output_info.items():
            dump_file_name = os.path.basename(dump_file_path)
            if dump_file_name.startswith(node_name.replace("/", "_").replace(":", ".")):
                match = True
                _correct_the_wrong_order(left_index, right_index, golden_net_output_info)
                break
        if not match:
            utils.print_warn_log("the original name: {} of net output maybe not correct!".format(node_name))
            break

def run(args, input_shape, output_json_path, original_out_path):
    if input_shape:
        args.input_shape = input_shape
        args.out_path = os.path.join(original_out_path, get_shape_to_directory_name(args.input_shape))

    # generate dump data by the original model
    golden_dump = _generate_golden_data_model(args)
    golden_dump_data_path = golden_dump.generate_dump_data()
    golden_net_output_info = golden_dump.get_net_output_info()

    # compiling and running source codes
    npu_dump = NpuDumpData(args, output_json_path)
    npu_dump_data_path, npu_net_output_data_path = npu_dump.generate_dump_data()
    expect_net_output_node = npu_dump.get_expect_output_name()

    # convert data from bin to npy if --convert is used
    data_convert(npu_dump_data_path, npu_net_output_data_path, args)

    # if it's dynamic batch scenario, golden data files should be renamed
    utils.handle_ground_truth_files(npu_dump.om_parser, npu_dump_data_path, golden_dump_data_path)

    if not args.dump:
        # only compare the final output
        net_compare = NetCompare(npu_net_output_data_path, golden_dump_data_path, output_json_path, args)
        net_compare.net_output_compare(npu_net_output_data_path, golden_net_output_info)
    else:
        # compare the entire network
        net_compare = NetCompare(npu_dump_data_path, golden_dump_data_path, output_json_path, args)
        net_compare.accuracy_network_compare()
    # Check and correct the mapping of net output node name.
    if len(expect_net_output_node) == 1:
        _check_output_node_name_mapping(expect_net_output_node, golden_net_output_info)
        net_compare.net_output_compare(npu_net_output_data_path, golden_net_output_info)
    analyser.Analyser(args.out_path)()

def cmp_main(my_args:MyArgs):
    my_args.offline_model_path = os.path.realpath(my_args.offline_model_path)
    my_args.cann_path = os.path.realpath(my_args.cann_path)
    try:
        utils.check_file_or_directory_path(my_args.model_path)
        utils.check_file_or_directory_path(my_args.offline_model_path)
        utils.check_device_param_valid(my_args.device)
        utils.check_file_or_directory_path(os.path.realpath(my_args.out_path), True)
        time_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())
        original_out_path = os.path.realpath(os.path.join(my_args.out_path, time_dir))
        my_args.out_path = original_out_path

        # convert the om model to json
        output_json_path = AtcUtils(my_args).convert_model_to_json()

        # deal with the dymShape_range param if exists
        input_shapes = []
        if my_args.dymShape_range:
            input_shapes = utils.parse_dymshape_range(my_args.dymShape_range)
        if not input_shapes:
            input_shapes.append("")
        for input_shape in input_shapes:
            run(my_args, input_shape, output_json_path, original_out_path)
    except utils.AccuracyCompareException as error:
        sys.exit(error.error_info)

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
    return cmp_main(my_agrs)

if __name__ == '__main__':
    compare_cli_enter()
