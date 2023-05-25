#!/usr/bin/env python
# coding=utf-8
"""
Function:
This class mainly involves the main function.
Copyright Information:
HuaWei Technologies Co.,Ltd. All Rights Reserved © 2021
"""

import argparse
import os
import sys
import time
from collections import namedtuple

from msquickcmp.atc.atc_utils import AtcUtils
from msquickcmp.common import utils
from msquickcmp.common.utils import AccuracyCompareException, get_shape_to_directory_name
from msquickcmp.net_compare import analyser
from msquickcmp.net_compare.net_compare import NetCompare
from msquickcmp.npu.npu_dump_data import NpuDumpData
from msquickcmp.npu.npu_dump_data_bin2npy import data_convert
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.npu.om_parser import OmParser


DumpDataInfo = namedtuple('DumpDataInfo', 'golden_dump_data_path, golden_net_output_info, npu_dump_data_path, \
        npu_net_output_data_path, expect_net_output_node')


def _generate_golden_data_model(args):
    model_name, extension = utils.get_model_name_and_extension(args.model_path)
    if ".pb" == extension:
        from msquickcmp.tf.tf_dump_data import TfDumpData
        return TfDumpData(args)
    elif ".onnx" == extension:
        from msquickcmp.onnx_model.onnx_dump_data import OnnxDumpData
        return OnnxDumpData(args)
    else:
        utils.logger.error("Only model files whose names end with .pb or .onnx are supported")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_MODEL_TYPE_ERROR)


def _correct_the_wrong_order(left_index, right_index, golden_net_output_info):
    if left_index not in golden_net_output_info.keys() or right_index not in golden_net_output_info.keys():
        return
    if left_index != right_index:
        tmp = golden_net_output_info[left_index]
        golden_net_output_info[left_index] = golden_net_output_info[right_index]
        golden_net_output_info[right_index] = tmp
        utils.logger.info("swap the {} and {} item in golden_net_output_info!"
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
            utils.logger.warning("the original name: {} of net output maybe not correct!".format(node_name))
            break


def cmp_process(args:CmpArgsAdapter, use_cli:bool):
    """
    Function Description:
        main process function
    Exception Description:
        exit the program when an AccuracyCompare Exception  occurs
    """
    args.model_path = os.path.realpath(args.model_path)
    args.offline_model_path = os.path.realpath(args.offline_model_path)
    args.cann_path = os.path.realpath(args.cann_path)
    try:
        check_and_run(args, use_cli)
    except utils.AccuracyCompareException as error:
        raise error


def dump_data(args, output_json_path, use_cli):
    # generate dump data by the original model
    golden_dump = _generate_golden_data_model(args)
    golden_dump_data_path = golden_dump.generate_dump_data()
    golden_net_output_info = golden_dump.get_net_output_info()

    # compiling and running source codes
    npu_dump = NpuDumpData(args, output_json_path)
    npu_dump_data_path, npu_net_output_data_path = npu_dump.generate_dump_data(use_cli)
    expect_net_output_node = npu_dump.get_expect_output_name()

    # if it's dynamic batch scenario, golden data files should be renamed
    utils.handle_ground_truth_files(npu_dump.om_parser, npu_dump_data_path, golden_dump_data_path)

    dump_data_info = DumpDataInfo(golden_dump_data_path, golden_net_output_info, npu_dump_data_path, \
                                  npu_net_output_data_path, expect_net_output_node)

    return dump_data_info


def dump_data_aipp(args, output_json_path, use_cli):
    npu_dump = NpuDumpData(args, output_json_path)
    npu_dump.generate_input_data()
    npu_dump_data_path, npu_net_output_data_path = npu_dump.generate_dump_data(use_cli)
    expect_net_output_node = npu_dump.get_expect_output_name()

    golden_dump = _generate_golden_data_model(args)
    golden_dump_data_path = golden_dump.generate_dump_data_aipp(npu_dump_data_path)
    golden_net_output_info = golden_dump.get_net_output_info()

    # if it's dynamic batch scenario, golden data files should be renamed
    utils.handle_ground_truth_files(npu_dump.om_parser, npu_dump_data_path, golden_dump_data_path)

    dump_data_info = DumpDataInfo(golden_dump_data_path, golden_net_output_info, npu_dump_data_path, \
        npu_net_output_data_path, expect_net_output_node)

    return dump_data_info


def run(args, input_shape, output_json_path, original_out_path, use_cli:bool, aipp_om):
    if input_shape:
        args.input_shape = input_shape
        args.out_path = os.path.join(original_out_path, get_shape_to_directory_name(args.input_shape))

    if aipp_om:
        dump_data_info = dump_data_aipp(args, output_json_path, use_cli)
    else:
        dump_data_info = dump_data(args, output_json_path, use_cli)

    # convert data from bin to npy if --convert is used
    data_convert(dump_data_info.npu_dump_data_path, dump_data_info.npu_net_output_data_path, args)

    if not args.dump:
        # only compare the final output
        net_compare = NetCompare(dump_data_info.npu_net_output_data_path,
                                 dump_data_info.golden_dump_data_path, output_json_path, args)
        net_compare.net_output_compare(dump_data_info.npu_net_output_data_path, dump_data_info.golden_net_output_info)
    else:
        # compare the entire network
        net_compare = NetCompare(dump_data_info.npu_dump_data_path,
                                 dump_data_info.golden_dump_data_path, output_json_path, args)
        net_compare.accuracy_network_compare()
    # Check and correct the mapping of net output node name.
    if len(dump_data_info.expect_net_output_node) == 1:
        _check_output_node_name_mapping(dump_data_info.expect_net_output_node, dump_data_info.golden_net_output_info)
        net_compare.net_output_compare(dump_data_info.npu_net_output_data_path, dump_data_info.golden_net_output_info)
    analyser.Analyser(args.out_path)()


def check_and_run(args:CmpArgsAdapter, use_cli:bool):
    utils.check_file_or_directory_path(args.model_path)
    utils.check_file_or_directory_path(args.offline_model_path)
    utils.check_device_param_valid(args.device)
    utils.check_file_or_directory_path(os.path.realpath(args.out_path), True)
    utils.check_convert_is_valid_used(args.dump, args.bin2npy)
    time_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())
    original_out_path = os.path.realpath(os.path.join(args.out_path, time_dir))
    args.out_path = original_out_path

    # convert the om model to json
    output_json_path = AtcUtils(args).convert_model_to_json()
    temp_om_parser = OmParser(output_json_path)
    aipp_om = True if temp_om_parser.get_aipp_config_content() else False

    # deal with the dymShape_range param if exists
    input_shapes = []
    if args.dym_shape_range:
        input_shapes = utils.parse_dym_shape_range(args.dym_shape_range)
    if not input_shapes:
        input_shapes.append("")
    for input_shape in input_shapes:
        run(args, input_shape, output_json_path, original_out_path, use_cli, aipp_om)
