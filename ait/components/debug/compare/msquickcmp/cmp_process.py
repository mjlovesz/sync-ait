# coding=utf-8
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

"""
Function:
This class mainly involves the main function.
"""

import argparse
import os
import stat
import shutil
import time
import subprocess
import csv
import logging
import site
import pandas as pd

import sklearn as _ # import first, bypassing error libgomp-xxx.so.xxx: cannot allocate memory in static TLS block
import onnxruntime
import acl

from auto_optimizer import OnnxGraph
from msquickcmp.atc import atc_utils
from auto_optimizer.graph_refactor import Node
from msquickcmp.common import utils
from msquickcmp.common.utils import AccuracyCompareException, get_shape_to_directory_name, safe_delete_path_if_exists
from msquickcmp.common.convert import convert_bin_dump_data_to_npy
from msquickcmp.common.convert import convert_npy_to_bin
from msquickcmp.net_compare import analyser
from msquickcmp.net_compare.net_compare import NetCompare
from msquickcmp.npu.npu_dump_data import NpuDumpData, DynamicInput
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.npu.om_parser import OmParser
from msquickcmp.accuracy_locat import accuracy_locat as al
from msquickcmp.single_op import single_op as sp
from components.utils.security_check import check_write_directory

WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
READ_WRITE_FLAGS = os.O_RDWR | os.O_CREAT
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
ERROR_INTERVAL_INFO_FILE = "error_interval_info.txt"
MAX_MEMORY_USE = 6 * 1024 * 1024 * 1024


def _generate_golden_data_model(args, npu_dump_npy_path):
    model_name, extension = utils.get_model_name_and_extension(args.model_path)
    if args.weight_path and ".prototxt" == extension:
        from msquickcmp.caffe_model.caffe_dump_data import CaffeDumpData

        return CaffeDumpData(args)
    elif ".pb" == extension:
        from msquickcmp.tf.tf_dump_data import TfDumpData

        return TfDumpData(args)
    elif ".onnx" == extension:
        from msquickcmp.onnx_model.onnx_dump_data import OnnxDumpData

        return OnnxDumpData(args, npu_dump_npy_path)
    elif ".om" == extension:
        return NpuDumpData(arguments=args, is_golden=True)

    else:
        utils.logger.error("Only model files whose names end with .pb or .onnx or .prototxt are supported")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_MODEL_TYPE_ERROR)


def _correct_the_wrong_order(left_index, right_index, golden_net_output_info):
    if left_index not in golden_net_output_info.keys() or right_index not in golden_net_output_info.keys():
        return
    if left_index != right_index:
        tmp = golden_net_output_info[left_index]
        golden_net_output_info[left_index] = golden_net_output_info[right_index]
        golden_net_output_info[right_index] = tmp
        utils.logger.info('swap the %s and %s item in golden_net_output_info!', left_index, right_index)


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


def _get_single_csv_in_folder(csv_path):
    for file_name in os.listdir(csv_path):
        if file_name.endswith('.csv'):
            return os.path.join(csv_path, file_name)
    raise IOError(f"None csv file exists in folder {csv_path}")


def _append_is_npu_ops_to_csv(csv_path):
    csv_path = _get_single_csv_in_folder(csv_path)
    if os.path.islink(csv_path):
        os.unlink(csv_path)
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        header = rows[0]
        ground_truth_col = header.index("GroundTruth")
        header.append('IsNpuOps')
        for row in rows[1:]:
            is_npu_ops = "YES" if row[ground_truth_col] == "*" else "NO"
            row.append(is_npu_ops)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)


def mindir_to_om_process(args: CmpArgsAdapter):
    is_mindir_compare_accuracy = False
    model_path_ext = os.path.splitext(args.model_path)[-1]
    offline_model_path_ext = os.path.splitext(args.offline_model_path)[-1]
    if model_path_ext in [".onnx", ] and offline_model_path_ext in [".mindir", ]:
        is_mindir_compare_accuracy = True
        LD_PRELOAD = "LD_PRELOAD"
        ld_preload = os.getenv(LD_PRELOAD, "")
        save_om_so_path = os.path.join(site.getsitepackages()[0], "msquickcmp", "libsaveom.so")
        if not os.path.exists(save_om_so_path):
            utils.logger.error("libsaveom.so not found, check the installation process")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
        if not os.getenv(LD_PRELOAD):
            os.environ[LD_PRELOAD] = save_om_so_path
        elif (save_om_so_path not in os.getenv(LD_PRELOAD).split(":")):
            if len(ld_preload):
                os.environ[LD_PRELOAD] = save_om_so_path + ":" + ld_preload
            else:
                os.environ[LD_PRELOAD] = save_om_so_path

        utils.logger.info("Using {}".format(save_om_so_path))

        pwd = os.path.realpath("./")
        check_write_directory(pwd, check_user_stat=True)
        command = [
            "benchmark",
            "--modelFile={}".format(args.offline_model_path),
            "--device=Ascend"
        ]
        utils.execute_command(command)

        cur_om_path = os.path.join(pwd, ".om")
        if not os.path.exists(cur_om_path):
            utils.logger.info("om model not found, please check.")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)

        args.offline_model_path = os.path.realpath(os.path.join(pwd, ".mslite.om"))
        shutil.move(cur_om_path, args.offline_model_path)

    return is_mindir_compare_accuracy


def cmp_process(args: CmpArgsAdapter, use_cli: bool):
    """
    Function Description:
        main process function
    Exception Description:
        exit the program when an AccuracyCompare Exception  occurs
    """
    args.model_path = os.path.realpath(args.model_path)
    args.weight_path = os.path.realpath(args.weight_path) if args.weight_path else None
    args.offline_model_path = os.path.realpath(args.offline_model_path)
    args.cann_path = os.path.realpath(args.cann_path)
    args.input_path = convert_npy_to_bin(args.input_path)

    is_mindir_compare_accuracy = mindir_to_om_process(args)

    try:
        check_and_run(args, use_cli)
        if is_mindir_compare_accuracy:
            safe_delete_path_if_exists(args.offline_model_path)
    except utils.AccuracyCompareException as error:
        if is_mindir_compare_accuracy:
            safe_delete_path_if_exists(args.offline_model_path)
        raise error


def run(args:CmpArgsAdapter, input_shape, original_out_path, use_cli: bool):
    if input_shape:
        args.input_shape = input_shape
        args.out_path = os.path.join(original_out_path, get_shape_to_directory_name(args.input_shape))

    # whether use aipp
    output_json_path = atc_utils.convert_model_to_json(args.cann_path, args.offline_model_path, args.out_path)
    golden_json_path = None
    if args.model_path.endswith('.om'):
        golden_json_path = atc_utils.convert_model_to_json(args.cann_path, args.model_path, args.out_path)

    temp_om_parser = OmParser(output_json_path)
    use_aipp = True if temp_om_parser.get_aipp_config_content() else False

    if use_aipp and args.fusion_switch_file is not None:
        utils.logger.error("if .om model is using aipp config, --fusion-switch-file arg is not support.")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)

    npu_dump = NpuDumpData(args, is_golden=False)

    # generate npu inputs data
    npu_dump.generate_inputs_data(use_aipp=use_aipp)

    # generate npu dump data
    npu_dump_data_path, npu_net_output_data_path = npu_dump.generate_dump_data(use_cli=use_cli)

    # convert data from bin to npy if --convert is used, or if custom_op is not empty
    if args.bin2npy or args.custom_op != "":
        npu_dump_npy_path = convert_bin_dump_data_to_npy(npu_dump_data_path, npu_net_output_data_path, args.cann_path)
    else:
        npu_dump_npy_path = ""

    # generate onnx inputs data
    golden_dump = _generate_golden_data_model(args, npu_dump_npy_path)
    golden_dump.generate_inputs_data(npu_dump_data_path, use_aipp)

    expect_net_output_node = npu_dump.get_expect_output_name()

    # generate dump data by golden model
    golden_dump_data_path = golden_dump.generate_dump_data(npu_dump_npy_path, npu_dump.om_parser)
    golden_net_output_info = golden_dump.get_net_output_info()

    # if it's dynamic batch scenario, golden data files should be renamed
    utils.handle_ground_truth_files(npu_dump.om_parser, npu_dump_data_path, golden_dump_data_path)
    
    if not args.dump:
        # only compare the final output
        net_compare = NetCompare(npu_net_output_data_path, golden_dump_data_path, 
                                 output_json_path, args, golden_json_path)
        net_compare.net_output_compare(npu_net_output_data_path, golden_net_output_info)
    else:
        # compare the entire network
        net_compare = NetCompare(npu_dump_data_path, golden_dump_data_path, 
                                 output_json_path, args, golden_json_path)
        net_compare.accuracy_network_compare()
    
    # Check and correct the mapping of net output node name.
    if len(expect_net_output_node) == 1:
        _check_output_node_name_mapping(expect_net_output_node, golden_net_output_info)
    if not args.locat:
        invalid_rows, _ = analyser.Analyser(args.out_path)()
    else:
        invalid_rows, _ = analyser.Analyser(args.out_path)('ALL_INVALID')
    print_advisor_info(args.out_path)
    _append_is_npu_ops_to_csv(args.out_path)
    return invalid_rows


def print_advisor_info(out_path):
    advisor_info_txt_path = os.path.join(out_path, 'advisor_summary.txt')
    if os.path.exists(advisor_info_txt_path):
        utils.logger.info(f"The advisor summary (.txt) is saved in :\"{advisor_info_txt_path}\"")
        with open(advisor_info_txt_path, 'r') as advisor_file:
            lines = advisor_file.readlines()
            for line in lines:
                utils.logger.info(line.strip())


def fusion_close_model_convert(args:CmpArgsAdapter):
    if args.fusion_switch_file:
        args.fusion_switch_file = os.path.realpath(args.fusion_switch_file)
        utils.check_file_or_directory_path(args.fusion_switch_file)
        
        om_json_path = atc_utils.convert_model_to_json(args.cann_path, args.offline_model_path, args.out_path)
        om_parser = OmParser(om_json_path)
        atc_input_shape_in_offline_model = DynamicInput.get_input_shape_from_om(om_parser)

        close_fusion_om_file = os.path.join(args.out_path, 'close_fusion_om_model')
        atc_command_file_path = atc_utils.get_atc_path(args.cann_path)
        atc_cmd = [atc_command_file_path, "--framework=5", 
                   "--soc_version=" + acl.get_soc_name(), 
                   "--model=" + args.model_path, 
                   "--output=" + close_fusion_om_file, 
                   "--fusion_switch_file=" + args.fusion_switch_file]
        if atc_input_shape_in_offline_model:
            atc_cmd.append("--input_shape=" + atc_input_shape_in_offline_model)

        utils.execute_command(atc_cmd)
        args.model_path = close_fusion_om_file + ".om"


def check_and_run(args: CmpArgsAdapter, use_cli: bool):
    utils.check_file_or_directory_path(args.model_path)
    utils.check_file_or_directory_path(args.offline_model_path)
    if args.weight_path:
        utils.check_file_or_directory_path(args.weight_path)
    utils.check_device_param_valid(args.device)
    utils.check_file_or_directory_path(os.path.realpath(args.out_path), True)
    utils.check_convert_is_valid_used(args.dump, args.bin2npy, args.custom_op)
    utils.check_locat_is_valid(args.dump, args.locat)
    sp.check_single_op_is_valid(args.single_op, args.dump, args.custom_op, args.locat)
    utils.check_max_size_param_valid(args.max_cmp_size)

    time_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())
    original_out_path = os.path.realpath(os.path.join(args.out_path, time_dir))
    args.out_path = original_out_path

    fusion_close_model_convert(args)

    # deal with the dymShape_range param if exists
    input_shapes = []
    if args.dym_shape_range:
        input_shapes = utils.parse_dym_shape_range(args.dym_shape_range)
    if not input_shapes:
        input_shapes.append("")
    for input_shape in input_shapes:
        res = run(args, input_shape, original_out_path, use_cli)
        if args.single_op:
            single_op_compare(args, input_shape)
            continue
        if res and args.locat:
            endnode_names_list = res[0]["GroundTruth"].split(",")
            endnode_name = endnode_names_list[0]
            error_node_list = find_accuracy_interval(args, endnode_name, input_shape)
            error_interval_info_file = os.path.join(args.out_path, ERROR_INTERVAL_INFO_FILE)
            with os.fdopen(os.open(error_interval_info_file, READ_WRITE_FLAGS, WRITE_MODES), "a+") as fp_writer:
                output_error_interval_info(fp_writer, error_node_list)
    if args.dym_shape_range:
        csv_sum(original_out_path)


def single_op_compare(args, input_shape):
    # load onnx model
    og = OnnxGraph.parse(args.model_path)
    og.infer_shape()

    # set broken single operator onnx file path
    subgraph_onnx_file = os.path.join(args.out_path, "broken.onnx")
    sp.broken(og, subgraph_onnx_file)

    # load broken single operator onnx
    subog = OnnxGraph.parse(subgraph_onnx_file)
    single_op_dir = sp.generate_single_op_dir(args.out_path)
    memory_size = sp.get_memory_size_by_soc_type(args.device)

    # devide onnx into fixed size onnxs
    subonnx_list = sp.dynamic_divide_onnx(args.out_path, subog, memory_size)
    
    # set csv list
    csv_list = []

    # set golden dump data source file
    onnx_data_path = os.path.join(args.out_path, 'dump_data/onnx')
    
    # for each onnx run compare
    for idx, subonnx in enumerate(subonnx_list):
        # run atc to get om file
        subgraph_om_file = os.path.join(args.out_path, 'broken')
        sp.atc_conversion(subonnx, subgraph_om_file)

        # get onnx input data from golden dump data
        # load single operator onnx
        utils.logger.info("Start to loading input data")
        subog = OnnxGraph.parse(subonnx)
        
        # load onnx input description
        inputs_list = [(ii.name, ii.shape) for ii in onnxruntime.InferenceSession(subonnx).get_inputs()]

        # find all the data needed
        input_need_list = al.input_completion(og, inputs_list)
        pattern = '|'.join(input_need_list)
        try:
            matched_files = al.find_npy_files_with_prefix(onnx_data_path, pattern)
        except Exception as e:
            utils.logger.error("Failed to find onnx dump data, please check whether file path is right")
            raise AccuracyCompareException(utils.ACCRACY_COMPARISON_FETCH_DATA_ERROR) from e
        sort_matched_files = []
        for prefix in input_need_list:
            for match_file in matched_files:
                file_name = os.path.basename(match_file)
                if file_name.startswith(prefix):
                    sort_matched_files.append(match_file)
        bin_files_path = al.create_bin_file(args.out_path, sort_matched_files)
        tmp_bin_path = os.path.join(args.out_path, 'tmp')
        utils.logger.info("Loading data Finished!")

        # set single op output data
        tmp_out_path = os.path.join(single_op_dir, f"single_op_{idx}")
        os.makedirs(tmp_out_path)
        time_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())
        original_out_path = os.path.realpath(os.path.join(args.out_path, time_dir))

        # set compare run args
        cmg_args = CmpArgsAdapter(subonnx, os.path.join(args.out_path, "broken.om"),
                                "", bin_files_path, args.cann_path, tmp_out_path, "", args.device,
                                "", "", False, "", True, False, custom_op="", locat=False, single_op=True)

        utils.logger.info("Start to run comparision")

        # run compare
        utils.logger.setLevel(logging.ERROR)
        res = run(cmg_args, input_shape, original_out_path, True)
        utils.logger.setLevel(logging.INFO)
        csv_list.extend(sp.find_all_csv(tmp_out_path))
        utils.logger.info("Comparision finished")
        # remove temp bin files
        shutil.rmtree(tmp_bin_path)
    
    # merge csv
    summary_csv_path = utils.merge_csv(csv_list, single_op_dir, 'single_op_summary.csv')
    # analyze csv and print
    analyser.Analyser(summary_csv_path)()


def output_error_interval_info(fp_writer, error_node_list):
    for [l_node, r_node] in error_node_list:
        fp_writer.write(f"{l_node}:{r_node}")


def find_accuracy_interval(args, endnode_name, input_shape):
    """
    Function:
        find accuracy interval of the error node
    Return:
        an error node interval list
    """
    if input_shape:
        args.out_path = os.path.join(args.out_path, get_shape_to_directory_name(input_shape))

    # 读入onnx数据文件的路径
    onnx_file_path = 'dump_data/onnx'
    onnx_data_path = os.path.join(args.out_path, onnx_file_path)

    # 读入onnx模型
    og = OnnxGraph.parse(args.model_path)
    og.infer_shape()

    # 获取精度异常节点
    endnode = og.get_node(endnode_name, node_type=Node)

    output_file = './accuracy_location_log.txt'
    output_file = os.path.realpath(output_file)
    error_node_list = []
    # 验证单层算子是否有问题
    node_interval = [endnode, endnode]
    # 单层算子无问题
    if not subgraph_check(og, node_interval, args, onnx_data_path, input_shape):
        for node in og.nodes:
            if al.check_input_node(og, node):
                input_node_interval = [node, endnode]
                l_node, r_node = bin_divide(og, input_node_interval, args, onnx_data_path, input_shape)
                utils.logger.info("Accumulated Error interval has been found.")
                error_node_list.append([l_node, r_node])
        return error_node_list
    return [[endnode, endnode]]


def subgraph_check(og, node_interval, args, onnx_data_path, input_shape):
    startnode, endnode = node_interval
    subgraph_onnx_file = os.path.join(args.out_path, 'tmp_for_accuracy_locat.onnx')
    try:
        og.extract_subgraph([startnode.name], [endnode.name], subgraph_onnx_file)
    except Exception as e:
        utils.logger.error("Failed to extract subgraph model")
        raise AccuracyCompareException(utils.ACCRACY_COMPARISON_EXTRACT_ERROR) from e
    utils.logger.info("Extracting model Sucess!")
    utils.logger.info("Start using atc to convert onnx to om file")
    subgraph_om_file = os.path.join(args.out_path, 'tmp_for_accuracy_locat')
    atc_cmd = ["atc", "--framework=5", "--soc_version=" + acl.get_soc_name(), "--model=" + subgraph_onnx_file, \
               "--output=" + subgraph_om_file]
    subprocess.run(atc_cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    utils.logger.info("atc conversion Success!")
    utils.logger.info("Start to loading input data")
    subog = OnnxGraph.parse(subgraph_onnx_file)
    inputs_list = [(ii.name, ii.shape) for ii in onnxruntime.InferenceSession(subgraph_onnx_file).get_inputs()]
    input_need_list = al.input_completion(og, inputs_list)
    pattern = '|'.join(input_need_list)
    try:
        matched_files = al.find_npy_files_with_prefix(onnx_data_path, pattern)
    except Exception as e:
        utils.logger.error("Failed to find onnx dump data, please check whether file path is right")
        raise AccuracyCompareException(utils.ACCRACY_COMPARISON_FETCH_DATA_ERROR) from e
    sort_matched_files = []
    for prefix in input_need_list:
        for match_file in matched_files:
            file_name = os.path.basename(match_file)
            if file_name.startswith(prefix):
                sort_matched_files.append(match_file)
    bin_files_path = al.create_bin_file(args.out_path, sort_matched_files)
    tmp_bin_path = os.path.join(args.out_path, 'tmp')
    utils.logger.info("Loading data Finished!")
    tmp_out_path = os.path.join(args.out_path, 'tmpres')
    if not os.path.exists(tmp_out_path):
        os.makedirs(tmp_out_path)
    time_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())
    original_out_path = os.path.realpath(os.path.join(args.out_path, time_dir))
    cmg_args = CmpArgsAdapter(subgraph_onnx_file, os.path.join(args.out_path, "tmp_for_accuracy_locat.om"),
                              "", bin_files_path, args.cann_path, tmp_out_path, "", args.device,
                              "", "", False, "", True, False, custom_op=args.custom_op, locat=True)
    utils.logger.info("Start to run comparision")
    res = run(cmg_args, input_shape, original_out_path, True)
    utils.logger.info("Comparision finished")
    shutil.rmtree(tmp_out_path)
    shutil.rmtree(tmp_bin_path)
    if al.check_res(res, endnode):
        return True
    return False


def bin_divide(og, node_interval, args, onnx_data_path, input_shape):
    """
    Function:
        using binary search to find the accuracy error interval
    Return:
        an accuracy error interval list
    """
    startnode, endnode = node_interval
    subgraph_model_path = os.path.join(args.out_path, 'tmp_for_subgraph.onnx')
    og.extract_subgraph([startnode.name], [endnode.name], subgraph_model_path)
    subog = OnnxGraph.parse(subgraph_model_path)

    utils.logger.info("Binary Search for error interval starts.")
    # 直线化
    satisfied_nodes = []
    satisfied_nodes = al.calculate_flow(subog, startnode, endnode)
    low = 0
    high = len(satisfied_nodes) - 1

    # 二分
    while low < high:
        mid = (low + high + 1) // 2
        input_node_interval = [satisfied_nodes[mid], endnode]
        if subgraph_check(og, input_node_interval, args, onnx_data_path, input_shape):
            low = mid
        else:
            high = mid - 1
    utils.logger.info("Binary Search for error interval ends.")
    return satisfied_nodes[low], endnode


def csv_sum(original_out_path):
    """
    Function:
        Summarize csv files under different shapes
        genrate a xlsx file   
    """
    csv_file_list = []
    sheet_name_list = []

    for files in os.listdir(original_out_path):
        if not os.path.isdir(os.path.join(original_out_path, files)):
            continue
        for sub_file in os.listdir(os.path.join(original_out_path, files)):
            if sub_file.endswith(".csv"):
                csv_file_list.append(os.path.join(original_out_path, files, sub_file))
                sheet_name_list.append(files)

    xlsx_file_summary = os.path.join(original_out_path, "result_summary.xlsx")

    if os.path.exists(xlsx_file_summary):
        logging.error("Error, file already exists!")
        os.remove(xlsx_file_summary)

    with os.fdopen(os.open(xlsx_file_summary, WRITE_FLAGS, WRITE_MODES), 'wb') as fp_write:
        with pd.ExcelWriter(fp_write) as writer:
            for i, csv_file in enumerate(csv_file_list):
                data = pd.read_csv(csv_file, na_values=['NAN'])
                data.to_excel(writer, sheet_name=sheet_name_list[i], index=False, na_rep='NAN')