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

from collections import deque
from collections import OrderedDict
import re
import argparse
import os
import sys
import time

import numpy as np
import onnxruntime

from auto_optimizer.graph_refactor import Node
from auto_optimizer import OnnxGraph
from msquickcmp.atc.atc_utils import AtcUtils
from msquickcmp.common import utils
from msquickcmp.net_compare import analyser
from msquickcmp.net_compare.net_compare import NetCompare
from msquickcmp.npu.npu_dump_data import NpuDumpData
from msquickcmp.npu.npu_dump_data_bin2npy import data_convert
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.cmp_process import run


def find_accuracy_interval(args, endnode_name, input_shape):
    """
    Function:
        find accuracy interval of the error node
    Return:
        an error node interval list
    """
    if input_shape:
        args.out_path = os.path.join(args.out_path, get_shape_to_directory_name(input_shape))
    
    #读入onnx数据文件的路径
    onnx_file_path = 'dump_data/onnx'
    onnx_data_path = os.path.join(args.out_path, onnx_file_path)

    #读入onnx模型
    og = OnnxGraph.parse(args.model_path)
    og.infer_shape()

    #获取精度异常节点
    endnode = og.get_node(endnode_name, node_type=Node)

    output_file = './accuracy_location_log.txt'
    output_file = os.path.realpath(output_file)
    error_node_list = []
    #验证单层算子是否有问题
    #单层算子无问题
    if not subgraph_check(og, endnode, endnode, args, 'Ascend310P3', onnx_data_path, input_shape):
        for node in og.nodes:
            if check_node_valid_normal(og, node):
                l_node, r_node = bin_divide(og, node, endnode, args, 'Ascend310P3', onnx_data_path, input_shape)
                error_node_list.append([l_node, r_node])
        return error_node_list
    return [[endnode, endnode]]


def subgraph_check(og, startnode, endnode, args, soc_version, onnx_data_path, input_shape):
    #onnx临时文件，为切分子图后的模型文件
    subgraph_onnx_file = './tmp.onnx'
    subgraph_onnx_file = os.path.realpath(subgraph_onnx_file)
    utils.logger.info(f"Start extracting subgraph model, model saved in {subgraph_onnx_file}")
    try:
        og.extract_subgraph([startnode.name], [endnode.name], subgraph_onnx_file)
    except Exception as e:
        utils.logger.error("Failed to extract subgraph model")
        raise AccuracyCompareException(utils.ACCRACY_COMPARISON_EXTRACT_ERROR) from e
    utils.logger.info("Extracting model Sucess!")
    utils.logger.info("Start using atc to convert onnx to om file")
    atc_cmd = f"atc --framework=5 --soc_version={soc_version} --model={subgraph_onnx_file} --output=tmp"
    os.system(atc_cmd)
    utils.logger.info("atc conversion Sucess!")
    #获得onnx与om模型后
    utils.logger.info("Start to loading input data")
    subog = OnnxGraph.parse(subgraph_onnx_file)
    input_need_list = []
    inputs_list = [(ii.name, ii.shape) for ii in onnxruntime.InferenceSession(subgraph_onnx_file).get_inputs()]
    input_need_list = input_completion(og, inputs_list)
    #按照需要读入所有需要的输入文件
    pattern = '|'.join(input_need_list)
    matched_files = find_npy_files_with_prefix(onnx_data_path, pattern)
    sort_matched_files = []
    for prefix in input_need_list:
        for match_file in matched_files:
            if match_file.startwith(prefix):
                sort_matched_files.append(match_file)
    bin_files_path = create_bin_file(sort_matched_files)
    utils.logger.info("Loading data Finished!")
    tmp_out_path = os.path.realpath('./tmpres')
    if not os.path.exists(tmp_out_path):
        os.makedirs(tmp_out_path)
    time_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())
    original_out_path = os.path.realpath(os.path.join(args.out_path, time_dir))
    cmg_args = CmpArgsAdapter(subgraph_onnx_file, subgraph_om_path, bin_files_path, 
                              args.cann_path, tmp_out_path, "", args.device,
                              args.output_size, args.output_nodes, False, "", True, False)
    output_json_path = AtcUtils(cmg_args).convert_model_to_json()
    utils.logger.info("Start to run comparision")
    res = run(cmg_args, input_shape, output_json_path, original_out_path, True)
    utils.logger.info("Comparision finished")
    clr_cmd = 'rm -rf ./tmp/ ./tmpres/'
    os.system(clr_cmd)
    if check_res(res, endnode):
        return True
    return False


def bin_divide(og, startnode, endnode, args, soc_version, onnx_data_path, input_shape):
    """
    Function:
        using binary search to find the accuracy error interval
    Return:
        an accuracy error interval list
    """
    og.extract_subgraph([startnode.name], [endnode.name], './tmp.onnx')
    subog = OnnxGraph.parse('./tmp.onnx')

    # 直线化
    satisfied_nodes = []
    satisfied_nodes = calculate_flow(subog, startnode, endnode)
    low = 0
    high = len(satisfied_nodes) - 1

    #二分
    while low < high:
        mid = (low + high + 1) // 2
        if subgraph_check(og, satisfied_nodes[mid], endnode, args, soc_version, onnx_data_path, input_shape):
            low = mid
        else:
            high = mid - 1
    return satisfied_nodes[low], endnode


def calculate_flow(graph, startnode, endnode):
    """
    Function:
        simplifying the graph by using flow calculation to a linear node list
    Return:
        a node list which is linear
    """
    #误差限
    eps = 1e-10
    lin = 0
    for output_name in startnode.outputs:
        for next_node in graph.get_next_nodes(output_name):
            if next_node is not None:
                lin += 1
    if lin < 512:
        lin *= 512
    
    flow = {}
    incnt = {}
    for node in graph.nodes:
        flow[node.name] = float(0)
        incnt[node.name] = len(node.inputs)
    flow[startnode.name] = float(lin)
    satisfied_nodes = []
    visited = set()
    queue = deque([(startnode, flow.get(startnode.name))])
    visited.add(startnode)
    while queue:
        current_node, current_flow = queue.popleft()
        if abs(current_flow - lin) < eps:
            satisfied_nodes.append(current_node)
        outdegree = 0
        for output_name in current_node.outputs:
            for next_node in graph.get_next_nodes(output_name):
                if next_node is not None:
                    outdegree += 1
        
        if outdegree != 0:
            flow_increment = float(current_flow) / float(outdegree)
        for output_name in current_node.outputs:
            for next_node in graph.get_next_nodes(output_name):
                if next_node is not None:
                    flow[next_node.name] += flow_increment
                    incnt[next_node.name] -= 1
                if next_node is not None and check_node_valid(incnt, graph, next_node):
                    queue.append([next_node, flow.get(next_node.name)])
                    visited.add(next_node)
    return satisfied_nodes


def find_npy_files_with_prefix(workdir, prefix):
    """
    Function:
        according given prefix list, find all the satisfied files
    Return:
        a matching file path list
    """
    pattern = r'^{}.*\.npy'.format(prefix)
    regex = re.compile(pattern)
    matched_files = []
    for root, _, files in os.walk(workdir):
        for file in files:
            if regex.match(file):
                matched_files.append(os.path.join(root, file))
    return matched_files


def create_bin_file(matched_files):
    """
    Function:
        convert all the matched_files in npy format
        to bin format
    Return:
        bin file path list
    """
    bin_files_list = ""
    bin_file_path = './tmp'
    bin_file_path = os.path.realpath(bin_file_path)
    if not os.path.exists(bin_file_path):
        os.makedirs(bin_file_path)
    for i, npy_file in enumerate(matched_files):
        data = np.load(npy_file)
        bin_file_name = f'{i}.bin'
        bin_file = os.path.join(bin_file_path, bin_file_name)
        if i != 0:
            bin_file_list += ","+bin_file
        else:
            bin_file_list += bin_file
        data.tofile(bin_file)
    return bin_files_list


def input_completion(og, inputs_list):
    """
    Function:
        find all the inputs needed according to inputs_list
        generate a need list
    Return:
        return a need file name list
    """
    input_need_list = []
    index = 0
    for node_input in inputs_list:
        input_node = og.get_prev_node(node_input[0])
        if input_node is None:
            continue
        for i, pre_input in enumerate(input_node.inputs):
            if pre_input == node_input[0]:
                index = i
                break
        input_need_list.append(f"{input_node.name}\.{index}\.")
    input_need_list = list(OrderedDict.fromkeys(input_need_list))
    return input_need_list


def check_node_valid(incnt, graph, node):
    """
    Function:
        check node is the current input node in graph
        using incnt to present the incount of node
    Return:
        true if node is the current input node of graph
        false otherwise
    """
    if incnt.get(node.name) == 0:
        return True
    else:
        emp_cnt = 0
        for node_input in node.inputs:
            input_node = graph.get_prev_node(node_input)
            if input_node is not None:
                emp_cnt += 1
        if emp_cnt == incnt.get(node.name):
            return True
    return False


def check_node_valid_normal(og, node):
    """
    Function:
        check node is an input node in model og
    Return:
        true if check node is an input node in model og
        false otherwise
    """
    input_cnt = 0
    for node_input in node.inputs:
        input_node = og.get_prev_node(node_input)
        if input_node is not None:
            input_cnt += 1
    if input_cnt == len(node.inputs):
        return True
    return False


def check_res(res, endnode):
    """
    check result rows
    check error is relative to endnode
    """
    for row in res:
        for ground_truth_name in row["GroundTruth"]:
            if ground_truth_name == endnode.name:
                return True
    return False