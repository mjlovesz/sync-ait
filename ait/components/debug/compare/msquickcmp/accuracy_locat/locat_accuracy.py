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
from msquickcmp.cmp_process import run

def find_accuracy_interval(args, endnode_name, input_shape):
    """

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
    """
    """
    #onnx临时文件，为切分子图后的模型文件
    subgraph_onnx_file = './tmp.onnx'
    subgraph_onnx_file = os.path.realpath(subgraph_onnx_file)
    og.extract_subgraph(startnode.name, endnode.name, subgraph_onnx_file)

    #执行atc转换
    atc_cmd = f"atc --framework=5 --soc_version={soc_version} --model={subgraph_onnx_file} --output=tmp"
    os.system(atc_cmd)

    #获得onnx与om模型后
    subog = OnnxGraph.parse(subgraph_onnx_file)
    input_need_list = []
    inputs_list = [(ii.name, ii.shape) for ii in onnxruntime.InferenceSession(subgraph_onnx_file).get_inputs()]
    input_need_list = input_completion(og, inputs_list)
    #按照需要读入所有需要的输入文件
    pattern = '|'.join(input_need_list)
    print(pattern)
    matched_files = find_npy_files_with_prefix(onnx_data_path, pattern)
    sort_matched_files = []
    for prefix in input_need_list:
        for match_file in matched_files:
            if match_file.startwith(prefix):
                sort_matched_files.append(match_file)
    bin_files_path = create_bin_file(sort_matched_files)
    subgraph_om_path = './tmp.om'
    model_path = os.path.realpath(subgraph_onnx_file)
    om_model = subgraph_om_path
    input_data_path = bin_files_path
    cann_path = args.cann_path
    tmp_out_path = './tmpres'
    tmp_out_path = os.path.realpath('./tmpres')
    if not os.path.exists(tmp_out_path):
        os.makedirs(tmp_out_path)
    out_path = tmp_out_path
    device = args.device
    output_size = ""
    output_nodes = ""
    advisor = False
    dym_shape_range = ""
    dump = True
    bin2npy = False
    time_dir = time.strftime("%Y%m%d%H%M%S",time.localtime())
    original_out_path = os.path.realpath(os.path.join(args.out_path, time_dir))
    cmg_args = CmpArgsAdapter(model_path, om_model, input_data_path, cann_path, out_path, "", device,
                              output_size, output_nodes, advisor, dym_shape_range, dump, bin2npy)
    output_json_path = AtcUtils(cmg_args).convert_model_to_json()
    res = run(cmg_args, input_shape, output_json_path, original_out_path, True)
    clr_cmd = 'rm -rf ./tmp/ ./tmpres/'
    os.system(clr_cmd)
    if check_res(res, endnode):
        return True
    return False


def bin_divide(og, startnode, endnode, args, soc_version, onnx_data_path, input_shape):
    og.extract_subgraph(startnode.name, endnode.name, './tmp.onnx')
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
    queue = deque([(startnode, flow[startnode.name])])
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
    pattern = r'^{}.*\.npy'.format(prefix)
    regex = re.compile(pattern)
    matched_files = []
    for root, _, files in os.walk(workdir):
        for file in files:
            if regex.match(file):
                matched_files.append(os.path.join(root, file))
    return matched_files


def create_bin_file(matched_files):
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
    """
    input_need_list = []
    index = 0
    for node_input in inputs_list:
        input_node = og.get_prev_node(node_input[0])
        if input_node is not None:
            input_need_list.append(f"{input_node.name}\.")
    input_need_list = list(OrderedDict.fromkeys(input_need_list))
    return input_need_list


def check_node_valid(incnt, graph, node):
    """
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
    input_cnt = 0
    for node_input in node.inputs:
        input_node = og.get_prev_node(node_input)
        if input_node is not None:
            input_cnt += 1
    if input_cnt == len(node.inputs):
        return True
    return False


def check_res(res, endnode):
    for row in res:
        for ground_truth_name in row["GroundTruth"]:
            if ground_truth_name == endnode.name:
                return True
    return False