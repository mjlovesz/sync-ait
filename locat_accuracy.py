from collections import deque
import re
import argparse
import os
import sys
import time

import numpy as np

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

    error_node_list = []
    #验证单层算子是否有问题
    #单层算子无问题
    if not subgraph_check(og, endnode, endnode, args, 'Ascend310P3', onnx_data_path, input_shape):
        for node in og.nodes:
            if og.get_prev_node(node.inputs[0]) is None:
                l_node, r_node = bin_divide(og, node, endnode, args, 'Ascend310P3', onnx_data_path, input_shape)
                error_node_list.append([l_node, r_node])
    return [endnode, endnode]


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
    input_need_list = set()
    for node in subog.nodes:
        for node_input in node.inputs:
            input_node = og.get_prev_node(node_input)
            if input_node is not None and input_node not in subog.nodes:
                input_need_list.add(f"{input_node.name}\.")

    #按照需要读入所有需要的输入文件
    pattern = '|'.join(input_need_list)
    print(pattern)
    matched_files = find_npy_files_with_prefix(onnx_data_path, pattern)
    bin_files_path = create_bin_file(matched_files)
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
    if len(res) != 0:
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
    lin = len(startnode.outputs)
    # 流量值
    flow = {}
    # 入度
    incnt = {}
    for node in graph.nodes:
        flow[node.name] = 0.0
        incnt[node.name] = len(node.inputs)
    flow[startnode.name] = lin
    satisfied_nodes = []
    visited = set()
    queue = deque([(startnode, flow[startnode.name])])
    visited.add(startnode)
    while queue:
        current_node, current_flow = queue.popleft()
        if current_flow == lin:
            satisfied_nodes.append(current_node)
        outdegree = len(current_node.outputs)
        if outdegree != 0:
            flow_increment = current_flow // outdegree
        for output_name in current_node.outputs:
            for next_node in graph.get_next_nodes(output_name):
                if next_node is not None:
                    flow[next_node.name] += flow_increment
                    incnt[next_node.name] -= 1
                if next_node is not None and incnt.get(next_node.name) == 0"
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