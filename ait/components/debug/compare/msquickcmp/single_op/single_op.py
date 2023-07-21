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

import os
import subprocess
import acl

from auto_optimizer import OnnxGraph
from auto_optimizer.graph_refactor import Node
from auto_optimizer.graph_refactor.onnx import OnnxNode, OnnxPlaceHolder, OnnxInitializer
from auto_optimizer.graph_refactor.interface import PlaceHolder
from msquickcmp.common import utils
from msquickcmp.common.utils import AccuracyCompareException


def get_memory_size_by_soc_type():
    npu_id = -1
    memory_size = -1
    pre_cmd = "npu-smi info -l"
    res = subprocess.run(pre_cmd.split(), shell=False, stdout=subprocess.PIPE)

    for line in res.stdout.decode().split('\n'):
        if "NPU ID" in line:
            npu_id = int(line.split()[-1])
            break
    
    if npu_id == -1:
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DEVICE_ERROR)
    
    cmd = f"npu-smi info -t memory -i {npu_id}"
    res = subprocess.run(cmd.split(), shell=False, stdout=subprocess.PIPE)
    
    for line in res.stdout.decode().split('\n'):
        if "DDR Capacity(MB)" in line:
            memory_size = int(line.split()[-1])
            break
    if memory_size == -1:
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DEVICE_ERROR)

    # get size by Byte Unit
    return memory_size // 4 * 1024 * 1024


def generate_single_op_dir(out_path):
    """
    generate the outputdir for single op comparision outputs
    """
    single_op_dir = os.path.join(out_path, 'single_op')
    if os.path.exists(single_op_dir):
        os.rmdir(dir_path)
    os.makedirs(single_op_dir)
    return single_op_dir


def broken(og: OnnxGraph, subgraph_onnx_file: str):
    """
    Function: break onnx into single operator pieces and keep in one onnx
    
    Input: og -> OnnxGraph, subgraph_onnx_file -> output onnx file path

    Output:  single operator pieces onnx file
    """
    g_inputs = [ph.name for ph in og.inputs]
    g_outputs = [ph.name for ph in og.outputs]

    input_name_list = []
    for node_idx, node in enumerate(og.nodes):
        in_ph_list = []
        for idx, inp in enumerate(node.inputs):
            if inp in g_inputs or og.get_node(inp, OnnxInitializer):
                continue
            ph = og.get_node(inp, PlaceHolder)
            if ph and inp not in input_name_list:
                in_placeholder = OnnxPlaceHolder(ph.name, ph.dtype, ph.shape)
                node.inputs[idx] = in_placeholder.name
                in_ph_list.append(in_placeholder)
                input_name_list.append(in_placeholder.name)
        
        out_ph_list = []
        for idx, out in enumerate(node.outputs):
            if out in g_outputs:
                continue
            new_name = f"out_{idx}_{node_idx}_{out}"
            ph = og.get_node(out, PlaceHolder)
            out_placeholder = OnnxPlaceHolder(new_name, ph.dtype, ph.shape)
            node.outputs[idx] = out_placeholder.name
            out_ph_list.append(out_placeholder)
        
        og.inputs.extend(in_ph_list)
        og.outputs.extend(out_ph_list)
    og.save(subgraph_onnx_file)


def find_all_csv(out_path):
    all_csv_list = []
    for f in os.listdir(out_path):
        if f.endswith('.csv'):
            all_csv_list.append(os.path.join(out_path, f))
    return all_csv_list


def atc_conversion(onnx_path, om_path):
    atc_cmd = ["atc", "--framework=5", "--soc_version=" + acl.get_soc_name(), "--model=" + onnx_path,\
                "--output=" + om_path]
    subprocess.run(atc_cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    utils.logger.info("atc conversion Success!")


def accumulate_shape_size(node, og):
    """
    Function: calculate the memory needed for the given node

    Input: node -> node description, og -> global OnnxGraph

    Output: the memory size needed for the given node
    """
    ans = 0
    for node_input in node.inputs:
        ph = og.get_node(node_input, PlaceHolder)
        shape_size = 1
        if ph:
            for shape in ph.shape:
                shape_size *= shape
            ans += ph.dtype.itemsize * shape_size
    for node_output in node.outputs:
        ph = og.get_node(node_output, PlaceHolder)
        shape_size = 1
        if ph:
            for shape in ph.shape:
                shape_size *= shape
            ans += ph.dtype.itemsize * shape_size
    return ans


def dynamic_divide_onnx(out_path: str, subog: OnnxGraph, memory_size: int):
    """
    Function:
    according to the patchsize to divide the given onnx into suitable size onnxs.
    
    Input:subog:OnnxGraph needed to be divided
    
    Output:
    Divided onnx list which contains a series of onnx paths
    """
    subonnx_list = []
    startnode_list = []
    endnode_list = []
    size_sum = 0
    idx = 0
    for idx, node in enumerate(subog.nodes):
        startnode_list.append(node.name)
        endnode_list.append(node.name)
        size_sum += accumulate_shape_size(node, subog)
        if size_sum >= memory_size:
            size_sum = 0
            subonnx_file_path = os.path.join(out_path, f"{idx}_broken.onnx")
            subog.extract_subgraph(startnode_list, endnode_list, subonnx_file_path)
            startnode_list.clear()
            endnode_list.clear()
            subonnx_list.append(subonnx_file_path)

    # process rest nodes
    if startnode_list:
        subonnx_file_path = os.path.join(out_path, f"{idx}_broken.onnx")
        subog.extract_subgraph(startnode_list, endnode_list, subonnx_file_path)
        subonnx_list.append(subonnx_file_path)
    return subonnx_list