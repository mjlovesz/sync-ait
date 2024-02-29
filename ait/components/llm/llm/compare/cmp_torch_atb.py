# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
from collections import Counter
from typing import List

from llm.compare.op_mapping import ATB_TORCH_BUILT_IN_OP_MAPPING
from llm.dump.torch_dump.topo import ModelTree, TreeNode
from llm.common.log import logger
from llm.compare.acc_cmp import fill_row_data, save_compare_dataframe_to_csv
from llm.common.constant import CSV_GOLDEN_HEADER

import pandas as pd


MIN_LAYER_NUMBER = 10


def search_layer_node(root_node: TreeNode):
    child_op_type = [child_node.node_type for child_node in root_node.children]
    if len(child_op_type) > MIN_LAYER_NUMBER:
        op_type_counts = Counter(child_op_type)
        most_count = op_type_counts.most_common(1)[0][1]
        if most_count > MIN_LAYER_NUMBER / 2:
            most_op_type = op_type_counts.most_common(1)[0][0]
            return most_op_type
    else:
        for child_node in root_node.children:
            search_layer_node(child_node)


def get_layer_node(root_node: TreeNode, layer_type: str, layer_nodes: List):
    for child_node in root_node.children:
        if child_node.node_type == layer_type:
            layer_nodes.append(child_node)
        else:
            get_layer_node(child_node, layer_type, layer_nodes)


def get_leaf_nodes(root_node, nodes):
    for child_node in root_node.children:
        if child_node.children:
            get_leaf_nodes(child_node)
        else:
            nodes.append(child_node)


def cmp_torch_atb_model(golden_json, my_json, output_path):
    compared_result = []
    golden_root_node = ModelTree.json_to_tree(golden_json)
    golden_layer_type = search_layer_node(golden_root_node)
    golden_layer_nodes = []
    get_layer_node(golden_root_node, golden_layer_type, golden_layer_nodes)

    my_root_node = ModelTree.json_to_tree(my_json)
    my_layer_type = search_layer_node(my_root_node)
    my_layer_nodes = []
    get_layer_node(my_root_node, my_layer_type, my_layer_nodes)

    for golden_layer, my_layer in zip(golden_layer_nodes, my_layer_nodes):
        g_layer_leaf_nodes = []
        get_leaf_nodes(golden_layer, g_layer_leaf_nodes)
        m_layer_leaf_nodes = []
        get_leaf_nodes(my_layer, m_layer_leaf_nodes)
        for atb_op_type, torch_op_type in ATB_TORCH_BUILT_IN_OP_MAPPING.items():
            atb_nodes = []
            torch_nodes = []
            for m_leaf_node in m_layer_leaf_nodes:
                if m_leaf_node.op_type == atb_op_type:
                    atb_nodes.append(m_leaf_node)
            for g_leaf_node in g_layer_leaf_nodes:
                if g_leaf_node.op_type == torch_op_type:
                    torch_nodes.append(g_leaf_node)
            if len(atb_nodes) != len(torch_nodes):
                logger.warning("The number of %s node in atb is not equal to %s node in torch",
                               atb_op_type, torch_op_type)
                continue
            for atb_node, torch_node in zip(atb_nodes, torch_nodes):
                my_tensor_path = os.path.join(atb_node.tensor_path, "after", "outtensor0.bin")
                golden_tensor_path = os.path.join(atb_node.tensor_path, "output_exec1.pth")
                row_data = fill_row_data(0, 0, golden_tensor_path, my_tensor_path)
                compared_result.append(row_data)

    data_frame = pd.DataFrame(compared_result, columns=CSV_GOLDEN_HEADER)
    save_compare_dataframe_to_csv(data_frame, output_path)
