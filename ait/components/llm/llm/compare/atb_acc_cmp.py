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
import re
import glob
import json

from llm.compare.cmp_utils import compare_data, read_data
from llm.common.log import logger
from llm.compare.cmp_utils import BasicDataInfo, fill_row_data, save_compare_reault_to_csv
from llm.compare.op_mapping import ATB_TORCH_BUILT_IN_OP_OUTPUT_MAPPING, ATB_TORCH_CUSTOM_OP_OUTPUT_MAPPING
from llm.dump.torch_dump.topo import ModelTree

from tqdm import tqdm


def acc_compare(golden_path, my_path, output_path=".", mapping_file_path="."):
    if os.path.isdir(golden_path):
        golden_tensor_path = os.path.join(golden_path, "golden_tensor")
        golden_topo_flag, golden_topo_json_path = is_model_topo_exist(golden_path)
        my_topo_flag, my_topo_json_path = is_model_topo_exist(my_path)
        torch_model_topo_file = os.path.join(golden_path, "..", "model_tree.json")
        if os.path.isdir(golden_tensor_path):
            # 存在golden_tensor路径，走手动映射比对逻辑
            logger.info("Manual mapping comparing starts! Comparing manual dump tensors and ATB tensors...")
            compare_metadata(golden_tensor_path, output_path)
        elif os.path.exists(torch_model_topo_file):
            # 存在torch_model_topo_file路径，走torch模型和加速库模型比对逻辑
            logger.info("Automatic mapping comparison starts! Comparing torch tensors and ATB tensors...")
            cmp_torch_atb(torch_model_topo_file, golden_path, my_path, output_path, mapping_file_path)
        elif golden_topo_flag and my_topo_flag:
            # 存在模型的拓扑信息，走加速库模型间的比对逻辑
            if compare_topo_json(golden_topo_json_path, my_topo_json_path):
                # topo信息一致，走dtype和bs比对逻辑：
                logger.info("Automatic mapping comparison starts! Comparing ATB tensors, the topos are same...")
                compare_atb_metadata_auto(golden_path, my_path, golden_topo_json_path, my_topo_json_path, output_path)
            else:
                # topo信息不一致，走量化比对逻辑，待补充
                logger.info('Automatic mapping comparison starts! Comparing ATB tensors, the topos are different...')

    elif os.path.isfile(golden_path) and os.path.isfile(my_path):
        res = compare_file(golden_path, my_path)
        logger.info("Compared results: %s", res)
    else:
        logger.error("The golden_path and my_path must both be directory or file.")
        exit(1)


def is_model_topo_exist(golden_path):
    # 判断用户输入路径的ait_dump目录下是否包括/model路径，即是否包括模型拓扑信息
    absolute_path = os.path.abspath(golden_path)
    model_dir_path = os.path.join(absolute_path, '../../../', 'model')
    model_dir_path = os.path.normpath(model_dir_path)
    if not os.path.isdir(model_dir_path):
        msg = f"Cannot find {model_dir_path}, please check! Use ait llm dump if needed."
        logger.info(msg)
        return False, ""
    # 搜索/model目录下的所有文件，查找JSON文件
    for root, _, files in os.walk(model_dir_path):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                return True, json_file_path
    # 如果没有找到json文件，返回False和空字符串
    msg = f"Cannot find model topo json in {model_dir_path}, please check! Use ait llm dump if needed."
    logger.info(msg)
    return False, ""


def compare_topo_json(golden_topo_json_path, my_topo_json_path):
    try:
        with open(golden_topo_json_path, 'r') as golden_file:
            golden_data = json.load(golden_file)
        with open(my_topo_json_path, 'r') as my_file:
            my_data = json.load(my_file)
        if golden_data == my_data:
            return True
        else:
            return False
    except (IOError, json.JSONDecodeError):
        return False


def compare_file(golden_path, my_path):
    golden_data = read_data(golden_path)
    my_data = read_data(my_path)
    return compare_data(golden_data, my_data)


# 手动映射比对能力
def compare_metadata(golden_path, output_path="."):
    golden_meta_path = os.path.join(golden_path, "metadata.json")
    with open(golden_meta_path, "r") as file:
        golden_meta = json.load(file)
    gathered_row_data = fill_in_data(golden_meta)
    return save_compare_reault_to_csv(gathered_row_data, output_path)


def fill_in_data(golden_meta):
    gathered_row_data = []
    for data_id, golden_info in tqdm(golden_meta.items(), total=len(golden_meta)):
        for token_id, path_list in golden_info.items():

            # 读取映射关系json文件中的tenor路径
            if not isinstance(path_list, (list, tuple)) or len(path_list) < 2:
                logger.warning(f"Invalid data in golden metadata.json, data_id: {data_id}, token_id: {token_id}")
                continue
            data_info = BasicDataInfo(path_list[0], path_list[1], token_id, data_id)
            row_data = fill_row_data(data_info)
            gathered_row_data.append(row_data)
    return gathered_row_data


def traverse_tree(node: dict, path, traverse_type='torch', node_id=''):
    def enumerate_children(children, path, traverse_type='torch_model', node_id=''):
        res = []
        for idx, children_node in enumerate(children):
            if node_id != '':
                res.extend(traverse_tree(children_node, path, traverse_type, node_id + f'_{idx}'))
            else:
                res.extend(traverse_tree(children_node, path, traverse_type, str(idx)))
        return res

    res = []  # 用于保存遍历模型topo结构后得到的节点列表
    node['id'] = node_id
    if traverse_type == 'torch':
        node['golden_path'] = os.path.join(os.path.abspath(path), node['name'])
        res.append(node)
        if len(node['children']) > 0:
            res.extend(enumerate_children(node['children'], path, traverse_type, node_id))
    else:
        node['my_path'] = os.path.join(os.path.abspath(path), '_*/'.join(node_id.split('_')) + '_*', 'after')
        res.append(node)
        if 'nodes' in node.keys() and len(node['nodes']) > 0:
            res.extend(enumerate_children(node['nodes'], path, traverse_type, node_id))
    return res


# 加速库模型间比对相关
def compare_atb_metadata_auto(golden_path, my_path, golden_topo_json_path, my_topo_json_path, output_path="."):
    cur_my_path = os.path.dirname(os.path.abspath(my_path))
    token_id = os.path.basename(cur_my_path).split('_')[1]

    with open(golden_topo_json_path, "r") as file:
        golden_topo = json.load(file)
    with open(my_topo_json_path, "r") as file:
        my_topo = json.load(file)

    gathered_golden_data = traverse_tree(golden_topo, golden_path, 'atb')
    gathered_my_data = traverse_tree(my_topo, my_path, 'atb')
    matched_path_pair = search_mapping_relationships(gathered_golden_data, gathered_my_data)
    gathered_row_data = []
    for data_id, match in enumerate(matched_path_pair):
        data_info = BasicDataInfo(match['golden'], match['my'], token_id, data_id)
        row_data = fill_row_data(data_info, is_broadcast_tensor=True)
        gathered_row_data.append(row_data)
    return save_compare_reault_to_csv(gathered_row_data, output_path)


def search_mapping_relationships(gathered_golden_data, gathered_my_data):
    matches = []
    matched_path_pair = []
    for golden_item, my_item in zip(gathered_golden_data, gathered_my_data):
        if "opType" in golden_item and "opType" in my_item:
            matches.append({'golden': golden_item, 'my': my_item})

    for match in matches:
        try:
            _golden_path = glob.glob(match['golden']['my_path'])[0]
            golden_out_path = get_paths(_golden_path, split_pattern='outtensor')
            _my_path = glob.glob(match['my']['my_path'])[0]
            my_out_path = get_paths(_my_path, split_pattern='outtensor')
            for _golden_tensor_path, _my_tensor_path in zip(golden_out_path, my_out_path):
                matched_path_pair.append({'golden': _golden_tensor_path, 'my': _my_tensor_path})
        except IndexError as e:
            msg = f"Cannot find path! golden: {match['golden']['my_path']}, my: {match['my']['my_path']}"
            logger.debug(msg)

    return matched_path_pair


def get_paths(path_dir, split_pattern):
    out_paths = [x for x in os.listdir(path_dir) if x.startswith('out')]
    out_paths.sort(key=lambda x: int(x.split(split_pattern)[-1].split('.')[0]))
    out_paths = [os.path.join(path_dir, x) for x in out_paths]
    return out_paths


def pair_built_in_op(g_nodes, m_nodes, op_mapping, my_root_node):
    compared_result = []
    for atb_op_type, torch_op_type in op_mapping.items():
        atb_nodes = [m_node for m_node in m_nodes if m_node.op_type == atb_op_type]
        torch_nodes = [g_node for g_node in g_nodes if g_node.op_type == torch_op_type]
        if len(atb_nodes) != len(torch_nodes):
            msg = f"The number of {atb_op_type} node in atb is not equal to {torch_op_type} node in torch"
            logger.debug(msg)
            continue
        for atb_node, torch_node in zip(atb_nodes, torch_nodes):
            if atb_node.op_type == "LinearOperation" and not atb_node.op_param.get("hasBias"):
                next_sibling_node = my_root_node.get_next_sibling_node(atb_node)
                # 当有些算子如ParallelLinearBaseV2，是将w*x+b的操作拆分成两个算子，linear+add，而torch中使用一个算子Linear实现，
                # 因此add node的输出映射的是torch中Linear的输出
                if next_sibling_node and next_sibling_node.op_type == "ElewiseOperation" \
                        and next_sibling_node.op_param.get('elewiseType') == 8:
                    atb_node = next_sibling_node
            my_tensor_path = os.path.join(atb_node.tensor_path, "after", "outtensor0.bin")
            golden_tensor_path = os.path.join(torch_node.tensor_path, "output.pth")
            if os.path.exists(golden_tensor_path) and os.path.exists(my_tensor_path):
                data_info = BasicDataInfo(golden_tensor_path, my_tensor_path, data_id=0)
                row_data = fill_row_data(data_info)
                compared_result.append(row_data)
            else:
                msg = f"golden tensor path: {golden_tensor_path} or my_tensor_path: {my_tensor_path} is not exist."
                logger.debug(msg)
    return compared_result


def pair_custom_op(g_nodes, m_nodes, op_mapping):
    compared_result = []

    op_mapping_flat = []
    for atb_op_type, torch_op_type in op_mapping.items():
        op_mapping_flat.extend([(atb_op_type, x) for x in torch_op_type])

    for atb_op_type, torch_op_type in op_mapping_flat:
        if '_' in atb_op_type:
            atb_op_type, atb_output = atb_op_type.split('_', 1)[0], atb_op_type.split('_', 1)[1]
        else:
            atb_output = "outtensor0"
        if '_' in torch_op_type:
            torch_op_type, torch_output = torch_op_type.split('_', 1)[0], torch_op_type.split('_', 1)[1]
        else:
            torch_output = "outtensor0"
        atb_nodes = [m_node for m_node in m_nodes if atb_op_type in m_node.op_type]
        torch_nodes = [g_node for g_node in g_nodes if torch_op_type in g_node.op_type]
        if len(atb_nodes) != len(torch_nodes):
            msg = f"The number of {atb_op_type} node in atb is not equal to {torch_op_type} node in torch"
            logger.debug(msg)
            continue
        for atb_node, torch_node in zip(atb_nodes, torch_nodes):
            my_tensor_path = os.path.join(atb_node.tensor_path, "after", f"{atb_output}.bin")
            golden_tensor_path = os.path.join(torch_node.tensor_path, f"{torch_output}.pth")
            if os.path.exists(golden_tensor_path) and os.path.exists(my_tensor_path):
                data_info = BasicDataInfo(golden_tensor_path, my_tensor_path, data_id=0)
                row_data = fill_row_data(data_info)
                compared_result.append(row_data)
            else:
                msg = f"golden tensor path: {golden_tensor_path} or my_tensor_path: {my_tensor_path} is not exist."
                logger.debug(msg)

    return compared_result


def cmp_torch_atb_model(data_info, output_path, mapping_dic):
    golden_json = data_info.get("golden_json")
    my_json = data_info.get("my_json")
    torch_tensor_path = data_info.get("torch_tensor_path")
    atb_tensor_path = data_info.get("atb_tensor_path")
    compared_result = []

    golden_root_node = ModelTree.json_to_tree(golden_json, torch_tensor_path)
    golden_layer_type = golden_root_node.get_layer_node_type()
    logger.info("golden_layer_type: %s", golden_layer_type)
    golden_layer_nodes = golden_root_node.get_layer_node(golden_layer_type)

    my_root_node = ModelTree.atb_json_to_tree(my_json, atb_tensor_path)
    my_layer_type = my_root_node.get_layer_node_type()
    logger.info("my_layer_type: %s", my_layer_type)
    my_layer_nodes = my_root_node.get_layer_node(my_layer_type)

    # 原生算子比对
    for golden_layer, my_layer in zip(golden_layer_nodes, my_layer_nodes):
        g_layer_leaf_nodes = golden_layer.get_leaf_nodes()
        m_layer_leaf_nodes = my_layer.get_leaf_nodes()
        compared_result.extend(pair_built_in_op(g_layer_leaf_nodes, m_layer_leaf_nodes,
                                                mapping_dic.get("ATB_TORCH_BUILT_IN_OP_OUTPUT_MAPPING"), my_root_node))

    # 自定义算子比对
    for golden_layer, my_layer in zip(golden_layer_nodes, my_layer_nodes):
        g_layer_all_nodes = golden_layer.get_all_nodes()
        m_layer_all_nodes = my_layer.get_all_nodes()
        compared_result.extend(pair_custom_op(g_layer_all_nodes, m_layer_all_nodes,
                                              mapping_dic.get("ATB_TORCH_CUSTOM_OP_OUTPUT_MAPPING")))

    save_compare_reault_to_csv(compared_result, output_path)


def validate_json(json_obj):
    for key, value in json_obj.items():
        if not re.match(r"^[a-zA-Z0-9_]*$", key):
            return False
        if not isinstance(value, str) and not isinstance(value, list):
            return False
        if isinstance(value, str):
            if not re.match(r"^[a-zA-Z0-9_]*$", value):
                return False
        if isinstance(value, list):
            for v in value:
                if not re.match(r"^[a-zA-Z0-9_]*$", v):
                    return False
    return True


def load_mapping(mapping_file_path): 
    mapping_dic = {
        "ATB_TORCH_BUILT_IN_OP_OUTPUT_MAPPING": ATB_TORCH_BUILT_IN_OP_OUTPUT_MAPPING,
        "ATB_TORCH_CUSTOM_OP_OUTPUT_MAPPING": ATB_TORCH_CUSTOM_OP_OUTPUT_MAPPING,
    }
    mapping_file = os.path.join(mapping_file_path, "op_mapping_file.json")
    if os.path.exists(mapping_file):
        with open(mapping_file, "r") as file:
            file_content = json.load(file)
        if validate_json(file_content):
            for k, v in file_content.items():
                mapping_dic["ATB_TORCH_CUSTOM_OP_OUTPUT_MAPPING"][k] = v
            msg = f"Using user-specified op_mapping from file: {mapping_file}"
            logger.info(msg)
        else:
            msg = f"Invalid op_mapping file: {mapping_file}"
            logger.error(msg)
    else:
        logger.debug("Using built-in op_mapping")
    return mapping_dic


def cmp_torch_atb(torch_model_topo_file, golden_path, my_path, output_path, mapping_file_path):
    try:
        pid = str(my_path.split("/")[-2].split("_")[1])
    except IndexError as e:
        pid = ""
        msg = f"Cannot parse the right pid from my_path! my_path: {my_path}"
        logger.error(msg)

    atb_model_topo_file_path = os.path.join(my_path, "../../..", "model", pid)
    if os.path.exists(atb_model_topo_file_path):
        atb_model_topo_name = os.listdir(atb_model_topo_file_path)[0]
        atb_model_topo_file = os.path.join(atb_model_topo_file_path, atb_model_topo_name)
        if os.path.exists(atb_model_topo_file):
            mapping_dic = load_mapping(mapping_file_path)
            data_info = {
                "golden_json": torch_model_topo_file,
                "my_json": atb_model_topo_file,
                "torch_tensor_path": golden_path,
                "atb_tensor_path": my_path,
            }
            cmp_torch_atb_model(data_info, output_path, mapping_dic)
        else:
            msg = f"Cannot find atb model file: {atb_model_topo_file}"
            logger.error(msg)
    else:
        msg = f"Cannot find atb model file path: {atb_model_topo_file_path}"
        logger.error(msg)
