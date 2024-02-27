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
import glob
import numpy as np
import pandas as pd
import json
import torch
import queue
from tqdm import tqdm

from llm.common.log import logger
from llm.common.tool import read_atb_data
from llm.compare.cmp_algorithm import CMP_ALG_MAP
from llm.common.constant import (
    TOKEN_ID,
    DATA_ID,
    MY_DATA_PATH,
    CMP_FAIL_REASON,
    MY_DTYPE,
    MY_SHAPE,
    MY_MAX_VALUE,
    MY_MIN_VALUE,
    MY_MEAN_VALUE,
    GOLDEN_DATA_PATH,
    GOLDEN_DTYPE,
    GOLDEN_SHAPE,
    GOLDEN_MAX_VALUE,
    GOLDEN_MIN_VALUE,
    GOLDEN_MEAN_VALUE,
    CSV_GOLDEN_HEADER,
)
from llm.compare import torchair_utils


NCHW_DIMS = 4
NC1HWC0_DIMS = 5

def acc_compare(golden_path, my_path, output_path="."):
    torchair_ge_graph_path = torchair_utils.get_torchair_ge_graph_path(my_path)
    if torchair_ge_graph_path is not None:
        compare_torchair(golden_path, my_path, torchair_ge_graph_path, output_path=output_path)
    elif os.path.isdir(golden_path):
        golden_tensor_path = os.path.join(golden_path, "golden_tensor")
        golden_topo_flag, golden_topo_json_path = if_dumped_model_topo(golden_path)
        my_topo_flag, my_topo_json_path = if_dumped_model_topo(my_path)
        model_tree_path = os.path.join(os.path.dirname(os.path.abspath(golden_path)), "model_tree.json")
        if os.path.isdir(golden_tensor_path):
            # 存在golden_tensor路径，走手动映射比对逻辑
            logger.info("Manual mapping comparing starts! Comparing manual dump tensors and ATB tensors...")
            compare_metadata(golden_tensor_path, output_path)
        elif os.path.exists(model_tree_path):
            # 存在model_tree_path路径，走torch模型和加速库模型比对逻辑
            logger.info("Automatic mapping comparison starts! Comparing torch tensors and ATB tensors...")
            compare_metadata_auto(golden_path, my_path, model_tree_path, output_path)
        elif golden_topo_flag and my_topo_flag:
            # 存在模型的拓扑信息，走加速库模型间的比对逻辑
            if compare_topo_json(golden_topo_json_path, my_topo_json_path):
                # topo信息一致，走dtype和bs比对逻辑：
                logger.info("Automatic mapping comparison starts! Comparing ATB tensors, the topos of tensors are same...")
                compare_atb_metadata_auto(golden_path, my_path, golden_topo_json_path, my_topo_json_path, output_path)
            else:
                # topo信息不一致，走量化比对逻辑，待补充
                logger.info('Automatic mapping comparison starts! Comparing ATB tensors, the topos of tensors are different...')

    elif os.path.isfile(golden_path) and os.path.isfile(my_path):
        res = compare_file(golden_path, my_path)
        logger.info("Compared results: %s", res)
    else:
        logger.error("The golden_path and my_path must both be directory or file.")
        exit(1)


def read_data(data_path):
    if data_path.endswith(".npy"):
        data = torch.as_tensor(np.load(data_path))
    elif data_path.endswith(".bin"):
        data = read_atb_data(data_path)
    elif data_path.endswith(".pth") or data_path.endswith(".pt"):
        data = torch.load(data_path, map_location=torch.device("cpu"))
    else:
        logger.error("Unsupported data format %s", data_path)
        raise TypeError("Unsupported data format.")
    return data.cpu()


def compare_file(golden_path, my_path):
    golden_data = read_data(golden_path)
    my_data = read_data(my_path)
    return compare_data(golden_data, my_data)


def compare_data(golden_data, my_data):
    golden_data_fp32 = golden_data.reshape(-1).float()
    my_data_fp32 = my_data.reshape(-1).float()
    return compare_tensor(golden_data_fp32, my_data_fp32)


def check_tensor(golden_data_fp32, my_data_fp32):
    tensor_pass = True
    fail_reasons = []

    # 检验golden tensor和my tensor的shape是否一致
    if len(golden_data_fp32) != len(my_data_fp32):
        fail_reasons.append("data shape doesn't match.")
        tensor_pass = False
    # 检验golden_data中是否存在NAN或者inf
    if not torch.all(torch.isfinite(golden_data_fp32)):
        fail_reasons.append("golden_data includes NAN or inf.")
        tensor_pass = False
    # 检验my_data中是否存在NAN或者inf
    if not torch.all(torch.isfinite(my_data_fp32)):
        fail_reasons.append("my_data includes NAN or inf.")
        tensor_pass = False
    return tensor_pass, " ".join(fail_reasons)


def compare_tensor(golden_data_fp32, my_data_fp32):
    row_data, fail_messages = {}, []

    # 检查tensor的shape是否一致、是否存在NAN或inf
    tensor_pass, message = check_tensor(golden_data_fp32, my_data_fp32)
    if not tensor_pass:
        logger.warning(f"check_tensor failed: {message}")
        row_data[CMP_FAIL_REASON] = message
        return row_data

    for name, cmp_func in CMP_ALG_MAP.items():
        result, message = cmp_func(golden_data_fp32, my_data_fp32)
        row_data[name] = result
        if len(message) > 0:
            fail_messages.append(message)
    row_data[CMP_FAIL_REASON] = " ".join(fail_messages)
    return row_data


# 手动映射比对能力
def compare_metadata(golden_path, output_path="."):
    golden_meta_path = os.path.join(golden_path, "metadata.json")
    with open(golden_meta_path, "r") as file:
        golden_meta = json.load(file)
    data_frame = fill_in_data(golden_meta)
    return save_compare_dataframe_to_csv(data_frame, output_path)


def save_compare_dataframe_to_csv(data_frame, output_path="."):
    cur_pid = str(os.getpid())
    csv_data_path = os.path.join(output_path, cur_pid)
    if not os.path.exists(csv_data_path):
        os.makedirs(csv_data_path)

    csv_save_path = os.path.join(csv_data_path, "cmp_report.csv")
    data_frame.fillna(value="", inplace=True)
    data_frame.dropna(axis=0, how="all", inplace=True)
    data_frame.to_csv(csv_save_path, index=False)
    logger.info(f"Saved comparing results: {csv_save_path}")
    return csv_save_path


# 自动映射比对能力
def enumerate_children(children, path, traverse_type='torch', node_id=''):
    res = []
    for idx, children_node in enumerate(children):
        if node_id != '':
            res.extend(traverse_tree(children_node, path, traverse_type, node_id + f'_{idx}'))
        else:
            res.extend(traverse_tree(children_node, path, traverse_type, str(idx)))
    return res


def traverse_tree(node, path, traverse_type='torch', node_id=''):
    res = []
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


def match_first_layer(gathered_golden_data, gathered_my_data, golden_first_layer, my_first_layer):
    matched_layer = []
    j = 0
    for x in gathered_golden_data:
        golden_type = x['type']
        if golden_type == golden_first_layer:
            while j < len(gathered_my_data):
                if 'opType' in gathered_my_data[j].keys() and gathered_my_data[j]['opType'] == my_first_layer:
                    matched_layer.append({'golden': x, 'my': gathered_my_data[j]})
                    j += 1
                    break
                else:
                    j += 1
    return matched_layer


def match_layers(gathered_golden_data, gathered_my_data, golden_hierarchy, my_hierarchy):
    matched_layers = []
    golden_layers = golden_hierarchy.split('/')
    my_layers = my_hierarchy.split('/')
    matched_first_layers = match_first_layer(gathered_golden_data, gathered_my_data, golden_layers[0], my_layers[0])
    if len(golden_layers) > 1 and len(my_layers) > 1:
        for first_layer in matched_first_layers:
            if 'children' in first_layer['golden'].keys() and 'nodes' in first_layer['my'].keys():
                matched_layers.extend(match_layers(first_layer['golden']['children'], first_layer['my']['nodes'], '/'.join(golden_layers[1:]), '/'.join(my_layers[1:])))
    else:
        matched_layers.extend(matched_first_layers)
    return matched_layers


def match_pair(matched_layer):
    golden = matched_layer['golden']
    my = matched_layer['my']
    print(golden)
    print(my)
    matched_path_pair = []    
    return matched_path_pair


def compare_metadata_auto(golden_path, my_path, model_tree_path, output_path="."):
    # 读取torch侧模型文件
    with open(model_tree_path, "r") as file:
        golden_meta = json.load(file)
    
    # 读取atb侧模型文件
    my_meta_flag, my_meta_path = if_dumped_model_topo(my_path)
    if my_meta_flag:
        with open(my_meta_path, "r") as file:
            my_meta = json.load(file)
    else:
        msg = f"Cannot find ATB model! model path: {my_meta_path}"
        logger.error(msg)
        return
    
    # 解析模型文件
    gathered_golden_data = []
    gathered_golden_data.extend(traverse_tree(golden_meta, golden_path, 'torch'))
    gathered_my_data = []
    gathered_my_data.extend(traverse_tree(my_meta, my_path, 'atb'))
    
    # 读取自定义算子映射文件
    op_mapping_dic = {
        'BloomBlock/BloomMLP':'Bloom7bCommonLayer/MlpGateLayerV2',
        'BloomMLP':'MlpGateLayerV2',
    }
    
    # 获取对比路径对
    matched_path_pair = []
    for golden_hierarchy, my_hierarchy in op_mapping_dic.items():
        matched_layers = match_layers(gathered_golden_data, gathered_my_data, golden_hierarchy, my_hierarchy)
        for matched_layer in matched_layers:
            matched_path_pair.extend(match_pair(matched_layer))
    
    # 输出csv文件
    token_id = os.path.basename(os.path.dirname(os.path.abspath(my_path))).split('_')[1]
    gathered_row_data = []
    for data_id, match in enumerate(matched_path_pair):
        row_data = fill_row_data(token_id, data_id, match['golden'], match['my'])
        gathered_row_data.append(row_data)
    data_frame = pd.DataFrame(gathered_row_data, columns=CSV_GOLDEN_HEADER)
    return save_compare_dataframe_to_csv(data_frame, output_path)


# torchair 比对相关
def compare_torchair(golden_path, my_path, ge_graph_path, output_path="."):
    logger.info(f"[compare_torchair], golden_path: {golden_path}, my_path: {my_path}, ge_graph_path: {ge_graph_path}")
    metadata = torchair_utils.build_metadata(golden_path, my_path, ge_graph_path)
    data_frame = fill_in_data(metadata)
    return save_compare_dataframe_to_csv(data_frame, output_path)


def fill_in_data(golden_meta):
    gathered_row_data = []
    for data_id, golden_info in tqdm(golden_meta.items(), total=len(golden_meta)):
        for token_id, path_list in golden_info.items():

            # 读取映射关系json文件中的tenor路径
            if not isinstance(path_list, (list, tuple)) or len(path_list) < 2:
                logger.warning(f"Invalid data in golden metadata.json, data_id: {data_id}, token_id: {token_id}")
                continue
            golden_data_path = path_list[0]
            my_path = path_list[1]

            if torchair_utils.is_torchair_dump_data(golden_data_path, my_path):
                sub_gathered_row_data = fill_row_data_torchair(token_id, data_id, golden_data_path, my_path)
                gathered_row_data.extend(sub_gathered_row_data)
            else:
                row_data = fill_row_data(token_id, data_id, golden_data_path, my_path)
                gathered_row_data.append(row_data)
    return pd.DataFrame(gathered_row_data, columns=CSV_GOLDEN_HEADER)


# torchair 比对相关
def fill_row_data_torchair(token_id, data_id, golden_data_path, my_path):
    my_inputs, my_outputs = torchair_utils.parse_torchair_bin_dump_data(my_path)
    sub_gathered_row_data = []
    logger.debug(
        f"my_inputs length: {len(my_inputs)}, golden_data_path inputs length:, {len(golden_data_path['inputs'])}"
    )
    logger.debug(
        f"my_outputs length: {len(my_outputs)}, golden_data_path outputs length:, {len(golden_data_path['outputs'])}"
    )

    for cur_id, (golden_input, my_input) in enumerate(zip(golden_data_path["inputs"], my_inputs)):
        sub_my_path = "{},{},{}".format(my_path, "inputs", cur_id)
        row_data = fill_row_data(token_id, data_id, golden_input, sub_my_path, loaded_my_data=my_input)
        sub_gathered_row_data.append(row_data)
    for cur_id, (golden_output, my_output) in enumerate(zip(golden_data_path["outputs"], my_outputs)):
        sub_my_path = "{},{},{}".format(my_path, "outputs", cur_id)
        row_data = fill_row_data(token_id, data_id, golden_output, sub_my_path, loaded_my_data=my_output)
        sub_gathered_row_data.append(row_data)
    return sub_gathered_row_data


def is_converting_nc1hwc0_to_nchw(golden_data, my_data):
    if not (golden_data.dim() == NCHW_DIMS and my_data.dim() == NC1HWC0_DIMS):
        return False

    golden_shape, my_shape = golden_data.shape, my_data.shape
    if not (golden_shape[0] == my_shape[0] and golden_shape[2] == my_shape[2] and golden_shape[3] == my_shape[3]):
        return False
    if np.prod(golden_shape) != np.prod(my_shape):
        return False
    return True


def fill_row_data(token_id, data_id, golden_data_path, my_path, loaded_my_data=None):
    # 创建一条比较数据
    logger.debug(f"[fill_row_data], golden_data_path: {golden_data_path}, my_path: {my_path}")
    row_data = {TOKEN_ID: str(token_id), DATA_ID: data_id, GOLDEN_DATA_PATH: golden_data_path, MY_DATA_PATH: my_path}
    if not os.path.isfile(golden_data_path):
        row_data[CMP_FAIL_REASON] = f"golden_data_path: {golden_data_path} is not a file."
        return row_data
    if loaded_my_data is None and not os.path.isfile(my_path):
        row_data[CMP_FAIL_REASON] = f"my_path: {my_path} is not a file."
        return row_data

    golden_data = read_data(golden_data_path)
    my_data = read_data(my_path) if loaded_my_data is None else torch.from_numpy(loaded_my_data)
    if is_converting_nc1hwc0_to_nchw(golden_data, my_data):
        logger.debug(f"[fill_row_data] NC1HWC0 -> NCHW, my_data: {my_data.shape}, golden_data: {golden_data.shape}")
        my_data.permute([0, 4, 1, 2, 3]).reshape(golden_data.shape)

    # 比较数据
    row_data.update(compare_data(golden_data, my_data))
    row_data.update(set_tensor_basic_info_in_row_data(golden_data, my_data))
    return row_data


def set_tensor_basic_info_in_row_data(golden_data, my_data):
    row_data = {}
    row_data[GOLDEN_DTYPE] = str(golden_data.dtype)
    row_data[GOLDEN_SHAPE] = str(list(golden_data.shape))
    if 0 not in golden_data.shape:
        golden_data = golden_data.float()
        row_data[GOLDEN_MAX_VALUE] = golden_data.max().item()
        row_data[GOLDEN_MIN_VALUE] = golden_data.min().item()
        row_data[GOLDEN_MEAN_VALUE] = golden_data.mean().item()

    row_data[MY_DTYPE] = str(my_data.dtype)
    row_data[MY_SHAPE] = str(list(my_data.shape))
    if 0 not in my_data.shape:
        my_data = my_data.float()
        row_data[MY_MAX_VALUE] = my_data.max().item()
        row_data[MY_MIN_VALUE] = my_data.min().item()
        row_data[MY_MEAN_VALUE] = my_data.mean().item()
    return row_data


# 加速库模型间比对相关
def compare_atb_metadata_auto(golden_path, my_path, golden_topo_json_path, my_topo_json_path, output_path="."):
    cur_my_path = os.path.dirname(os.path.abspath(my_path))
    token_id = os.path.basename(cur_my_path).split('_')[1]

    with open(golden_topo_json_path, "r") as file:
        golden_topo = json.load(file)
    with open(my_topo_json_path, "r") as file:
        my_topo = json.load(file)

    gathered_golden_data = []
    gathered_golden_data.extend(traverse_tree(golden_topo, golden_path, 'atb'))
    gathered_my_data = []
    gathered_my_data.extend(traverse_tree(my_topo, my_path, 'atb'))

    matched_path_pair = search_mapping_relationships(gathered_golden_data, gathered_my_data)
    gathered_row_data = []
    for data_id, match in enumerate(matched_path_pair):
        _golden_tensor_path = match['golden']
        _my_tensor_path = match['my']
        row_data = fill_row_data(token_id, data_id, _golden_tensor_path, _my_tensor_path)
        gathered_row_data.append(row_data)
    data_frame = pd.DataFrame(gathered_row_data, columns=CSV_GOLDEN_HEADER)
    return save_compare_dataframe_to_csv(data_frame, output_path)


def compare_topo_json(golden_topo_json_path, my_topo_json_path):  
    try: 
        with open(golden_topo_json_path, 'r') as file1:  
            data1 = json.load(file1)  
        with open(my_topo_json_path, 'r') as file2:  
            data2 = json.load(file2)  
        if data1 == data2:  
            return True  
        else: 
            return False  
    except (IOError, json.JSONDecodeError):  
        return False


def if_dumped_model_topo(golden_path):
    # 判断用户输入路径的ait_dump目录下是否包括/model路径，即是否包括模型拓扑信息
    absolute_path = os.path.abspath(golden_path)      
    model_dir_path = os.path.join(absolute_path, '../../../', 'model')
    model_dir_path = os.path.normpath(model_dir_path)
    if not os.path.isdir(model_dir_path): 
        return False, "" 
    # 搜索/model目录下的所有文件，查找JSON文件  
    for root, dirs, files in os.walk(model_dir_path):  
        for file in files:
            if file.endswith('.json'):    
                json_file_path = os.path.join(root, file)  
                return True, json_file_path  
    # 如果没有找到json文件，返回False和空字符串         
    return False, ""


def search_mapping_relationships(gathered_golden_data, gathered_my_data):
    matches = []
    matched_path_pair = []  # 初始化匹配路径对的空列表  
  
    # 获取两个列表的最小长度，避免索引越界  
    min_length = min(len(gathered_golden_data), len(gathered_my_data))  
  
    # 遍历两个列表  
    for i in range(min_length):  
        golden_item = gathered_golden_data[i]  
        my_item = gathered_my_data[i]  
  
        # 检查两个元素是否都包含"opType"属性  
        if "opType" in golden_item and "opType" in my_item:  
            # 如果都包含，则将"my_path"属性以对象的形式添加到matched_path_pair列表中  
            matches.append({'golden': golden_item, 'my': my_item})
 
    for match in matches:
        try:
            _golden_path = glob.glob(match['golden']['my_path'])[0]
            golden_out_path = [x for x in os.listdir(_golden_path) if x.startswith('out')]
            golden_out_path.sort(key=lambda x: int(x.split('outtensor')[1].split('.')[0]))
            golden_out_path = [os.path.join(_golden_path, x) for x in golden_out_path]
            _my_path = glob.glob(match['my']['my_path'])[0]
            my_out_path = [x for x in os.listdir(_my_path) if x.startswith('out')]
            my_out_path.sort(key=lambda x: int(x.split('outtensor')[1].split('.')[0]))
            my_out_path = [os.path.join(_my_path, x) for x in my_out_path]
            for _golden_tensor_path, _my_tensor_path in zip(golden_out_path, my_out_path):
                print(_golden_tensor_path, _my_tensor_path)
                res = compare_file(_golden_tensor_path, _my_tensor_path)
                matched_path_pair.append({'golden': _golden_tensor_path, 'my': _my_tensor_path})
        except IndexError as e:
            msg = f"Cannot find path! golden: {match['golden']['my_path']}, my: {match['my']['my_path']}"
            logger.debug(msg)

    return matched_path_pair  
