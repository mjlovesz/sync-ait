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

import numpy as np
import pandas as pd
import json
import torch
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
from llm.compare.cmp_utils import search_layer_node, get_layer_node, get_leaf_nodes, get_all_nodes
from llm.compare.op_mapping import ATB_TORCH_BUILT_IN_OP_MAPPING
from llm.dump.torch_dump.topo import ModelTree

NCHW_DIMS = 4
NC1HWC0_DIMS = 5


def acc_compare(golden_path, my_path, output_path="."):
    torchair_ge_graph_path = torchair_utils.get_torchair_ge_graph_path(my_path)
    if torchair_ge_graph_path is not None:
        compare_torchair(golden_path, my_path, torchair_ge_graph_path, output_path=output_path)
    elif os.path.isdir(golden_path):
        golden_tensor_path = os.path.join(golden_path, "golden_tensor")
        if os.path.isdir(golden_tensor_path):
            compare_metadata(golden_tensor_path, output_path)
        else:
            logger.error("Can not find 'golden_tensor'.")
            torch_model_topo_file = os.path.join(golden_path, "..", "model_tree.json")
            pid = str(my_path.split("/")[-2].split("_")[1])
            atb_model_topo_file_path = os.path.join(my_path, "../../..", "model", pid)
            logger.info("atb model file: %s", atb_model_topo_file_path)
            logger.info("torch_model_topo_file: %s", torch_model_topo_file)
            if os.path.exists(torch_model_topo_file) and os.path.exists(atb_model_topo_file_path):
                logger.info("start to compare atb model with torch model.")
                atb_model_topo_name = os.listdir(atb_model_topo_file_path)[0]
                atb_model_topo_file = os.path.join(atb_model_topo_file_path, atb_model_topo_name)
                logger.info("atb_model_topo_file: %s", atb_model_topo_file)
                if os.path.exists(atb_model_topo_file):
                    cmp_torch_atb_model(torch_model_topo_file, atb_model_topo_file, golden_path,
                                        my_path, output_path)
                else:
                    logger.error("atb model file %s is not exist.", atb_model_topo_file)

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


def cmp_torch_atb_model(golden_json, my_json, torch_tensor_path, atb_tensor_path, output_path):
    compared_result = []
    golden_root_node = ModelTree.json_to_tree(golden_json, torch_tensor_path)
    golden_layer_type = search_layer_node(golden_root_node)
    logger.info("golden_layer_type: %s", golden_layer_type)
    golden_layer_nodes = get_layer_node(golden_root_node, golden_layer_type)

    my_root_node = ModelTree.atb_json_to_tree(my_json, atb_tensor_path)
    my_layer_type = search_layer_node(my_root_node)
    logger.info("my_layer_type: %s", my_layer_type)
    my_layer_nodes = get_layer_node(my_root_node, my_layer_type)
    
    # 原生算子比对
    for golden_layer, my_layer in zip(golden_layer_nodes, my_layer_nodes):
        g_layer_leaf_nodes = get_leaf_nodes(golden_layer)
        m_layer_leaf_nodes = get_leaf_nodes(my_layer)
        for atb_op_type, torch_op_type in ATB_TORCH_BUILT_IN_OP_MAPPING.items():
            atb_nodes = []
            torch_nodes = []
            for m_leaf_node in m_layer_leaf_nodes:
                if m_leaf_node.node_type == atb_op_type:
                    atb_nodes.append(m_leaf_node)
            for g_leaf_node in g_layer_leaf_nodes:
                if g_leaf_node.node_type == torch_op_type:
                    torch_nodes.append(g_leaf_node)
            if len(atb_nodes) != len(torch_nodes):
                logger.warning("The number of %s node in atb is not equal to %s node in torch",
                               atb_op_type, torch_op_type)
                continue
            for atb_node, torch_node in zip(atb_nodes, torch_nodes):
                my_tensor_path = os.path.join(atb_node.tensor_path, "after", "outtensor0.bin")
                golden_tensor_path = os.path.join(torch_node.tensor_path, "output_exec1.pth")
                logger.info("my_tensor_path: %s", my_tensor_path)
                logger.info("golden_tensor_path: %s", golden_tensor_path)
                if os.path.exists(golden_tensor_path) and os.path.exists(my_tensor_path):
                    row_data = fill_row_data(0, 0, golden_tensor_path, my_tensor_path)
                else:
                    logger.debug("golden tensor path: %s or my_tensor_path: %s is not exist.",
                                 golden_tensor_path, my_tensor_path)
                compared_result.append(row_data)
    
    op_mapping = {
        "CommonLayer": ["GLMBlock", "BloomBlock"],
        "MlpGateLayerV2":["BloomMLP", "MLP"],
        "RmsNormOperation":["RMSNorm"],
        "SelfAttentionOperation":["CoreAttention"],
    }

    op_tensor_mapping = {
        "CommonLayer_GLMBlock": [(0, 0)],
        "CommonLayer_BloomBlock": [(0, 0)],
    }

    # 自定义算子比对
    for golden_layer, my_layer in zip(golden_layer_nodes, my_layer_nodes):
        g_layer_all_nodes = get_all_nodes(golden_layer)
        m_layer_all_nodes = get_all_nodes(my_layer)
        for atb_op_type, torch_op_type_list in op_mapping.items():
            for torch_op_type in torch_op_type_list:
                atb_nodes = []
                torch_nodes = []
                for m_node in m_layer_all_nodes:
                    if atb_op_type in m_node.node_type:
                        atb_nodes.append(m_node)
                for g_node in g_layer_all_nodes:
                    if torch_op_type in g_node.node_type:
                        torch_nodes.append(g_node)
                if len(atb_nodes) != len(torch_nodes):
                    logger.warning("The number of %s node in atb is not equal to %s node in torch",
                                atb_op_type, torch_op_type)
                    continue
                for atb_node, torch_node in zip(atb_nodes, torch_nodes):
                    tensor_mapping_key = atb_op_type + '_' + torch_op_type
                    if tensor_mapping_key in op_tensor_mapping.keys():
                        mapping_idx_list = op_tensor_mapping[tensor_mapping_key]
                        for atb_idx, torch_idx in mapping_idx_list:
                            my_tensor_path = os.path.join(atb_node.tensor_path, "after", f"outtensor{atb_idx}.bin")
                            golden_tensor_path = os.path.join(torch_node.tensor_path, f"output_exec1_{torch_idx}.pth")
                            logger.info("my_tensor_path: %s", my_tensor_path)
                            logger.info("golden_tensor_path: %s", golden_tensor_path)
                            if os.path.exists(golden_tensor_path) and os.path.exists(my_tensor_path):
                                row_data = fill_row_data(0, 0, golden_tensor_path, my_tensor_path)
                            else:
                                logger.debug("golden tensor path: %s or my_tensor_path: %s is not exist.",
                                            golden_tensor_path, my_tensor_path)
                            compared_result.append(row_data)                             

                    else:
                        my_tensor_path = os.path.join(atb_node.tensor_path, "after", "outtensor0.bin")
                        golden_tensor_path = os.path.join(torch_node.tensor_path, "output_exec1.pth")
                        logger.info("my_tensor_path: %s", my_tensor_path)
                        logger.info("golden_tensor_path: %s", golden_tensor_path)
                        if os.path.exists(golden_tensor_path) and os.path.exists(my_tensor_path):
                            row_data = fill_row_data(0, 0, golden_tensor_path, my_tensor_path)
                        else:
                            logger.debug("golden tensor path: %s or my_tensor_path: %s is not exist.",
                                        golden_tensor_path, my_tensor_path)
                        compared_result.append(row_data) 


    data_frame = pd.DataFrame(compared_result, columns=CSV_GOLDEN_HEADER)
    save_compare_dataframe_to_csv(data_frame, output_path)
