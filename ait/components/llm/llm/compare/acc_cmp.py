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


def acc_compare(golden_path, my_path, output_path):
    torchair_ge_dump_path = torchair_utils.get_torchair_ge_dump_path(my_path)
    if torchair_ge_dump_path is not None:
        compare_torchair(golden_path, my_path, torchair_ge_dump_path, output_path=output_path)
    elif os.path.isdir(golden_path): 
        golden_tensor_path = os.path.join(golden_path, "golden_tensor")
        if os.path.isdir(golden_tensor_path):
            compare_metadata(golden_tensor_path, output_path)
        else:
            logger.error("Can not find 'golden_tensor'.")
    elif os.path.isfile(golden_path) and os.path.isfile(my_path):
        res = compare_file(golden_path, my_path)
        logger.info("Compared results: %s", res)
    else:
        logger.error("The golden_path and my_path must both be directory or file.")
        exit(1)


def read_data(data_path):
    if data_path.endswith(".npy"):
        data = np.load(data_path)
    elif data_path.endswith(".bin"):
        data = read_atb_data(data_path)
    else:
        logger.error("Unsupported data format %s", data_path)
        raise TypeError("Unsupported data format.")
    return data


def compare_file(golden_path, my_path):
    golden_data = read_data(golden_path)
    my_data = read_data(my_path)
    return compare_data(golden_data, my_data)


def compare_data(golden_data, my_data):
    golden_data_fp32 = golden_data.reshape(-1).astype("float32")
    my_data_fp32 = my_data.reshape(-1).astype("float32")

    res_err = {}
    for name, cmp_func in CMP_ALG_MAP.items():
        result = cmp_func(golden_data_fp32, my_data_fp32)
        res_err.setdefault(name, result)
    return res_err


# 手动映射比对能力
def compare_metadata(golden_path, output_path="./"):
    golden_meta_path = os.path.join(golden_path, "metadata.json")
    with open(golden_meta_path, 'r') as file:
        golden_meta = json.load(file)
    data_frame = fill_in_data(golden_meta)
    save_compare_dataframe_to_csv(data_frame, output_path)


def save_compare_dataframe_to_csv(data_frame, output_path="./"):
    cur_pid = str(os.getpid())
    csv_data_path = os.path.join(output_path, cur_pid)
    if not os.path.exists(csv_data_path):
        os.makedirs(csv_data_path)

    data_frame.fillna(value="", inplace=True)
    data_frame.dropna(axis=0, how="all", inplace=True)
    data_frame.to_csv(os.path.join(csv_data_path, "cmp_report.csv"), index=False)


# torchair 比对相关
def compare_torchair(golden_path, my_path, ge_graph_path, output_path="./"):
    torchair_utils.set_msaccucmp_path_from_cann()
    graph_map = torchair_utils.parse_pbtxt_to_dict(ge_graph_path)
    ge_dump_data = torchair_utils.init_ge_dump_data_from_bin_path(my_path)
    fx_dump_data = torchair_utils.init_fx_dump_data_from_path(golden_path)
    metadata = torchair_utils.build_metadata(graph_map, ge_dump_data, fx_dump_data)

    data_frame = fill_in_data(metadata)
    save_compare_dataframe_to_csv(data_frame, output_path)


def fill_in_data(golden_meta):
    gathered_row_data = []
    for data_id, golden_info in golden_meta.items():
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
    return pd.DataFrame(gathered_row_data)


# torchair 比对相关
def fill_row_data_torchair(token_id, data_id, golden_data_path, my_path):
    my_inputs, my_ouytputs = torchair_utils.parse_torchair_bin_dump_data(my_path)
    sub_gathered_row_data = []
    print(">>>> len(my_inputs):", len(my_inputs), "len(golden_data_path['inputs']):", len(golden_data_path['inputs']))
    print(">>>> len(my_ouytputs):", len(my_ouytputs), "len(golden_data_path['outputs']):", len(golden_data_path['outputs']))

    for golden_input, my_input in zip(golden_data_path["inputs"], my_inputs):
        sub_gathered_row_data.append(fill_row_data(token_id, data_id, golden_input, my_path, loaded_my_data=my_input))
    for golden_output, my_output in zip(golden_data_path["outputs"], my_ouytputs):
        sub_gathered_row_data.append(fill_row_data(token_id, data_id, golden_output, my_path, loaded_my_data=my_output))
    return sub_gathered_row_data


def fill_row_data(token_id, data_id, golden_data_path, my_path, loaded_my_data=None):
    # 创建一条比较数据
    row_data = {TOKEN_ID: str(token_id), DATA_ID: data_id, GOLDEN_DATA_PATH: golden_data_path, MY_DATA_PATH: my_path}
    if not os.path.exists(golden_data_path):
        row_data[CMP_FAIL_REASON] = "golden_data_path is not exist."
        return row_data
    if not os.path.exists(my_path):
        row_data[CMP_FAIL_REASON] = "my_path is not exist."
        return row_data
    
    golden_data = np.load(golden_data_path)
    if loaded_my_data is not None:
        my_data = loaded_my_data
    elif my_path.endswith(".npy"):
        my_data = np.load(my_path)
    else:
        my_data = read_atb_data(my_path)

    # 转换数据格式：
    golden_data_fp32 = golden_data.reshape(-1).astype("float32")
    my_data_fp32 = my_data.reshape(-1).astype("float32")
    
    # 检查tensor的shape是否一致、是否存在NAN或inf
    tensor_pass = check_tensor(row_data, golden_data_fp32, my_data_fp32, golden_data, my_data)
    if not tensor_pass:
        return row_data

    # 填充数据
    set_tensor_basic_info_in_row_data(row_data, golden_data_fp32, my_data_fp32, golden_data, my_data)

    # 比较数据
    compare_tensor(row_data, golden_data_fp32, my_data_fp32)
    return row_data


def check_tensor(row_data, golden_data_fp32, my_data_fp32, golden_data, my_data):
    tensor_pass = True
    fail_reason = ''

    # 检验golden tensor和my tensor的shape是否一致
    if len(golden_data_fp32) != len(my_data_fp32):
        logger.warning(f"data shape doesn't match.")
        fail_reason = f"{fail_reason} data shape doesn't match."
        tensor_pass = False
    # 检验golden_data中是否存在NAN或者inf
    if not np.alltrue(np.isfinite(golden_data)):
        logger.warning(f"golden_data include NAN or inf.")
        fail_reason = f"{fail_reason} golden_data include NAN or inf."
        tensor_pass = False
    # 检验my_data中是否存在NAN或者inf
    if not np.alltrue(np.isfinite(my_data)):
        logger.warning(f"my_data include NAN or inf.")
        fail_reason = f"{fail_reason} my_data include NAN or inf."
        tensor_pass = False
    row_data[CMP_FAIL_REASON] = fail_reason

    return tensor_pass


def set_tensor_basic_info_in_row_data(row_data, golden_data_fp32, my_data_fp32, golden_data, my_data):
    row_data[GOLDEN_DTYPE] = str(golden_data.dtype)
    row_data[GOLDEN_SHAPE] = str(golden_data.shape)
    row_data[GOLDEN_MAX_VALUE] = np.max(golden_data_fp32)
    row_data[GOLDEN_MIN_VALUE] = np.min(golden_data_fp32)
    row_data[GOLDEN_MEAN_VALUE] = np.mean(golden_data_fp32)
    row_data[MY_DTYPE] = str(my_data.dtype)
    row_data[MY_SHAPE] = str(my_data.shape)
    row_data[MY_MAX_VALUE] = np.max(my_data_fp32)
    row_data[MY_MIN_VALUE] = np.min(my_data_fp32)
    row_data[MY_MEAN_VALUE] = np.mean(my_data_fp32)


def compare_tensor(row_data, golden_data_fp32, my_data_fp32):
    for name, cmp_func in CMP_ALG_MAP.items():
        result, message = cmp_func(golden_data_fp32, my_data_fp32)
        row_data[CMP_FAIL_REASON] = message
        row_data[name] = result
