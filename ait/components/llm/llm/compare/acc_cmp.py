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


def acc_compare(golden_path, my_path, output_path):
    if os.path.isdir(golden_path): 
        golden_tensor_path = os.path.join(golden_path, "golden_tensor")
        if os.path.isdir(golden_tensor_path):
            compare_metadata(golden_tensor_path, output_path)
        else:
            logger.error("Can not find 'golden_tensor'.")
            exit(1)
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
    curPid = str(os.getpid())
    golden_meta_path = os.path.join(golden_path, "metadata.json")

    with open(golden_meta_path, 'r') as file:
        golden_meta = json.load(file)
        data_frame = fill_in_data(golden_meta)

    data_frame.dropna(axis=0, how="all", inplace=True)
    csv_data_path = os.path.join(output_path, curPid)
    if not os.path.exists(csv_data_path):
        os.makedirs(csv_data_path)
    data_frame.to_csv(os.path.join(csv_data_path, "cmp_report.csv"), index=False)


def fill_in_data(golden_meta):
    # 创建data_frame
    data_frame = pd.DataFrame(columns=CSV_GOLDEN_HEADER, index=[0])

    for data_id, golden_info in golden_meta.items():
        for token_id, path_list in golden_info.items():

            # 读取映射关系json文件中的tenor路径
            golden_data_path = path_list[0]
            my_path = path_list[1]

            # 创建一条比较数据
            row_data = create_row_data(data_id, token_id, golden_data_path, my_path)

            # 检验my_path和golden data path是否存在并读取数据
            path_is_exist, golden_data, my_data = check_data_path(golden_data_path, my_path, row_data)
            if not path_is_exist:
                data_frame = pd.concat([data_frame, row_data], ignore_index=True)
                continue

            # 转换数据格式：
            golden_data_fp32 = golden_data.reshape(-1).astype("float32")
            my_data_fp32 = my_data.reshape(-1).astype("float32")

            # 检查tensor的shape是否一致、是否存在NAN或inf
            tensor_pass = check_tensor(row_data, golden_data_fp32, my_data_fp32, golden_data, my_data)
            if not tensor_pass:
                data_frame = pd.concat([data_frame, row_data], ignore_index=True)
                continue
            
            # 填充数据
            fill_row_data(row_data, golden_data_fp32, my_data_fp32, golden_data, my_data)

            # 比较数据
            compare_tensor(row_data, golden_data_fp32, my_data_fp32)

            # 将数据写入data_frame的下一行
            data_frame = pd.concat([data_frame, row_data], ignore_index=True)

    return data_frame


def create_row_data(data_id, token_id, golden_data_path, my_path):
    row_data = pd.DataFrame(
        {
            TOKEN_ID: [str(token_id)],
            DATA_ID: [data_id],
            GOLDEN_DATA_PATH: [golden_data_path],
            MY_DATA_PATH: [my_path],
        }
    )
    row_data.fillna(value="", inplace=True)

    return row_data


def check_data_path(golden_data_path, my_path, row_data):
    path_is_exist = True
    if os.path.exists(golden_data_path):
        golden_data = np.load(golden_data_path)
    else:
        logger.warning(f"golden data path is not exists.")
        row_data[CMP_FAIL_REASON] = "golden_data_path is not exist."
        golden_data = 0
        path_is_exist = False
    if os.path.exists(my_path):
        if my_path.endswith(".npy"):
            my_data = np.load(my_path)
        else:
            my_data = read_atb_data(my_path)
    else:
        logger.warning(f"my data path is not exists.")
        row_data[CMP_FAIL_REASON] = "my_path is not exist."
        my_data = 0
        path_is_exist = False

    return path_is_exist, golden_data, my_data


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


def fill_row_data(row_data, golden_data_fp32, my_data_fp32, golden_data, my_data):
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
        if result == 'NaN':
            row_data[CMP_FAIL_REASON] = message
            row_data[name] = result
        else:
            row_data[name] = result
