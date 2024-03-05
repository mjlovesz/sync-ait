import datetime
import os
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import torch

from llm.common.tool import read_atb_data
from llm.common.constant import TOKEN_ID, DATA_ID, GOLDEN_DATA_PATH, MY_DATA_PATH, CMP_FAIL_REASON, GOLDEN_DTYPE, \
    GOLDEN_SHAPE, GOLDEN_MAX_VALUE, GOLDEN_MIN_VALUE, GOLDEN_MEAN_VALUE, MY_DTYPE, MY_SHAPE, MY_MAX_VALUE, MY_MIN_VALUE, \
    MY_MEAN_VALUE, CSV_GOLDEN_HEADER
from llm.common.log import logger
from llm.compare.cmp_algorithm import CMP_ALG_MAP

MIN_LAYER_NUMBER = 10


class BasicDataInfo:
    count_data_id = 0  # Count data_id, increment by 1 every time creating a new instance

    @classmethod
    def _count(cls):
        cls.count_data_id += 1

    def __init__(self, golden_data_path, my_data_path, token_id=0, data_id=None):
        self.token_id, self.my_data_path, self.golden_data_path = token_id, my_data_path, golden_data_path
        self.data_id = self.count_data_id if data_id is None else data_id
        self._count()

    def to_dict(self):
        return {
            TOKEN_ID: str(self.token_id),
            DATA_ID: str(self.data_id),
            GOLDEN_DATA_PATH: self.golden_data_path,
            MY_DATA_PATH: self.my_data_path,
        }


def fill_row_data(data_info: BasicDataInfo, loaded_my_data=None, loaded_golden_data=None, is_broadcast_tensor=False):
    # 第三个参数“is_broadcast_tensor”用于两个模型batch size不一致时将低维的tensor广播到高维进行比较
    # 创建一条比较数据
    golden_data_path, my_data_path = data_info.golden_data_path, data_info.my_data_path
    logger.debug(f"[fill_row_data], golden_data_path: {golden_data_path}, my_data_path: {my_data_path}")
    row_data = data_info.to_dict()
    if loaded_golden_data is None and not os.path.isfile(golden_data_path):
        row_data[CMP_FAIL_REASON] = f"golden_data_path: {golden_data_path} is not a file."
        return row_data
    if loaded_my_data is None and not os.path.isfile(my_data_path):
        row_data[CMP_FAIL_REASON] = f"my_data_path: {my_data_path} is not a file."
        return row_data

    golden_data = read_data(golden_data_path) if loaded_golden_data is None else torch.from_numpy(loaded_golden_data)
    my_data = read_data(my_data_path) if loaded_my_data is None else torch.from_numpy(loaded_my_data)

    if is_broadcast_tensor:
        broadcast_golden_data, broadcast_my_data = torch.broadcast_tensors(golden_data, my_data)
        row_data.update(compare_data(broadcast_golden_data, broadcast_my_data))
    else:
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


def save_compare_reault_to_csv(gathered_row_data, output_path="."):
    cur_pid = str(os.getpid())
    csv_data_path = os.path.join(output_path, cur_pid)
    if not os.path.exists(csv_data_path):
        os.makedirs(csv_data_path)

    cur_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    csv_save_path = os.path.join(output_path, f"ait_cmp_report_{cur_time}.csv")

    data_frame = pd.DataFrame(gathered_row_data, columns=CSV_GOLDEN_HEADER)
    data_frame.fillna(value="", inplace=True)
    data_frame.dropna(axis=0, how="all", inplace=True)
    data_frame.to_csv(csv_save_path, index=False)
    logger.info(f"Saved comparing results: {csv_save_path}")
    return csv_save_path


def compare_data(golden_data, my_data):
    golden_data_fp32 = golden_data.reshape(-1).float()
    my_data_fp32 = my_data.reshape(-1).float()
    return compare_tensor(golden_data_fp32, my_data_fp32)


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
