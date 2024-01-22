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
from llm.compare.constant import (
    TOKEN_ID,
    DATA_ID,
    ACL_DATA_PATH,
    CMP_FLAG,
    CMP_FAIL_REASON,
    ACL_DTYPE,
    ACL_SHAPE,
    ACL_MAX_VALUE,
    ACL_MIN_VALUE,
    ACL_MEAN_VALUE,
    ATTR_END,
    ATTR_OBJECT_LENGTH,
    ATTR_OBJECT_PREFIX,
    GOLDEN_DATA_PATH,
    GOLDEN_DTYPE,
    GOLDEN_SHAPE,
    GOLDEN_MAX_VALUE,
    GOLDEN_MIN_VALUE,
    GOLDEN_MEAN_VALUE,
    CSV_GOLDEN_HEADER,
)

class TensorBinFile:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.dtype = 0
        self.format = 0
        self.dims = []

        self.__parse_bin_file()

    def get_data(self):
        if self.dtype == 0:
            dtype = np.float32
        elif self.dtype == 1:
            dtype = np.float16
        elif self.dtype == 2:  # int8
            dtype = np.int8
        elif self.dtype == 3:  # int32
            dtype = np.int32
        elif self.dtype == 9:  # int64
            dtype = np.int64
        elif self.dtype == 12:
            dtype = np.bool_
        else:
            logger.error("Unsupport dtype:", self.dtype)
            pass
        data = np.frombuffer(self.obj_buffer, dtype=dtype)
        data = data.reshape(self.dims)
        return data

    def __parse_bin_file(self):
        with open(self.file_path, "rb") as fd:
            file_data = fd.read()

            begin_offset = 0
            for i, byte in enumerate(file_data):
                if byte == ord("\n"):
                    line = file_data[begin_offset:i].decode("utf-8")
                    begin_offset = i + 1
                    fields = line.split("=")
                    attr_name = fields[0]
                    attr_value = fields[1]
                    if attr_name == ATTR_END:
                        self.obj_buffer = file_data[i + 1 :]
                        break
                    elif attr_name.startswith("$"):
                        self.__parse_system_atrr(attr_name, attr_value)
                    else:
                        self.__parse_user_attr(attr_name, attr_value)
                        pass

    def __parse_system_atrr(self, attr_name, attr_value):
        if attr_name == ATTR_OBJECT_LENGTH:
            self.obj_len = int(attr_value)
        elif attr_name == ATTR_OBJECT_PREFIX:
            pass

    def __parse_user_attr(self, attr_name, attr_value):
        if attr_name == "dtype":
            self.dtype = int(attr_value)
        elif attr_name == "format":
            self.format = int(attr_value)
        elif attr_name == "dims":
            self.dims = attr_value.split(",")
            self.dims = [int(dim) for dim in self.dims]

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

# 下面是和手动映射比对相关的
def compare_metadata(golden_path, output_path="./", dump_clean=False):
    if golden_path.endswith(".json"):
        golden_meta_path = golden_path
    else:
        golden_meta_path = os.path.join(golden_path, "metadata.json")

    with open(golden_meta_path, 'r') as file:
        golden_meta = json.load(file)
        data_frame = manual_compare_metadata(golden_meta)

    cmp_data_frame = compare_tensor(data_frame, dump_clean)
    cmp_data_frame.dropna(axis=0, how="all", inplace=True)
    cmp_data_frame.to_csv(os.path.join(output_path, "cmp_report.csv"), index=False)

def manual_compare_metadata(golden_meta):
    # 用于用户指定data_id的比对
    data_frame = pd.DataFrame(columns=CSV_GOLDEN_HEADER, index=[0])
    for data_id, golden_info in golden_meta.items():
        for token_id, path_list in golden_info.items():
            golden_data_path = path_list[0]
            acl_data_path = path_list[1]
            if not acl_data_path:
                logger.warning(f"acl data path is none.")
                continue
            if os.path.exists(acl_data_path):
                logger.warning(f"acl data path is not exists.")
                continue

            row_data = pd.DataFrame(
                {
                    TOKEN_ID: [str(token_id)],
                    DATA_ID: [data_id],
                    GOLDEN_DATA_PATH: [golden_data_path],
                    ACL_DATA_PATH: [acl_data_path],
                    CMP_FLAG: [False],
                }
            )

            data_frame = pd.concat([data_frame, row_data], ignore_index=True)
    return data_frame

def compare_tensor(csv_data: pd.DataFrame, dump_clean=False):
    csv_data.fillna(value="", inplace=True)
    data = csv_data[csv_data[CMP_FLAG] == False]
    if data.empty:
        return csv_data

    for idx in data.index:
        golden_data_path = _get_data_path(data, idx, data_src="golden")
        acl_data_path = _get_data_path(data, idx, data_src="acl")

        if os.path.exists(golden_data_path):
            golden_data = np.load(golden_data_path)
        else:
            csv_data[CMP_FAIL_REASON][idx] = "golden_data_path is not exist."
            csv_data[CMP_FLAG][idx] = True
            continue
        if os.path.exists(acl_data_path):
            if acl_data_path.endswith(".npy"):
                acl_data = np.load(acl_data_path)
            else:
                acl_data = read_acl_transformer_data(acl_data_path)
        else:
            csv_data[CMP_FAIL_REASON][idx] = "acl_data_path is not exist."
            csv_data[CMP_FLAG][idx] = True
            continue

        golden_data_fp32 = golden_data.reshape(-1).astype("float32")
        acl_data_fp32 = acl_data.reshape(-1).astype("float32")

        csv_data[GOLDEN_DTYPE][idx] = str(golden_data.dtype)
        csv_data[GOLDEN_SHAPE][idx] = str(golden_data.shape)
        csv_data[GOLDEN_MAX_VALUE][idx] = np.max(golden_data_fp32)
        csv_data[GOLDEN_MIN_VALUE][idx] = np.min(golden_data_fp32)
        csv_data[GOLDEN_MEAN_VALUE][idx] = np.mean(golden_data_fp32)

        csv_data[ACL_DTYPE][idx] = str(acl_data.dtype)
        csv_data[ACL_SHAPE][idx] = str(acl_data.shape)
        csv_data[ACL_MAX_VALUE][idx] = np.max(acl_data_fp32)
        csv_data[ACL_MIN_VALUE][idx] = np.min(acl_data_fp32)
        csv_data[ACL_MEAN_VALUE][idx] = np.mean(acl_data_fp32)

        if len(golden_data_fp32) != len(acl_data_fp32):
            csv_data[CMP_FAIL_REASON][idx] = "data shape doesn't match."
            csv_data[CMP_FLAG][idx] = True
            continue
        for name, cmp_func in CMP_ALG_MAP.items():
            result = cmp_func(golden_data_fp32, acl_data_fp32)
            csv_data[name][idx] = result
            csv_data[CMP_FLAG][idx] = True
        if dump_clean:
            os.remove(acl_data_path)
            os.remove(golden_data_path)
    return csv_data    

def _get_data_path(data, idx, data_src):
    if data_src == "acl":
        path_key = ACL_DATA_PATH
    else:
        path_key = GOLDEN_DATA_PATH

    data_path = data[path_key][idx]
    return data_path

def read_acl_transformer_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("{} is not exists".format(file_path))

    if file_path.endswith(".bin"):
        bin_tensor = TensorBinFile(file_path)
        data = bin_tensor.get_data()
        return data

    raise ValueError("Tensor file path must be end with .bin.")