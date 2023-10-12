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

import json
import pandas as pd
import numpy as np

from msquickcmp.pta_acl_cmp.cmp_algorithm import cmp_alg_map
from msquickcmp.common.utils import logger
from msquickcmp.pta_acl_cmp.constant import TOKEN_ID, DATA_ID, ACL_DATA_PATH, CMP_FLAG, \
    CMP_FAIL_REASON, ACL_DTYPE, ACL_SHAPE, ACL_MAX_VALUE, ACL_MIN_VALUE, ACL_MEAN_VALUE, ATTR_END, \
    ATTR_OBJECT_LENGTH, ATTR_OBJECT_PREFIX, GOLDEN_DATA_PATH, GOLDEN_DTYPE, GOLDEN_SHAPE, \
    GOLDEN_MAX_VALUE, GOLDEN_MIN_VALUE, GOLDEN_MEAN_VALUE, CSV_GOLDEN_HEADER


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
            dtype = np.bool8
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
            for i in range(len(file_data)):
                if file_data[i] == ord("\n"):
                    line = file_data[begin_offset: i].decode("utf-8")
                    begin_offset = i + 1
                    fields = line.split("=")
                    attr_name = fields[0]
                    attr_value = fields[1]
                    if attr_name == ATTR_END:
                        self.obj_buffer = file_data[i + 1:]
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
            for i in range(len(self.dims)):
                self.dims[i] = int(self.dims[i])


def read_acl_transformer_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("{} is not exists".format(file_path))

    if file_path.endswith(".bin"):
        bin = TensorBinFile(file_path)
        data = bin.get_data()
        return data

    raise ValueError("Tensor file path must be end with .bin.")


def compare_tensor(csv_data: pd.DataFrame):
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
        for name, cmp_func in cmp_alg_map.items():
            result = cmp_func(golden_data_fp32, acl_data_fp32)
            csv_data[name][idx] = result
            csv_data[CMP_FLAG][idx] = True

    return csv_data


def auto_compare_metadata(golden_meta, acl_meta):
    # 用于自动映射关系的比对
    data_frame = pd.DataFrame(columns=CSV_GOLDEN_HEADER, index=[0])
    for token_id, g_data in golden_meta.items():
        acl_data = acl_meta.get(token_id)
        if not acl_data:
            continue
        for w_md5, g_data_path in g_data.items():
            a_data_dir = acl_data.get(w_md5)
            if not a_data_dir:
                logger.warning(f"weight md5: {w_md5}, data_path is none.")
                continue
            acl_data_dir = os.path.join(a_data_dir[0], "outtensor0.bin")

            row_data = pd.DataFrame({
                TOKEN_ID: [str(token_id)],
                DATA_ID: [w_md5],
                GOLDEN_DATA_PATH: [g_data_path[0]],
                ACL_DATA_PATH: [acl_data_dir],
                CMP_FLAG: [False]
            })

            data_frame = pd.concat([data_frame, row_data], ignore_index=True)
    return data_frame


def manual_compare_metadata(golden_meta, acl_meta):
    # 用于用户指定data_id的比对
    data_frame = pd.DataFrame(columns=CSV_GOLDEN_HEADER, index=[0])
    for data_id, golden_info in golden_meta.items():
        acl_info = acl_meta.get(data_id)
        if not acl_info:
            continue
        for token_id, golden_data_path in golden_info.items():
            acl_data_path = acl_info.get(token_id)
            if not acl_data_path:
                logger.warning(f"acl data path is none.")
                continue

            row_data = pd.DataFrame({
                TOKEN_ID: [str(token_id)],
                DATA_ID: [data_id],
                GOLDEN_DATA_PATH: [golden_data_path],
                ACL_DATA_PATH: [acl_data_path],
                CMP_FLAG: [False]
            })

            data_frame = pd.concat([data_frame, row_data], ignore_index=True)
    return data_frame


def compare_metadata(golden_path, acl_path, output_path="./"):
    if golden_path.endswith(".json"):
        golden_meta_path = golden_path
    else:
       golden_meta_path = os.path.join(golden_path, "metadata.json")

    with open(golden_meta_path, 'r') as file:
        golden_meta = json.load(file)

    if acl_path.endswith(".json"):
        with open(acl_path, 'r') as file:
            acl_meta = json.load(file)
        data_frame = manual_compare_metadata(golden_meta, acl_meta)
    else:
        from msquickcmp.pta_acl_cmp import acl_metadata

        acl_meta = acl_metadata.init_acl_metadata_by_dump_data(acl_path)
        data_frame = auto_compare_metadata(golden_meta, acl_meta)

    cmp_data_frame = compare_tensor(data_frame)
    cmp_data_frame.dropna(axis=0, how="all", inplace=True)
    cmp_data_frame.to_csv(os.path.join(output_path, "cmp_report.csv"), index=False)


def _get_data_path(data, idx, data_src):
    if data_src == "acl":
        path_key = ACL_DATA_PATH
    else:
        path_key = GOLDEN_DATA_PATH

    data_path = data[path_key][idx]
    return data_path


def write_json_file(data_id, data_path, json_path, token_id):
    # 建议与json解耦，需要的时候用
    import json
    try:
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        json_data = {}
    json_data[data_id] = {token_id: data_path}
    with open(json_path, "w") as f:
        json.dump(json_data, f)