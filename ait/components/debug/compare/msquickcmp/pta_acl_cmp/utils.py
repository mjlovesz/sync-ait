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
from msquickcmp.pta_acl_cmp.constant import TOKEN_ID, CSV_HEADER, DATA_ID, PTA_DATA_PATH, ACL_DATA_PATH, CMP_FLAG, \
    CMP_FAIL_REASON, PTA_DTYPE, PTA_SHAPE, ACL_DTYPE, ACL_SHAPE, PTA_MAX_VALUE, PTA_MIN_VALUE, PTA_MEAN_VALUE, \
    ACL_MAX_VALUE, ACL_MIN_VALUE, ACL_MEAN_VALUE, ATTR_END, ATTR_OBJECT_LENGTH, ATTR_OBJECT_PREFIX


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
        # pta_data_path, pta_dtype, pta_shape = _get_data_info(data, idx, data_src="pta")
        # acl_data_path, acl_dtype, acl_shape = _get_data_info(data, idx, data_src="acl")
        pta_data_path = _get_data_path(data, idx, data_src="pta")
        acl_data_path = _get_data_path(data, idx, data_src="acl")

        if os.path.exists(pta_data_path):
            # pta_data = np.fromfile(pta_data_path, pta_dtype).reshape(pta_shape)
            pta_data = np.load(pta_data_path)
        else:
            csv_data[CMP_FAIL_REASON][idx] = "pta_data_path is not exist."
            csv_data[CMP_FLAG][idx] = True
            continue

        if os.path.exists(acl_data_path):
            if acl_data_path.endswith(".npy"):
                # acl_data = np.fromfile(acl_data_path, acl_dtype).reshape(acl_shape)
                acl_data = np.load(acl_data_path)
            else:
                acl_data = read_acl_transformer_data(acl_data_path)
        else:
            csv_data[CMP_FAIL_REASON][idx] = "acl_data_path is not exist."
            csv_data[CMP_FLAG][idx] = True
            continue

        csv_data[PTA_DTYPE][idx] = str(pta_data.dtype)
        csv_data[PTA_SHAPE][idx] = str(pta_data.shape)
        csv_data[ACL_DTYPE][idx] = str(acl_data.dtype)
        csv_data[ACL_SHAPE][idx] = str(acl_data.shape)

        pta_data_fp32 = pta_data.reshape(-1).astype("float32")
        acl_data_fp32 = acl_data.reshape(-1).astype("float32")

        csv_data[PTA_MAX_VALUE][idx] = np.max(pta_data_fp32)
        csv_data[PTA_MIN_VALUE][idx] = np.min(pta_data_fp32)
        csv_data[PTA_MEAN_VALUE][idx] = np.mean(pta_data_fp32)

        csv_data[ACL_MAX_VALUE][idx] = np.max(acl_data_fp32)
        csv_data[ACL_MIN_VALUE][idx] = np.min(acl_data_fp32)
        csv_data[ACL_MEAN_VALUE][idx] = np.mean(acl_data_fp32)

        for name, cmp_func in cmp_alg_map.items():
            result = cmp_func(pta_data_fp32, acl_data_fp32)
            csv_data[name][idx] = result
            csv_data[CMP_FLAG][idx] = True

    return csv_data


def compare_metadata(golden_path, acl_path, output_path="./"):
    golden_meta_path = os.path.join(golden_path, "metadata.json")
    with open(golden_meta_path, 'r') as file:
        golden_meta = json.load(file)

    if acl_path.endswith(".json"):
        with open(acl_path, 'r') as file:
            acl_meta = json.load(file)
    else:
        from msquickcmp.pta_acl_cmp import acl_metadata

        acl_meta = acl_metadata.init_acl_metadata_by_dump_data(acl_path)

    data_frame = pd.DataFrame(columns=[TOKEN_ID] + CSV_HEADER, index=[0])

    for token_id, g_data in golden_meta.items():
        acl_data = acl_meta.get(token_id)
        if not acl_data:
            continue
        for w_md5, g_data_path in g_data.items():
            acl_data_dir = acl_data.get(w_md5)
            if not a_data_dir:
                logger.warning(f"weight md5: {w_md5}, data_path is none.")
                continue
            acl_data_dir = os.path.join(a_data_dir[0], "outtensor0.bin")

            row_data = pd.DataFrame({
                TOKEN_ID: [str(token_id)],
                DATA_ID: [w_md5],
                PTA_DATA_PATH: [g_data_path[0]],
                ACL_DATA_PATH: [acl_data_dir],
                CMP_FLAG: [False]
            })

            data_frame = pd.concat([data_frame, row_data], ignore_index=True)

    cmp_data_frame = compare_tensor(data_frame)
    cmp_data_frame.dropna(axis=0, how="all", inplace=True)
    cmp_data_frame.to_csv(os.path.join(output_path, "cmp_report.csv"), index=False)


def _get_data_path(data, idx, data_src):
    if data_src == "pta":
        path_key = PTA_DATA_PATH
    else:
        path_key = ACL_DATA_PATH

    data_path = data[path_key][idx]
    return data_path
