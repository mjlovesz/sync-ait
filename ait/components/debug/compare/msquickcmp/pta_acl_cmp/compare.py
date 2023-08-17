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
import time

import pandas as pd
import numpy as np
import torch

from msquickcmp.common.utils import logger
from msquickcmp.pta_acl_cmp.cmp_algorithm import cmp_alg_map

ATTR_VERSION = "$Version"
ATTR_END = "$End"
ATTR_OBJECT_LENGTH = "$Object.Length"
ATTR_OBJECT_COUNT = "$Object.Count"
ATTR_OBJECT_PREFIX = "$Object."

PTA = "pta"
ACL = "acl"
DATA_ID = 'data_id'
PTA_DATA_PATH = 'pta_data_path'
ACL_DATA_PATH = 'acl_data_path'
PTA_DTYPE = "pta_dtype"
PTA_SHAPE = "pta_shape"
PTA_STACK = "pta_stack"
ACL_DTYPE = "acl_dtype"
ACL_SHAPE = "acl_shape"
ACL_STACK = "acl_stack"
CMP_FLAG = "cmp_flag"
CMP_FAIL_REASON = "cmp_fail_reason"
CSV_HEADER = [DATA_ID, PTA_DATA_PATH, PTA_DTYPE, PTA_SHAPE, ACL_DATA_PATH, ACL_DTYPE, ACL_SHAPE, CMP_FLAG]
CSV_HEADER.extend(list(cmp_alg_map.keys()))
CSV_HEADER.append(CMP_FAIL_REASON)

token_counts = 0


def set_task_id():
    pid = os.getpid()
    dump_env_name = str(pid) + "_" + "DUMP_PATH"
    if not os.getenv(dump_env_name):
        os.environ[dump_env_name] = str(pid)

    global token_counts
    task_id = str(pid) + "_" + str(token_counts)
    if os.getenv("AIT_CMP_TASK_ID") != task_id:
        os.environ["AIT_CMP_TASK_ID"] = task_id

    logger.info("Acl transformer dump data dir: {}".format(os.getenv(dump_env_name)))

    token_counts += 1


def gen_id():
    return "data_" + str(time.time())


def save_pta_data(csv_data, data_id, data_val, data_path):
    if data_val is None:
        return csv_data

    data_val = data_val.cpu().numpy()
    mapping_data = csv_data[csv_data[DATA_ID] == data_id]
    if mapping_data.empty:
        data_val.tofile(data_path)
        row_data = pd.DataFrame({
            DATA_ID: [data_id],
            PTA_DATA_PATH: [data_path],
            PTA_DTYPE: [str(data_val.dtype)],
            PTA_SHAPE: [str(data_val.shape)],
            CMP_FLAG: [False]
        })
        csv_data = pd.concat([csv_data, row_data], ignore_index=True)
    else:
        index = mapping_data.index.values[0]
        data_val.tofile(data_path)
        csv_data[PTA_DATA_PATH][index] = data_path
        csv_data[PTA_DTYPE][index] = str(data_val.dtype)
        csv_data[PTA_SHAPE][index] = str(data_val.shape)

        # 对应的acl_data存在时，触发比对
        csv_data = compare_tensor(csv_data=csv_data)

    return csv_data


def save_acl_data(csv_data, data_id, data_val, data_path):
    if data_val is None:
        return csv_data

    data_val = data_val.cpu().numpy()
    mapping_data = csv_data[csv_data[DATA_ID] == data_id]
    if mapping_data.empty:
        data_val.tofile(data_path)
        row_data = pd.DataFrame({
            DATA_ID: [data_id],
            ACL_DATA_PATH: [data_path],
            ACL_DTYPE: [str(data_val.dtype)],
            ACL_SHAPE: [str(data_val.shape)],
            CMP_FLAG: [False]
        })
        csv_data = pd.concat([csv_data, row_data], ignore_index=True)
    else:
        index = mapping_data.index.values[0]
        data_val.tofile(data_path)
        csv_data[ACL_DATA_PATH][index] = data_path
        csv_data[ACL_DTYPE][index] = str(data_val.dtype)
        csv_data[ACL_SHAPE][index] = str(data_val.shape)

        # 对应的pta数据存在时，触发比对
        csv_data = compare_tensor(csv_data=csv_data)

    return csv_data


def save_acl_dump_tensor(csv_data, data_id, tensor_path):
    mapping_data = csv_data[csv_data[DATA_ID] == data_id]
    if mapping_data.empty:
        row_data = pd.DataFrame({DATA_ID: [data_id], ACL_DATA_PATH: [tensor_path], CMP_FLAG: [False]})
        csv_data = pd.concat([csv_data, row_data], ignore_index=True)
    else:
        index = mapping_data.index.values[0]
        csv_data[ACL_DATA_PATH][index] = tensor_path

    return csv_data


def set_label(data_src: str, data_id: str, data_val=None, tensor_path=None):
    if data_val is None and tensor_path is None:
        return

    if data_val is not None and not isinstance(data_val, torch.Tensor):
        return

    task_id = os.getenv("AIT_CMP_TASK_ID")
    csv_path = os.path.join(".", task_id + "_cmp_result.csv")

    dump_data_dir = "cmp_dump_data"
    if not os.path.exists(dump_data_dir):
        os.mkdir(dump_data_dir)

    if not os.path.exists(csv_path):
        data = pd.DataFrame(columns=CSV_HEADER, index=[0])
    else:
        data = pd.read_csv(csv_path, header=0)

    if data_src == "pta":
        pta_data_dir = os.path.join(".", dump_data_dir, "pta_tensor")
        if not os.path.exists(pta_data_dir):
            os.makedirs(pta_data_dir)

        pta_data_path = os.path.join(pta_data_dir, data_id + '_tensor.bin')
        data = save_pta_data(csv_data=data, data_id=data_id, data_val=data_val, data_path=pta_data_path)

    elif data_src == "acl":
        acl_data_dir = os.path.join(".", dump_data_dir, "acl_tensor")
        if not os.path.exists(acl_data_dir):
            os.makedirs(acl_data_dir)

        if data_val is not None:
            data_path = os.path.join(acl_data_dir, data_id + '_tensor.bin')
            data = save_acl_data(csv_data=data, data_id=data_id, data_val=data_val, data_path=data_path)
        elif tensor_path:
            pid = os.getpid()
            dump_path = str(pid) + "_DUMP_PATH"
            tensor_path = os.path.join(os.getenv("ACLTRANSFORMER_HOME_PATH"), "tensors",
                                       os.getenv(dump_path), task_id, tensor_path)
            data = save_acl_dump_tensor(csv_data=data, data_id=data_id, tensor_path=tensor_path)

    data.to_csv(csv_path, index=False)


def compare_tensor(csv_data: pd.DataFrame):
    csv_data.fillna(value="", inplace=True)
    data = csv_data[csv_data[CMP_FLAG] == False]
    if data.empty:
        return csv_data

    for idx in data.index:
        pta_data_path, pta_dtype, pta_shape = _get_data_info(data, idx, data_src="pta")
        acl_data_path, acl_dtype, acl_shape = _get_data_info(data, idx, data_src="acl")

        if os.path.exists(pta_data_path):
            pta_data = np.fromfile(pta_data_path, pta_dtype).reshape(pta_shape)
        else:
            csv_data[CMP_FAIL_REASON][idx] = "pta_data_path is not exist."
            csv_data[CMP_FLAG][idx] = True
            continue

        if os.path.exists(acl_data_path):
            if acl_dtype and acl_shape:
                acl_data = np.fromfile(acl_data_path, acl_dtype).reshape(acl_shape)
            else:
                acl_data = read_acl_transformer_data(acl_data_path)
        else:
            csv_data[CMP_FAIL_REASON][idx] = "acl_data_path is not exist."
            csv_data[CMP_FLAG][idx] = True
            continue

        for name, cmp_func in cmp_alg_map.items():
            result = cmp_func(pta_data, acl_data)
            csv_data[name][idx] = result
            csv_data[CMP_FLAG][idx] = True

    return csv_data


def _get_data_info(data, idx, data_src):
    if data_src == "pta":
        path_key = PTA_DATA_PATH
        dtype_key = PTA_DTYPE
        shape_key = PTA_SHAPE
    else:
        path_key = ACL_DATA_PATH
        dtype_key = ACL_DTYPE
        shape_key = ACL_SHAPE

    data_path = data[path_key][idx]
    dtype = data[dtype_key][idx]
    shape = data[shape_key][idx]
    if isinstance(shape, str) and shape:
        shape = [int(s) for s in shape[1:-1].split(',')]

    if isinstance(dtype, str) and dtype:
        dtype = np.dtype(dtype)

    return data_path, dtype, shape


class TensorBinFile:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.dtype = 0
        self.format = 0
        self.dims = []

        self.__parse_bin_file()

    def get_tensor(self):
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
            print("error, unsupport dtype:", self.dtype)
            pass
        tensor = torch.tensor(np.frombuffer(self.obj_buffer, dtype=dtype))
        tensor = tensor.view(self.dims)
        return tensor

    def __parse_bin_file(self):
        end_str = f"{ATTR_END}=1"
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
        data = bin.get_tensor()
        return data.cpu().numpy()

    raise ValueError("Tensor file path must be end with .bin.")
