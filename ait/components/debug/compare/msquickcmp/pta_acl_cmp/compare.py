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
from msquickcmp.pta_acl_cmp.constant import ATTR_END, ATTR_OBJECT_LENGTH, ATTR_OBJECT_COUNT, \
    ATTR_OBJECT_PREFIX, PTA, ACL, DATA_ID, PTA_DATA_PATH, ACL_DATA_PATH, PTA_DTYPE, PTA_SHAPE, \
    PTA_MAX_VALUE, PTA_MIN_VALUE, PTA_MEAN_VALUE, PTA_STACK, ACL_DTYPE, ACL_SHAPE, ACL_MAX_VALUE, \
    ACL_MIN_VALUE, ACL_MEAN_VALUE, ACL_STACK, CMP_FLAG, CMP_FAIL_REASON, CSV_HEADER, \
    MODEL_INFER_TASK_ID, AIT_CMP_TASK_DIR, AIT_CMP_TASK, AIT_CMP_TASK_PID, ACL_DATA_MAP_FILE, \
    GOLDEN_DATA_PATH, GOLDEN_DTYPE, GOLDEN_SHAPE, CSV_GOLDEN_HEADER


CSV_HEADER.extend(list(cmp_alg_map.keys()))
CSV_HEADER.append(CMP_FAIL_REASON)
CSV_GOLDEN_HEADER.extend(list(cmp_alg_map.keys()))
CSV_GOLDEN_HEADER.append(CMP_FAIL_REASON)
token_counts = 0


def set_task_id():
    # 通过ait拉起精度比对任务，接口才会生效
    if os.getenv(AIT_CMP_TASK) != "1":
        return

    pid = os.getpid()
    dump_env_name = str(pid) + "_" + "DUMP_PATH"
    if not os.getenv(dump_env_name):
        os.environ[dump_env_name] = str(pid)

    global token_counts
    task_id = str(pid) + "_" + str(token_counts)
    if os.getenv(MODEL_INFER_TASK_ID) != task_id:
        os.environ[MODEL_INFER_TASK_ID] = task_id

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
        # csv_data = compare_tensor(csv_data=csv_data)

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
    # 通过ait拉起精度比对任务，接口才会生效
    if os.getenv(AIT_CMP_TASK) != "1":
        return

    if data_val is None and tensor_path is None:
        return

    if data_val is not None and not isinstance(data_val, torch.Tensor):
        return

    task_id = os.getenv(MODEL_INFER_TASK_ID)
    task_id = task_id or ""
    ait_task_dir = os.getenv(AIT_CMP_TASK_DIR)
    ait_task_dir = ait_task_dir or ""
    ait_cmp_task_pid = os.getenv(AIT_CMP_TASK_PID)
    ait_cmp_task_pid = ait_cmp_task_pid or ""

    csv_result_dir = os.path.join(ait_task_dir, ait_cmp_task_pid)
    csv_path = os.path.join(csv_result_dir, task_id + "_cmp_result.csv")

    pid = os.getpid()
    dump_data_dir = f"{pid}_cmp_dump_data"

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
        elif tensor_path:  # low-level
            write_acl_map_file(tensor_path)
            pid = os.getpid()
            tensor_path = os.path.join(os.getenv("ACLTRANSFORMER_HOME_PATH"), "tensors",
                                       str(pid), task_id, tensor_path)
            data = save_acl_dump_tensor(csv_data=data, data_id=data_id, tensor_path=tensor_path)

    data.to_csv(csv_path, index=False)


def write_acl_map_file(tensor_path):
    ait_cmp_task_pid = os.getenv(AIT_CMP_TASK_PID)
    ait_cmp_task_pid = ait_cmp_task_pid or ""
    acl_map_file_dir = os.path.join('/tmp', ait_cmp_task_pid)
    acl_map_file_path = os.path.join(acl_map_file_dir, ACL_DATA_MAP_FILE)
    if not os.path.exists(acl_map_file_dir):
        os.mkdir(acl_map_file_dir)

    if os.path.exists(acl_map_file_path):
        with open(acl_map_file_path, 'r') as file:
            tensor_paths = file.readlines()

        if tensor_path + "\n" not in tensor_paths:
            with open(acl_map_file_path, mode="a") as file:
                file.write(tensor_path)
                file.write("\n")
    else:
        with open(acl_map_file_path, mode="a") as file:
            file.write(tensor_path)
            file.write("\n")


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
            logger.error("Unsupport dtype:", self.dtype)
            pass
        tensor = torch.tensor(np.frombuffer(self.obj_buffer, dtype=dtype))
        tensor = tensor.view(self.dims)
        return tensor

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
        data = bin.get_tensor()
        return data.cpu().numpy()

    raise ValueError("Tensor file path must be end with .bin.")


def dump_data(data_src, data_id, data_val=None, tensor_path=None, token_id=0):
    if data_val is None and tensor_path is None:
        return

    if data_val is not None and not isinstance(data_val, torch.Tensor):
        return

    # 获取csv路径
    pid = os.getpid()
    csv_result_dir = os.path.join("./", str(pid))
    if not os.path.exists(csv_result_dir):
        os.mkdir(csv_result_dir)

    csv_path = os.path.join(csv_result_dir, f"{pid}_cmp_result.csv")


    dump_data_dir = f"{pid}_cmp_dump_data"
    # 如果没有dump数据文件夹新建一个
    if not os.path.exists(dump_data_dir):
        os.mkdir(dump_data_dir)

    # 如果没有csv新建一个
    if not os.path.exists(csv_path):
        data = pd.DataFrame(columns=CSV_GOLDEN_HEADER, index=[0])
    else:
        data = pd.read_csv(csv_path, header=0)
    if data_src == "golden":
        golden_data_dir = os.path.join(".", dump_data_dir, "golden_tensor", str(token_id))
        if not os.path.exists(golden_data_dir):
            os.makedirs(golden_data_dir)
        if data_val is not None:
            golden_data_path = os.path.join(golden_data_dir, f'{data_id}_tensor.bin')
            data = save_golden_data(csv_data=data, data_id=data_id, data_val=data_val, data_path=golden_data_path)
        elif tensor_path:  # low-level
            token_tensor_path = os.path.join(str(token_id), tensor_path)
            write_acl_map_file(token_tensor_path)
            tensor_path = os.path.join(os.getenv("ACLTRANSFORMER_HOME_PATH"), "tensors",
                                       f"thread_{str(pid)}", str(token_id), tensor_path)
            data = save_golden_dump_tensor(csv_data=data, data_id=data_id, tensor_path=tensor_path)
    elif data_src == "acl":
        acl_data_dir = os.path.join(".", dump_data_dir, "acl_tensor", str(token_id))
        if not os.path.exists(acl_data_dir):
            os.makedirs(acl_data_dir)
        if data_val is not None:
            data_path = os.path.join(acl_data_dir, f'{data_id}_tensor.bin')
            data = save_acl_data(csv_data=data, data_id=data_id, data_val=data_val, data_path=data_path)
        elif tensor_path:  # low-level
            token_tensor_path = os.path.join(str(token_id), tensor_path)
            write_acl_map_file(token_tensor_path)
            tensor_path = os.path.join(os.getenv("ACLTRANSFORMER_HOME_PATH"), "tensors",
                                       f"thread_{str(pid)}", str(token_id), tensor_path)
            data = save_acl_dump_tensor(csv_data=data, data_id=data_id, tensor_path=tensor_path)
    data.to_csv(csv_path, index=False)


def save_golden_data(csv_data, data_id, data_val, data_path):
    if data_val is None:
        return csv_data

    data_val = data_val.cpu().numpy()
    mapping_data = csv_data[csv_data[DATA_ID] == data_id]
    if mapping_data.empty:
        data_val.tofile(data_path)
        row_data = pd.DataFrame({
            DATA_ID: [data_id],
            GOLDEN_DATA_PATH: [data_path],
            GOLDEN_DTYPE: [str(data_val.dtype)],
            GOLDEN_SHAPE: [str(data_val.shape)],
            CMP_FLAG: [False]
        })
        csv_data = pd.concat([csv_data, row_data], ignore_index=True)
    else:
        index = mapping_data.index.values[0]
        data_val.tofile(data_path)
        csv_data[GOLDEN_DATA_PATH][index] = data_path
        csv_data[GOLDEN_DTYPE][index] = str(data_val.dtype)
        csv_data[GOLDEN_SHAPE][index] = str(data_val.shape)

    return csv_data


def save_golden_dump_tensor(csv_data, data_id, tensor_path):
    mapping_data = csv_data[csv_data[DATA_ID] == data_id]
    if mapping_data.empty:
        row_data = pd.DataFrame({DATA_ID: [data_id], GOLDEN_DATA_PATH: [tensor_path], CMP_FLAG: [False]})
        csv_data = pd.concat([csv_data, row_data], ignore_index=True)
    else:
        index = mapping_data.index.values[0]
        csv_data[GOLDEN_DATA_PATH][index] = tensor_path

    return csv_data


def compare_all(csv_data:pd.DataFrame):
    csv_data.fillna(value="", inplace=True)
    data = csv_data[csv_data[CMP_FLAG] == False]
    if data.empty:
        return csv_data

    for idx in data.index:
        golden_data_path, golden_dtype, golden_shape = _get_data_info(data, idx, data_src="golden")
        acl_data_path, acl_dtype, acl_shape = _get_data_info(data, idx, data_src="acl")

        if os.path.exists(golden_data_path):
            if golden_dtype and golden_shape:
                golden_data = np.fromfile(golden_data_path, golden_dtype).reshape(golden_shape)
            else:
                golden_data = read_acl_transformer_data(golden_data_path)
        else:
            csv_data[CMP_FAIL_REASON][idx] = "golden_data_path is not exist."
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

        golden_data_fp32 = golden_data.reshape(-1).astype("float32")
        acl_data_fp32 = acl_data.reshape(-1).astype("float32")

        csv_data[PTA_MAX_VALUE][idx] = np.max(golden_data_fp32)
        csv_data[PTA_MIN_VALUE][idx] = np.min(golden_data_fp32)
        csv_data[PTA_MEAN_VALUE][idx] = np.mean(golden_data_fp32)

        csv_data[ACL_MAX_VALUE][idx] = np.max(acl_data_fp32)
        csv_data[ACL_MIN_VALUE][idx] = np.min(acl_data_fp32)
        csv_data[ACL_MEAN_VALUE][idx] = np.mean(acl_data_fp32)

        for name, cmp_func in cmp_alg_map.items():
            result = cmp_func(golden_data_fp32, acl_data_fp32)
            csv_data[name][idx] = result
            csv_data[CMP_FLAG][idx] = True

    return csv_data
def csv_compare(csv_path_1, csv_path_2, output_path):
    # 读取两个CSV文件
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    # 合并两个DataFrame，根据data_id列进行合并，使用outer方式保留所有数据
    merged_df = df1.merge(df2, on='data_id', how='outer', suffixes=('', '_y'))
    merged_df = merged_df[[col for col in merged_df.columns if not col.endswith('_y')]]
    # 比对开始
    compare_all(merged_df)

    # 将合并后的数据保存到输出文件
    merged_df.to_csv(output_path, index=False)