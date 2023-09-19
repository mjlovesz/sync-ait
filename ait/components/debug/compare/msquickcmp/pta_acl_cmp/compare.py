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
from msquickcmp.pta_acl_cmp.constant import DATA_ID, PTA_DATA_PATH, ACL_DATA_PATH, \
    CMP_FLAG, CSV_HEADER, GOLDEN_DATA_PATH, GOLDEN_DTYPE, GOLDEN_SHAPE, CSV_GOLDEN_HEADER, \
    MODEL_INFER_TASK_ID, AIT_CMP_TASK_DIR, AIT_CMP_TASK, AIT_CMP_TASK_PID, ACL_DATA_MAP_FILE, TOKEN_ID
from msquickcmp.pta_acl_cmp.utils import compare_tensor, compare_all

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
        np.save(data_path, data_val)
        row_data = pd.DataFrame({
            DATA_ID: [data_id],
            PTA_DATA_PATH: [data_path],
            CMP_FLAG: [False]
        })
        csv_data = pd.concat([csv_data, row_data], ignore_index=True)
    else:
        index = mapping_data.index.values[0]
        np.save(data_path, data_val)
        csv_data[PTA_DATA_PATH][index] = data_path

        # 对应的acl_data存在时，触发比对
        csv_data = compare_tensor(csv_data=csv_data)

    return csv_data


def save_acl_data(csv_data, data_id, data_val, data_path):
    if data_val is None:
        return csv_data

    data_val = data_val.cpu().numpy()
    mapping_data = csv_data[csv_data[DATA_ID] == data_id]
    if mapping_data.empty:
        np.save(data_path, data_val)
        row_data = pd.DataFrame({
            DATA_ID: [data_id],
            ACL_DATA_PATH: [data_path],
            CMP_FLAG: [False]
        })
        csv_data = pd.concat([csv_data, row_data], ignore_index=True)
    else:
        index = mapping_data.index.values[0]
        np.save(data_path, data_val)
        csv_data[ACL_DATA_PATH][index] = data_path

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
            data = save_golden_data(csv_data=data, data_id=data_id, data_val=data_val, \
                                    data_path=golden_data_path, token_id=token_id)
        elif tensor_path:  # low-level
            token_tensor_path = os.path.join(str(token_id), tensor_path)
            write_acl_map_file(token_tensor_path)
            tensor_path = os.path.join(os.getenv("ACLTRANSFORMER_HOME_PATH"), "tensors",
                                       f"thread_{str(pid)}", str(token_id), tensor_path)
            data = save_golden_dump_tensor(csv_data=data, data_id=data_id, \
                                           tensor_path=tensor_path, token_id=token_id)
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


def save_golden_data(csv_data, data_id, data_val, data_path, token_id):
    if data_val is None:
        return csv_data

    data_val = data_val.cpu().numpy()
    data_path = os.path.realpath(data_path)
    mapping_data = csv_data[csv_data[DATA_ID] == data_id]
    if mapping_data.empty:
        data_val.tofile(data_path)
        row_data = pd.DataFrame({
            TOKEN_ID: [token_id],
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


def save_golden_dump_tensor(csv_data, data_id, tensor_path, token_id):
    mapping_data = csv_data[csv_data[DATA_ID] == data_id]
    if mapping_data.empty:
        row_data = pd.DataFrame({
            TOKEN_ID: [token_id],
            DATA_ID: [data_id], 
            GOLDEN_DATA_PATH: [tensor_path], 
            CMP_FLAG: [False]})
        csv_data = pd.concat([csv_data, row_data], ignore_index=True)
    else:
        index = mapping_data.index.values[0]
        csv_data[GOLDEN_DATA_PATH][index] = tensor_path

    return csv_data


def pure_save_acl_data(csv_data, data_id, data_val, data_path, token_id=token_id):
    if data_val is None:
        return csv_data

    data_val = data_val.cpu().numpy()
    mapping_data = csv_data[csv_data[DATA_ID] == data_id]
    if mapping_data.empty:
        np.save(data_path, data_val)
        row_data = pd.DataFrame({
            TOKEN_ID: [token_id],
            DATA_ID: [data_id],
            ACL_DATA_PATH: [data_path],
            CMP_FLAG: [False]
        })
        csv_data = pd.concat([csv_data, row_data], ignore_index=True)
    else:
        index = mapping_data.index.values[0]
        np.save(data_path, data_val)
        csv_data[ACL_DATA_PATH][index] = data_path


    return csv_data


def csv_compare(csv_path_1, csv_path_2, output_path):
    # 读取两个CSV文件
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    # 合并两个DataFrame，根据data_id列进行合并
    merged_df = df1.set_index('data_id').combine_first(df2.set_index('data_id')).reset_index()

    # 比对开始
    compare_all(merged_df)

    # 将合并后的数据保存到输出文件
    merged_df.to_csv(output_path, index=False)