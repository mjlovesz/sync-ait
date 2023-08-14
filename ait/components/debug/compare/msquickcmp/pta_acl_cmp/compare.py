import os
import subprocess
import time
import inspect

import pandas as pd
import numpy as np
import torch

from msquickcmp.pta_acl_cmp.cmp_algorithm import cmp_alg_map
from msquickcmp.common.utils import execute_command

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
PTA_STACK = "pta_stack"
ACL_DTYPE = "acl_dtype"
ACL_STACK = "acl_stack"
CMP_FLAG = "cmp_flag"
CSV_HEADER = [DATA_ID, PTA_DATA_PATH, PTA_DTYPE, PTA_STACK, ACL_DATA_PATH, ACL_DTYPE, ACL_STACK, CMP_FLAG]
CSV_HEADER.extend(list(cmp_alg_map.keys()))

token_counts = 1


def set_task_id():
    pid = os.getpid()
    global token_counts
    token_counts += 1
    task_id = str(pid) + "_" + str(token_counts)
    cmd = ["export", "AIT_CMP_TASK_ID"+"="+task_id]
    execute_command(cmd)


def gen_id():
    return "data_" + str(time.time())


def set_label(data_src: str, data_id: str, data_val=None, tensor_path=None):
    stacks = inspect.stack()
    stack_line = stacks[1][1] + ":" + str(stacks[1][2])
    # task_id = os.getenv("CMP_TASK_ID")
    task_id = str(0)
    csv_dir = os.path.join(".", task_id)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_path = os.path.join(".", task_id + "_cmp_result.csv")

    dump_data_dir = "cmp_dump_data"
    if not os.path.exists(dump_data_dir):
        os.mkdir(dump_data_dir)

    pta_data_dir = os.path.join(".", dump_data_dir, "pta_tensor")
    acl_data_dir = os.path.join(".", dump_data_dir, "acl_tensor")
    if not os.path.exists(pta_data_dir):
        os.makedirs(pta_data_dir)

    if not os.path.exists(acl_data_dir):
        os.makedirs(acl_data_dir)

    if not os.path.exists(csv_path):
        data = pd.DataFrame(columns=CSV_HEADER, index=[0])
    else:
        data = pd.read_csv(csv_path, header=0)

    mapping_data = data[data[DATA_ID] == data_id]
    if mapping_data.empty:
        if data_src == PTA and data_val is not None:
            pta_data_path = os.path.join(pta_data_dir, data_id + '_tensor.bin')
            data_val.cpu().numpy().tofile(pta_data_path)
            row_data = pd.DataFrame({
                DATA_ID: [data_id],
                PTA_DATA_PATH: [pta_data_path],
                PTA_DTYPE: [data_val.cpu().numpy().dtype],
                PTA_STACK: [stack_line],
                CMP_FLAG: [False]
            })
            data = pd.concat([data, row_data], ignore_index=True)
            data = compare_tensor(csv_data=data)
        elif data_src == ACL and data_val is not None:
            acl_data_path = os.path.join(acl_data_dir, data_id + '_tensor.bin')
            data_val.cpu().numpy().tofile(acl_data_path)
            row_data = pd.DataFrame({
                DATA_ID: [data_id],
                ACL_DATA_PATH: [acl_data_path],
                ACL_DTYPE: [data_val.cpu().numpy().dtype],
                ACL_STACK: [stack_line],
                CMP_FLAG: [False]
            })
            data = pd.concat([data, row_data], ignore_index=True)
            data = compare_tensor(csv_data=data)
        elif data_src == ACL and tensor_path:
            row_data = pd.DataFrame({DATA_ID: [data_id], ACL_DATA_PATH: [tensor_path], CMP_FLAG: [False]})
            data = pd.concat([data, row_data], ignore_index=True)
    else:
        index = mapping_data.index.values[0]
        if data_src == PTA and data_val is not None:
            pta_data_path = os.path.join(pta_data_dir, data_id + '_tensor.bin')
            data_val.cpu().numpy().tofile(pta_data_path)
            data[PTA_DATA_PATH][index] = pta_data_path
            data[PTA_DTYPE][index] = data_val.cpu().numpy().dtype
            data[PTA_STACK][index] = stack_line
            data = compare_tensor(csv_data=data)
        elif data_src == ACL and tensor_path:
            data[ACL_DATA_PATH][index] = tensor_path
            data[ACL_STACK][index] = stack_line
        elif data_src == ACL and data_val is not None:
            acl_data_path = os.path.join(acl_data_dir, data_id + '_tensor.bin')
            data_val.cpu().numpy().tofile(acl_data_path)
            data[ACL_DATA_PATH][index] = acl_data_path
            data[ACL_DTYPE][index] = data_val.cpu().numpy().dtype
            data[ACL_STACK][index] = stack_line
            # 触发精度比对
            data = compare_tensor(csv_data=data)

    data.to_csv(csv_path, index=False)


def compare_tensor(csv_data: pd.DataFrame):
    csv_data.fillna(value="", inplace=True)
    data = csv_data[csv_data[CMP_FLAG] == False]
    if data.empty:
        return
    for idx in data.index:
        pta_data_path = data[PTA_DATA_PATH][idx]
        pta_dtype = data[PTA_DTYPE][idx]
        acl_data_path = data[ACL_DATA_PATH][idx]
        acl_dtype = data[ACL_DTYPE][idx]
        if pta_data_path and acl_data_path:
            pta_data = np.fromfile(pta_data_path, pta_dtype)
            if acl_dtype:
                acl_data = np.fromfile(acl_data_path, acl_dtype)
            else:
                acl_data = read_acl_transformer_data(acl_data_path)

            for name, cmp_func in cmp_alg_map.items():
                result = cmp_func(pta_data, acl_data)
                csv_data[name][idx] = result

            csv_data[CMP_FLAG][idx] = True
    return csv_data


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
    if file_path.endswith(".bin"):
        bin = TensorBinFile(file_path)
        return bin.get_tensor()
    else:
        try:
            return list(torch.load(file_path).state_dict().values())[0]
        except:
            return torch.load(file_path)
