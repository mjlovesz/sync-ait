import os

import torch
import csv
import pandas as pd
import numpy as np

from msquickcmp.pta_acl_cmp.cmp_algorithm import cmp_alg_map

PTA = "pta"
ACL = "acl"
DATA_ID = 'data_id'
PTA_DATA_PATH = 'pta_data_path'
ACL_DATA_PATH = 'acl_data_path'
PTA_DTYPE = "pta_dtype"
ACL_DTYPE = "acl_dtype"
CMP_FLAG = "cmp_flag"
CSV_HEADER = [DATA_ID, PTA_DATA_PATH, PTA_DTYPE, ACL_DATA_PATH, ACL_DTYPE, CMP_FLAG]
CSV_HEADER.extend(list(cmp_alg_map.keys()))


def set_label(data_src: str, data_id: str, data_val: torch.Tensor = None, tensor_path: str = None):
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
    if not mapping_data.empty:
        index = mapping_data.index.values[0]
        if data_src == PTA and data_val is not None:
            pta_data_path = os.path.join(pta_data_dir, data_id + '_tensor.bin')
            data_val.cpu().numpy().tofile(pta_data_path)
            data[PTA_DATA_PATH][index] = pta_data_path
            data[PTA_DTYPE][index] = data_val.cpu().numpy().dtype
            data = compare_tensor(csv_data=data)
        elif data_src == ACL and tensor_path:
            data[ACL_DATA_PATH][index] = tensor_path
        elif data_src == ACL and data_val is not None:
            acl_data_path = os.path.join(acl_data_dir, data_id + '_tensor.bin')
            data_val.cpu().numpy().tofile(acl_data_path)
            data[ACL_DATA_PATH][index] = acl_data_path
            data[ACL_DTYPE][index] = data_val.cpu().numpy().dtype
            # 触发精度比对
            data = compare_tensor(csv_data=data)
    else:
        if data_src == PTA and data_val is not None:
            pta_data_path = os.path.join(pta_data_dir, data_id + '_tensor.bin')
            data_val.cpu().numpy().tofile(pta_data_path)
            row_data = pd.DataFrame({
                DATA_ID: [data_id],
                PTA_DATA_PATH: [pta_data_path],
                PTA_DTYPE: [data_val.cpu().numpy().dtype],
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
                CMP_FLAG: [False]
            })
            data = pd.concat([data, row_data], ignore_index=True)
            data = compare_tensor(csv_data=data)
        elif data_src == ACL and tensor_path:
            row_data = pd.DataFrame({DATA_ID: [data_id], ACL_DATA_PATH: [tensor_path], CMP_FLAG: [False]})
            data = pd.concat([data, row_data], ignore_index=True)

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

def read_acl_transformer_data(data_path):
    pass
