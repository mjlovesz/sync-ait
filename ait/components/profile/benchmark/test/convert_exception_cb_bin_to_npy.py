# Copyright (c) 2023 Huawei Technologies Co., Ltd
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
import re
import sys
import logging
import argparse
import numpy as np

num_to_aclFormat = {
    -1: "ACL_FORMAT_UNDEFINED",
    0: "ACL_FORMAT_NCHW",
    1: "ACL_FORMAT_NHWC",
    2: "ACL_FORMAT_ND",
    3: "ACL_FORMAT_NC1HWC0",
    4: "ACL_FORMAT_FRACTAL_Z",
    12: "ACL_FORMAT_NC1HWC0_C04",
    27: "ACL_FORMAT_NDHWC",
    29: "ACL_FORMAT_FRACTAL_NZ",
    30: "ACL_FORMAT_NCDHW",
    32: "ACL_FORMAT_NDC1HWC0",
    33: "ACL_FRACTAL_Z_3D"}

num_to_aclDataType = {
    -1: "ACL_DT_UNDEFINED",
    0: "ACL_FLOAT",
    1: "ACL_FLOAT16",
    2: "ACL_INT8",
    3: "ACL_INT32",
    4: "ACL_UINT8",
    6: "ACL_INT16",
    7: "ACL_UINT16",
    8: "ACL_UINT32",
    9: "ACL_INT64",
    10: "ACL_UINT64",
    11: "ACL_DOUBLE",
    12: "ACL_BOOL",
    13: "ACL_STRING",
    16: "ACL_COMPLEX64",
    17: "ACL_COMPLEX128"
}

acl_type_to_numpy_type = {
        "ACL_INT8": np.int8,
        "ACL_UINT8": np.uint8,
        "ACL_INT16": np.int16,
        "ACL_UINT16": np.uint16,
        "ACL_INT32": np.int32,
        "ACL_UINT32": np.uint32,
        "ACL_INT64": np.int64,
        "ACL_UINT64": np.uint64,
        "ACL_FLOAT16": np.float16,
        "ACL_FLOAT": np.float32,
        "ACL_DOUBLE": np.double,
        "ACL_BOOL": np.bool_,
    }

EXCEPTION_FILE_NAME_KEY_WORD = "exception_cb"


def get_format_dtype_shape(file_path):
    """
    bin file name like:
        exception_cb_index_0_input_0_format_2_dtype_1_shape_30522x768.bin
    """
    pattern = r'format_(.*?)_dtype'
    res = re.findall(pattern, file_path)
    format_num = int(res[0])

    pattern = r'dtype_(.*?)_shape'
    res = re.findall(pattern, file_path)
    dtype_num = int(res[0])

    pattern = r'shape_(.*?).bin'
    res = re.findall(pattern, file_path)
    shapestr = res[0]
    str_list = shapestr.split('x')
    shape = []
    if len(str_list) > 0 and str_list[0] != '':
        shape = [ int(i) for i in str_list]

    return format_num, dtype_num, shape


def parse_bin_file(bin_file):
    # 获取单纯的文件名
    bin_file_path = os.path.dirname(os.path.abspath(bin_file))
    bin_file_name = os.path.basename(bin_file)
    format_num, dtype_num, shape = get_format_dtype_shape(bin_file_name)

    logging.info("bin_file:{bin_file} format: {num_to_aclFormat.get(format_num)} dtype:
                 {num_to_aclDataType.get(dtype_num)} shape: {shape}")
    # 输出npy文件
    file_name = bin_file_name.strip(".bin")
    np_dtype = acl_type_to_numpy_type.get(num_to_aclDataType.get(dtype_num))
    data = np.fromfile(bin_file, dtype=np_dtype)
    if len(shape) == 0:
        logging.info("warning get shape failed convert [-1]")
        shape = [ -1 ]
    ndata = data.reshape(shape)
    npy_file_name = file_name + ".npy"
    npy_file = os.path.join(bin_file_path, npy_file_name)
    np.save(npy_file, ndata)
    logging.info("out npy dtype: {np_dtype} shape: {ndata.shape} out_file: {npy_file}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="exception cb bin file path.Support single file and folder path.")

    input_args = parser.parse_args()
    return input_args

if __name__ == '__main__':
    args = get_args()
    if args.input is  None or not os.path.exists(args.input):
        logging.info("bad parameters. lack of input parameter or bin file is not exist.")
        sys.exit(1)

    bin_files = []
    if os.path.isdir(args.input):
        path_list = os.listdir(args.input)
        for path in path_list:
            if os.path.isdir(path) :
                continue
            if EXCEPTION_FILE_NAME_KEY_WORD in path and path.endswith(".bin"):
                bin_files.append(os.path.join(args.input, path))
    else:
        if EXCEPTION_FILE_NAME_KEY_WORD in args.input and args.input.endswith(".bin"):
            bin_files.append(args.input)

    if len(bin_files) == 0:
        logging.info("bad parameters. No suitable exception file")
        sys.exit(1)

    for each_bin_file in bin_files:
        parse_bin_file(each_bin_file)

