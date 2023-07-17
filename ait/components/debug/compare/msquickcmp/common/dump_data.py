# coding=utf-8
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
"""
Function:
This class mainly involves generate dump data function.
"""
import os
import time

import numpy as np

from msquickcmp.common import utils
from msquickcmp.common.utils import logger
from msquickcmp.common.utils import AccuracyCompareException


MSACCUCMP_FILE_PATH =  "toolkit/tools/operator_cmp/compare/msaccucmp.py"


class DumpData(object):
    """
    Class for generate dump data.
    """

    def __init__(self):
        self.net_output = {}
        pass

    @staticmethod
    def _to_valid_name(name_str):
        return name_str.replace('.', '_').replace('/', '_')


    @staticmethod
    def _check_path_exists(input_path, extentions=None):
        input_path = os.path.realpath(input_path)
        if not os.path.exists(input_path):
            logger.error(f"path '{input_path}' not exists")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)

        if extentions and not any([input_path.endswith(extention) for extention in extentions]):
            logger.error(f"path '{input_path}' not ends with extention {extentions}")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)

        if not os.access(input_path, os.R_OK):
            logger.error(f"user doesn't have read permission to the file {input_path}.")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)

    def generate_dump_data(self):
        pass

    def get_net_output_info(self):
        return self.net_output

    def generate_inputs_data(self):
        pass

    def _generate_dump_data_file_name(self, name_str, node_id):
        return  ".".join([self._to_valid_name(name_str), str(node_id), str(round(time.time() * 1e6)), "npy"])

    def _check_input_data_path(self, input_path, inputs_tensor_info):
        if len(inputs_tensor_info) != len(input_path):
            logger.error("the number of model inputs tensor_info is not equal the number of "
                                  "inputs data, inputs tensor_info is: {}, inputs data is: {}".format(
                len(inputs_tensor_info), len(input_path)))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)

        for cur_path in input_path:
            if not os.path.exists(cur_path):
                logger.error(f"input data path '{cur_path}' not exists")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)

    def _generate_random_input_data(self, save_dir, names, shapes, dtypes):
        inputs_map = {}
        for index, (tensor_name, tensor_shape, tensor_dtype) in enumerate(zip(names, shapes, dtypes)):
            input_data = np.random.random(tensor_shape).astype(tensor_dtype)
            inputs_map[tensor_name] = input_data
            file_name = "input_" + str(index) + ".bin"
            input_data.tofile(os.path.join(save_dir, file_name))
            logger.info("save input file name: {}, shape: {}, dtype: {}".format(
                file_name, input_data.shape, input_data.dtype))
        return inputs_map

    def _read_input_data(self, input_pathes, names, shapes, dtypes):
        inputs_map = {}
        for input_path, name, shape, dtype in zip(input_pathes, names, shapes, dtypes):
            input_data = np.fromfile(input_path, dtype=dtype)
            if np.prod(input_data.shape) != np.prod(shape):
                cur = input_data.shape
                logger.error(f"input data shape not match, input_path: {input_path}, shape: {cur}, target: {shape}")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)
            input_data = input_data.reshape(shape)
            inputs_map[name] = input_data
            logger.info("load input file name: {}, shape: {}, dtype: {}".format(
                os.path.basename(input_path), input_data.shape, input_data.dtype))
        return inputs_map


def convert_bin_dump_data_to_npy(npu_dump_data_path, npu_net_output_data_path, cann_path):
    """
    Function Description:
        provide the interface for dump data conversion
    Exception Description:
        when invalid msaccucmp command throw exception
    """
    common_path = os.path.commonprefix([npu_dump_data_path, npu_net_output_data_path])
    npu_dump_data_path_diff = os.path.relpath(npu_dump_data_path, common_path)
    time_stamp_file_path = npu_dump_data_path_diff.split(os.path.sep)[1]
    convert_dir_path = npu_dump_data_path.replace(time_stamp_file_path, time_stamp_file_path+'_bin2npy')
    convert_dir_path = os.path.normpath(convert_dir_path)
    if not os.path.exists(convert_dir_path):
            os.makedirs(convert_dir_path)
    msaccucmp_command_file_path = os.path.join(cann_path, MSACCUCMP_FILE_PATH)
    python_version = sys.executable.split('/')[-1]
    bin2npy_cmd = [python_version, msaccucmp_command_file_path,
                    "convert", "-d", npu_dump_data_path, "-out", convert_dir_path]
    utils.execute_command(bin2npy_cmd)
    utils.logger.info("msaccucmp command line: %s " % " ".join(bin2npy_cmd))
    return convert_dir_path


def convert_bin_file_to_npy(bin_file_path, npy_dir_path, cann_path):
    """
    Function Description:
        convert a bin file to npy file.
    Parameter:
        bin_file_path: the path of the bin file needed to be converted to npy
        npy_dir_path: the dest dir to save the converted npy file
        cann_path: user or system cann_path for using msaccucmp.py
    """
    python_version = sys.executable.split('/')[-1]
    msaccucmp_command_file_path = os.path.join(cann_path, MSACCUCMP_FILE_PATH)
    bin2npy_cmd = [python_version, msaccucmp_command_file_path, "convert", "-d", bin_file_path, "-out", npy_dir_path]
    utils.logger.info("convert dump data: %s to npy file" % (bin_file_path))
    utils.execute_command(bin2npy_cmd)