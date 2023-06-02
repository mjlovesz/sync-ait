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

from msquickcmp.common.utils import logger
from msquickcmp.common.utils import AccuracyCompareException


class DumpData(object):
    """
    Class for generate dump data.
    """

    def __init__(self):
        self.net_output = {}
        pass

    def generate_dump_data(self):
        """
        Function Description:
            generate dump data
        """
        pass

    def get_net_output_info(self):
        """
        get_net_output_info
        """
        return self.net_output

    def generate_inputs_data(self):
        """
        Function Description:
            generate inputs data
        """
        pass

    def _generate_dump_data_file_name(self, name_str, node_id):
        name_str = name_str.replace('.', '_').replace('/', '_')
        return  ".".join([name_str, str(node_id), str(round(time.time() * 1e6)), "npy"])

    def _check_input_data_path(self, input_path, inputs_tensor_info):
        if len(inputs_tensor_info) != len(input_path):
            logger.error("the number of model inputs tensor_info is not equal the number of "
                                  "inputs data, inputs tensor_info is: {}, inputs data is: {}".format(
                len(inputs_tensor_info), len(input_path)))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)

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
            input_data = np.fromfile(input_path, dtype=dtype).reshape(shape)
            inputs_map[name] = input_data
            logger.info("load input file name: {}, shape: {}, dtype: {}".format(
                os.path.basename(input_path), input_data.shape, input_data.dtype))
        return inputs_map
